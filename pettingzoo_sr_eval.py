"""
PettingZoo Integration for Stackelberg Regret Evaluation
Uses MPE simple_spread_v3: 3 agents cooperate to cover 3 landmarks.

Agent 0 = leader, Agents 1 & 2 = followers.
We run three scenarios:
  (A) Aligned    – all agents use the same trained policy
  (B) Misaligned – follower 1 uses a random/adversarial policy
  (C) Fully Random – all agents act randomly (baseline)
"""

import os
os.environ["SDL_VIDEODRIVER"] = "dummy"

import torch
import torch.nn as nn
import numpy as np
from pettingzoo.mpe import simple_spread_v3

from stackelberg_regret_eval import (
    AgentStep,
    EpisodeBuffer,
    StackelbergRegretEvaluator,
    StackelbergRegretResult,
)


# ─────────────────────────────────────────────────────────────
# Agent Network: observation -> (latent, Q-values)
# ─────────────────────────────────────────────────────────────

class AgentNetwork(nn.Module):
    """
    Small encoder + Q-head that produces a latent representation
    and Q-values from raw observations. Serves as both the
    representation extractor and the decision-maker.
    """
    def __init__(self, obs_dim: int, latent_dim: int, n_actions: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
        )
        self.q_head = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, n_actions),
        )

    def forward(self, obs: torch.Tensor):
        z = self.encoder(obs)
        q = self.q_head(z)
        return z, q


# ─────────────────────────────────────────────────────────────
# Training: Simple online Q-learning to get non-random policies
# ─────────────────────────────────────────────────────────────

def train_agents(
    obs_dim: int,
    latent_dim: int,
    n_actions: int,
    n_episodes: int = 300,
    gamma: float = 0.99,
    lr: float = 3e-4,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
) -> dict[str, AgentNetwork]:
    """
    Trains 3 cooperative agents on simple_spread via independent Q-learning.
    Returns a dict mapping agent name -> trained AgentNetwork.
    """
    env = simple_spread_v3.env(N=3, max_cycles=25, render_mode=None)

    networks = {}
    optimizers = {}
    agent_names = [f"agent_{i}" for i in range(3)]

    for name in agent_names:
        net = AgentNetwork(obs_dim, latent_dim, n_actions)
        networks[name] = net
        optimizers[name] = torch.optim.Adam(net.parameters(), lr=lr)

    loss_fn = nn.SmoothL1Loss()
    episode_rewards = []

    for ep in range(n_episodes):
        epsilon = epsilon_start - (epsilon_start - epsilon_end) * (ep / n_episodes)
        env.reset()

        transitions = {name: [] for name in agent_names}
        total_reward = 0.0

        for agent_name in env.agent_iter():
            obs, reward, terminated, truncated, info = env.last()
            total_reward += reward

            if terminated or truncated:
                env.step(None)
                continue

            obs_t = torch.tensor(obs, dtype=torch.float32)
            z, q_vals = networks[agent_name](obs_t.unsqueeze(0))
            q_vals = q_vals.squeeze(0)

            if np.random.random() < epsilon:
                action = env.action_space(agent_name).sample()
            else:
                action = q_vals.argmax().item()

            transitions[agent_name].append((obs_t, action, reward, q_vals))
            env.step(action)

        # Simple MC-return Q-learning update for each agent
        for name in agent_names:
            if not transitions[name]:
                continue
            G = 0.0
            for obs_t, action, reward, q_vals in reversed(transitions[name]):
                G = reward + gamma * G
                target = q_vals.detach().clone()
                target[action] = G
                _, q_pred = networks[name](obs_t.unsqueeze(0))
                loss = loss_fn(q_pred.squeeze(0), target)
                optimizers[name].zero_grad()
                loss.backward()
                optimizers[name].step()

        episode_rewards.append(total_reward)

    env.close()

    avg_last50 = np.mean(episode_rewards[-50:])
    print(f"  Training done. Avg reward (last 50 eps): {avg_last50:.2f}")

    return networks


# ─────────────────────────────────────────────────────────────
# Episode rollout: collect data into EpisodeBuffer
# ─────────────────────────────────────────────────────────────

def run_episode(
    networks: dict[str, AgentNetwork],
    leader_name: str,
    misaligned_follower: str | None = None,
    latent_dim: int = 16,
) -> tuple[EpisodeBuffer, list[torch.Tensor], list[torch.Tensor]]:
    """
    Runs one episode of simple_spread.

    If misaligned_follower is set, that agent takes random actions
    (simulating a follower that has drifted from the intended policy).

    Returns the EpisodeBuffer, intended BR Q-values, and observed Q-values.
    """
    env = simple_spread_v3.env(N=3, max_cycles=25, render_mode=None)
    env.reset()

    buffer = EpisodeBuffer()
    leader_q_intended = []
    leader_q_observed = []

    step_data = {}
    timestep = 0

    for agent_name in env.agent_iter():
        obs, reward, terminated, truncated, info = env.last()

        if terminated or truncated:
            env.step(None)
            continue

        obs_t = torch.tensor(obs, dtype=torch.float32)

        with torch.no_grad():
            z, q_vals = networks[agent_name](obs_t.unsqueeze(0))
            z = z.squeeze(0)
            q_vals = q_vals.squeeze(0)

        is_misaligned = (agent_name == misaligned_follower)

        if is_misaligned:
            action = env.action_space(agent_name).sample()
            q_observed = q_vals.clone()
            q_observed[action] -= 0.5
        else:
            action = q_vals.argmax().item()
            q_observed = q_vals

        agent_id = int(agent_name.split("_")[1])
        agent_step = AgentStep(
            agent_id=agent_id,
            state=obs_t,
            latent=z,
            action=torch.tensor(action),
            q_values=q_vals,
            value=q_vals.max(),
        )

        if agent_name == leader_name:
            buffer.add_leader(agent_step, z.clone())
            leader_q_intended.append(q_vals.clone())
            leader_q_observed.append(q_observed.clone())
        else:
            buffer.add_follower(agent_step)

        env.step(action)

    env.close()
    return buffer, leader_q_intended, leader_q_observed


# ─────────────────────────────────────────────────────────────
# Collect aligned calibration data
# ─────────────────────────────────────────────────────────────

def collect_aligned_latents(
    networks: dict[str, AgentNetwork],
    leader_name: str,
    n_episodes: int = 30,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Runs several aligned episodes (no misalignment) and collects
    (leader_latent, avg_follower_latent) pairs for calibration.
    """
    leader_lats = []
    follower_lats = []

    for _ in range(n_episodes):
        buf, _, _ = run_episode(networks, leader_name, misaligned_follower=None)

        T = len(buf.leader_steps)
        follower_ids = list(buf.follower_steps.keys())
        T_f = min(len(buf.follower_steps[fid]) for fid in follower_ids)
        T_use = min(T, T_f)

        for t in range(T_use):
            leader_lats.append(buf.leader_steps[t].latent)
            avg_f = torch.stack([
                buf.follower_steps[fid][t].latent for fid in follower_ids
            ]).mean(dim=0)
            follower_lats.append(avg_f)

    return torch.stack(leader_lats), torch.stack(follower_lats)


# ─────────────────────────────────────────────────────────────
# Pretty-print results
# ─────────────────────────────────────────────────────────────

def print_result(label: str, result: StackelbergRegretResult):
    print(f"\n{'=' * 56}")
    print(f"  {label}")
    print(f"{'=' * 56}")
    print(f"  Behavioral SR    : {result.behavioral_sr:+.4f}")
    print(f"  Latent SR        : {result.latent_sr:.4f}")
    print(f"  Nash Deviation   : {result.nash_deviation:.4f}")
    print(f"  Latent Cone Rate : {result.latent_cone_rate:.4f}")
    print(f"  Composite SR     : {result.composite_sr:.4f}")
    print(f"  Is Aligned?      : {result.is_aligned}")
    print(f"  Per-Follower Best-Response Gap:")
    for fid, brg in result.per_follower_brg.items():
        flag = " <-- MISALIGNED" if brg > 0.3 else ""
        print(f"    Follower {fid}: BRG = {brg:.4f}{flag}")
    print(f"{'=' * 56}")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    LATENT_DIM = 16
    N_ACTIONS = 5  # simple_spread has 5 discrete actions: [no-op, left, right, down, up]
    LEADER = "agent_0"

    # Probe observation space size
    probe_env = simple_spread_v3.env(N=3, max_cycles=25, render_mode=None)
    probe_env.reset()
    obs_dim = probe_env.observation_space(LEADER).shape[0]
    probe_env.close()
    print(f"Environment: simple_spread_v3 (3 agents, obs_dim={obs_dim}, actions={N_ACTIONS})")

    # ── Step 1: Train cooperative agents ──────────────────────
    print("\n[1/4] Training agents (300 episodes of Q-learning)...")
    networks = train_agents(obs_dim, LATENT_DIM, N_ACTIONS, n_episodes=300)

    # ── Step 2: Calibrate evaluator on aligned episodes ───────
    print("\n[2/4] Collecting aligned calibration data (30 episodes)...")
    aligned_leader_lats, aligned_follower_lats = collect_aligned_latents(
        networks, LEADER, n_episodes=30
    )
    print(f"  Collected {aligned_leader_lats.shape[0]} aligned latent pairs")

    evaluator = StackelbergRegretEvaluator(
        latent_dim=LATENT_DIM,
        n_actions=N_ACTIONS,
        gamma=0.99,
        sigma_scale=3.0,
        sr_threshold=0.15,
    )
    evaluator.fit_aligned_cone(aligned_leader_lats)
    evaluator.fit_projection_head(aligned_leader_lats, aligned_follower_lats, epochs=100)

    # ── Step 3: Evaluate three scenarios ──────────────────────
    print("\n[3/4] Running evaluation episodes...")

    N_EVAL = 10
    results_aligned = []
    results_misaligned = []
    results_random = []

    # Build a "random" network for the fully-random scenario
    random_networks = {}
    for name in networks:
        random_networks[name] = AgentNetwork(obs_dim, LATENT_DIM, N_ACTIONS)

    for i in range(N_EVAL):
        # (A) Aligned: all agents use trained policy
        buf_a, iq_a, oq_a = run_episode(networks, LEADER, misaligned_follower=None)
        if buf_a.leader_steps and iq_a:
            results_aligned.append(evaluator.evaluate(buf_a, iq_a, oq_a))

        # (B) Misaligned: follower 1 uses random actions
        buf_b, iq_b, oq_b = run_episode(networks, LEADER, misaligned_follower="agent_1")
        if buf_b.leader_steps and iq_b:
            results_misaligned.append(evaluator.evaluate(buf_b, iq_b, oq_b))

        # (C) Fully random: untrained networks
        buf_c, iq_c, oq_c = run_episode(random_networks, LEADER, misaligned_follower=None)
        if buf_c.leader_steps and iq_c:
            results_random.append(evaluator.evaluate(buf_c, iq_c, oq_c))

    # ── Step 4: Aggregate and report ──────────────────────────
    print("\n[4/4] Results")

    def avg_result(results: list[StackelbergRegretResult]) -> StackelbergRegretResult:
        n = len(results)
        if n == 0:
            return StackelbergRegretResult(0, 0, 0, 0, 0)
        all_fids = set()
        for r in results:
            all_fids.update(r.per_follower_brg.keys())
        avg_brg = {}
        for fid in all_fids:
            vals = [r.per_follower_brg.get(fid, 0.0) for r in results]
            avg_brg[fid] = np.mean(vals)

        composite = np.mean([r.composite_sr for r in results])
        return StackelbergRegretResult(
            behavioral_sr=float(np.mean([r.behavioral_sr for r in results])),
            latent_sr=float(np.mean([r.latent_sr for r in results])),
            nash_deviation=float(np.mean([r.nash_deviation for r in results])),
            latent_cone_rate=float(np.mean([r.latent_cone_rate for r in results])),
            composite_sr=float(composite),
            per_follower_brg={k: float(v) for k, v in avg_brg.items()},
            is_aligned=bool(composite < evaluator.sr_threshold),
        )

    print_result(
        f"SCENARIO A: All Aligned (avg over {len(results_aligned)} episodes)",
        avg_result(results_aligned),
    )
    print_result(
        f"SCENARIO B: Follower 1 Misaligned (avg over {len(results_misaligned)} episodes)",
        avg_result(results_misaligned),
    )
    print_result(
        f"SCENARIO C: Fully Random / Untrained (avg over {len(results_random)} episodes)",
        avg_result(results_random),
    )

    # Drift analysis across the aligned episodes
    if len(results_aligned) >= 3:
        drift = evaluator.track_drift(results_aligned, window=3)
        print(f"\n{'─' * 56}")
        print("  Drift Tracking (Aligned, window=3):")
        for key, vals in drift.items():
            trend = "stable" if abs(vals[-1] - vals[0]) < 0.05 else (
                "increasing" if vals[-1] > vals[0] else "decreasing"
            )
            print(f"    {key:>16s}: first={vals[0]:.4f}  last={vals[-1]:.4f}  trend={trend}")
        print(f"{'─' * 56}")


if __name__ == "__main__":
    main()
