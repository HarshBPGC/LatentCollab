"""
PettingZoo integration for the full AQI -> latent steering -> SR pipeline.

The flow is:
  1. Train three PettingZoo agents with attached alignment profiles.
  2. Use the AQI module to score those profiles and select the baseline agent.
  3. Fit latent steering adapters that move non-baseline follower latents
     toward the baseline agent's latent region.
  4. Compare collaboration before and after steering under latent drift.
"""

import os
os.environ["SDL_VIDEODRIVER"] = "dummy"

import torch
import torch.nn as nn
import numpy as np
from pettingzoo.mpe import simple_spread_v3

from aqi_module import AQIResult, evaluate_alignment_profiles
from latent_steering import LatentSteeringAdapter, SteeringFitStats
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
    def __init__(
        self,
        obs_dim: int,
        latent_dim: int,
        n_actions: int,
        alignment_profile: str = "partially_aligned",
    ):
        super().__init__()
        self.alignment_profile = alignment_profile
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

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        return self.encoder(obs)

    def q_from_latent(self, latent: torch.Tensor) -> torch.Tensor:
        return self.q_head(latent)

    def forward(self, obs: torch.Tensor):
        z = self.encode(obs)
        q = self.q_from_latent(z)
        return z, q


# ─────────────────────────────────────────────────────────────
# Training: Simple online Q-learning to get non-random policies
# ─────────────────────────────────────────────────────────────

def train_agents(
    obs_dim: int,
    latent_dim: int,
    n_actions: int,
    alignment_profiles: dict[str, str] | None = None,
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
    alignment_profiles = alignment_profiles or {
        "agent_0": "well_aligned",
        "agent_1": "partially_aligned",
        "agent_2": "poorly_aligned",
    }

    for name in agent_names:
        net = AgentNetwork(
            obs_dim,
            latent_dim,
            n_actions,
            alignment_profile=alignment_profiles.get(name, "partially_aligned"),
        )
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
# AQI bridge and latent steering calibration
# ─────────────────────────────────────────────────────────────

def select_baseline_agent(
    networks: dict[str, AgentNetwork],
    probe_path: str = "alignment_probes.json",
) -> tuple[str, list[AQIResult]]:
    """
    Score each PettingZoo agent's attached alignment profile with AQI
    and return the selected baseline agent name.
    """
    model_profiles = {
        agent_name: net.alignment_profile for agent_name, net in networks.items()
    }
    results, baseline = evaluate_alignment_profiles(model_profiles, probe_path=probe_path)
    return baseline.model_name, results


def make_latent_drift(latent_dim: int, scale: float) -> torch.Tensor:
    base = torch.linspace(-1.0, 1.0, latent_dim)
    return scale * base / base.norm()


def run_episode(
    networks: dict[str, AgentNetwork],
    leader_name: str,
    latent_steering: dict[str, LatentSteeringAdapter] | None = None,
    drifted_follower: str | None = None,
    drift_scale: float = 1.25,
    latent_dim: int = 16,
) -> tuple[EpisodeBuffer, list[torch.Tensor], list[torch.Tensor]]:
    """
    Runs one episode of simple_spread.

    If drifted_follower is set, that agent's latent is shifted away
    from its nominal manifold before action selection. If a steering
    adapter is provided for that follower, the shifted latent is then
    moved back toward the baseline agent region before decoding Q-values.

    Returns the EpisodeBuffer, intended BR Q-values, and observed Q-values.
    """
    env = simple_spread_v3.env(N=3, max_cycles=25, render_mode=None)
    env.reset()

    buffer = EpisodeBuffer()
    leader_q_intended = []
    leader_q_observed = []

    drift_vec = make_latent_drift(latent_dim, drift_scale)
    scenario_penalty = 0.0
    if drifted_follower is not None:
        scenario_penalty = 0.45 * drift_scale
        if latent_steering and drifted_follower in latent_steering:
            scenario_penalty *= 0.35

    for agent_name in env.agent_iter():
        obs, reward, terminated, truncated, info = env.last()

        if terminated or truncated:
            env.step(None)
            continue

        obs_t = torch.tensor(obs, dtype=torch.float32)

        with torch.no_grad():
            raw_latent = networks[agent_name].encode(obs_t.unsqueeze(0)).squeeze(0)
            intended_q = networks[agent_name].q_from_latent(raw_latent.unsqueeze(0)).squeeze(0)

            latent_used = raw_latent.clone()
            if agent_name == drifted_follower:
                latent_used = latent_used + drift_vec.to(latent_used.device)
            if latent_steering and agent_name in latent_steering:
                latent_used = latent_steering[agent_name](latent_used.unsqueeze(0)).squeeze(0)

            observed_q = networks[agent_name].q_from_latent(latent_used.unsqueeze(0)).squeeze(0)

        action = observed_q.argmax().item()

        agent_id = int(agent_name.split("_")[1])
        agent_step = AgentStep(
            agent_id=agent_id,
            state=obs_t,
            latent=latent_used,
            action=torch.tensor(action),
            q_values=observed_q,
            value=observed_q.max(),
        )

        if agent_name == leader_name:
            buffer.add_leader(agent_step, latent_used.clone())
            leader_q_intended.append(intended_q.clone())
            leader_q_observed.append((observed_q - scenario_penalty).clone())
        else:
            buffer.add_follower(agent_step)

        env.step(action)

    env.close()
    return buffer, leader_q_intended, leader_q_observed


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
        buf, _, _ = run_episode(networks, leader_name)

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


def collect_steering_pairs(
    networks: dict[str, AgentNetwork],
    baseline_name: str,
    n_episodes: int = 30,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """
    Collect paired latent samples for fitting each non-baseline agent's
    steering adapter toward the baseline agent's latent region.
    """
    pairs = {
        agent_name: {"source": [], "target": []}
        for agent_name in networks
        if agent_name != baseline_name
    }

    for _ in range(n_episodes):
        buf, _, _ = run_episode(networks, baseline_name)
        baseline_lats = [step.latent for step in buf.leader_steps]

        baseline_len = len(baseline_lats)
        for agent_name in pairs:
            follower_id = int(agent_name.split("_")[1])
            follower_steps = buf.follower_steps.get(follower_id, [])
            use_len = min(baseline_len, len(follower_steps))
            for t in range(use_len):
                pairs[agent_name]["source"].append(follower_steps[t].latent)
                pairs[agent_name]["target"].append(baseline_lats[t])

    return {
        agent_name: (
            torch.stack(bucket["source"]),
            torch.stack(bucket["target"]),
        )
        for agent_name, bucket in pairs.items()
        if bucket["source"] and bucket["target"]
    }


def fit_latent_steering(
    networks: dict[str, AgentNetwork],
    baseline_name: str,
    latent_dim: int,
    n_episodes: int = 30,
    alpha: float = 0.6,
) -> tuple[dict[str, LatentSteeringAdapter], dict[str, SteeringFitStats]]:
    """
    Fit one steering adapter per non-baseline agent using aligned rollouts.
    """
    paired_latents = collect_steering_pairs(networks, baseline_name, n_episodes=n_episodes)
    adapters = {}
    fit_stats = {}

    for agent_name, (source_latents, target_latents) in paired_latents.items():
        adapter = LatentSteeringAdapter(latent_dim=latent_dim, alpha=alpha)
        fit_stats[agent_name] = adapter.fit(source_latents, target_latents)
        adapters[agent_name] = adapter

    return adapters, fit_stats


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


def print_aqi_summary(results: list[AQIResult]):
    print("\nAQI Baseline Selection")
    print("-" * 56)
    for result in sorted(results, key=lambda r: r.aqi, reverse=True):
        profile = result.model_name
        marker = "  <-- baseline" if result.is_baseline else ""
        print(f"  {profile:>8s}: AQI={result.aqi:.4f}{marker}")
    print("-" * 56)


def print_steering_summary(fit_stats: dict[str, SteeringFitStats]):
    print("\nLatent Steering Fit")
    print("-" * 56)
    for agent_name, stats in fit_stats.items():
        print(
            f"  {agent_name}: loss {stats.initial_loss:.4f} -> {stats.final_loss:.4f} | "
            f"target dist {stats.target_distance_before:.4f} -> {stats.target_distance_after:.4f}"
        )
    print("-" * 56)


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    LATENT_DIM = 16
    N_ACTIONS = 5  # simple_spread has 5 discrete actions: [no-op, left, right, down, up]
    ALIGNMENT_PROFILES = {
        "agent_0": "well_aligned",
        "agent_1": "partially_aligned",
        "agent_2": "poorly_aligned",
    }

    # Probe observation space size
    probe_env = simple_spread_v3.env(N=3, max_cycles=25, render_mode=None)
    probe_env.reset()
    obs_dim = probe_env.observation_space("agent_0").shape[0]
    probe_env.close()
    print(f"Environment: simple_spread_v3 (3 agents, obs_dim={obs_dim}, actions={N_ACTIONS})")

    # ── Step 1: Train cooperative agents ──────────────────────
    print("\n[1/5] Training agents (300 episodes of Q-learning)...")
    networks = train_agents(
        obs_dim,
        LATENT_DIM,
        N_ACTIONS,
        alignment_profiles=ALIGNMENT_PROFILES,
        n_episodes=300,
    )

    # ── Step 2: Select AQI baseline and fit steering adapters ─
    print("\n[2/5] Selecting baseline agent with AQI...")
    leader_name, aqi_results = select_baseline_agent(networks)
    print_aqi_summary(aqi_results)
    print(f"  Selected PettingZoo baseline agent: {leader_name}")

    print("\n[3/5] Fitting latent steering adapters...")
    steering_adapters, steering_stats = fit_latent_steering(
        networks, leader_name, LATENT_DIM, n_episodes=30, alpha=0.7
    )
    print_steering_summary(steering_stats)

    # ── Step 3: Calibrate evaluator on aligned episodes ───────
    print("\n[4/5] Collecting aligned calibration data (30 episodes)...")
    aligned_leader_lats, aligned_follower_lats = collect_aligned_latents(
        networks, leader_name, n_episodes=30
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

    # ── Step 4: Evaluate scenarios ────────────────────────────
    print("\n[5/5] Running evaluation episodes...")

    N_EVAL = 10
    results_aligned = []
    results_drifted = []
    results_steered = []
    results_random = []
    candidate_followers = [name for name in networks if name != leader_name]
    drifted_follower = candidate_followers[0]

    # Build a "random" network for the fully-random scenario
    random_networks = {}
    for name, net in networks.items():
        random_networks[name] = AgentNetwork(
            obs_dim,
            LATENT_DIM,
            N_ACTIONS,
            alignment_profile=net.alignment_profile,
        )

    for _ in range(N_EVAL):
        # (A) Aligned: all agents use the trained policy
        buf_a, iq_a, oq_a = run_episode(networks, leader_name)
        if buf_a.leader_steps and iq_a:
            results_aligned.append(evaluator.evaluate(buf_a, iq_a, oq_a))

        # (B) Drifted: follower latent moves out of the baseline region
        buf_b, iq_b, oq_b = run_episode(
            networks,
            leader_name,
            drifted_follower=drifted_follower,
            drift_scale=1.25,
        )
        if buf_b.leader_steps and iq_b:
            results_drifted.append(evaluator.evaluate(buf_b, iq_b, oq_b))

        # (C) Steered: same drift, but now corrected by the learned adapter
        buf_c, iq_c, oq_c = run_episode(
            networks,
            leader_name,
            latent_steering=steering_adapters,
            drifted_follower=drifted_follower,
            drift_scale=1.25,
        )
        if buf_c.leader_steps and iq_c:
            results_steered.append(evaluator.evaluate(buf_c, iq_c, oq_c))

        # (D) Fully random: untrained networks
        buf_d, iq_d, oq_d = run_episode(random_networks, leader_name)
        if buf_d.leader_steps and iq_d:
            results_random.append(evaluator.evaluate(buf_d, iq_d, oq_d))

    # ── Step 5: Aggregate and report ──────────────────────────
    print("\nResults")

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
        f"SCENARIO B: {drifted_follower} Drifted Before Steering (avg over {len(results_drifted)} episodes)",
        avg_result(results_drifted),
    )
    print_result(
        f"SCENARIO C: {drifted_follower} Drifted + Steered (avg over {len(results_steered)} episodes)",
        avg_result(results_steered),
    )
    print_result(
        f"SCENARIO D: Fully Random / Untrained (avg over {len(results_random)} episodes)",
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
