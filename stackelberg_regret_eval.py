"""
Stackelberg Regret Evaluation Metric
for Multi-Agent Alignment in Latent Space

Core idea:
  SR = V_leader(pi_leader, BR_intended) - V_leader(pi_leader, pi_followers_observed)

  SR == 0  -> perfect alignment (followers converged to intended equilibrium)
  SR  > 0  -> followers are misaligned (leader loses value relative to expectation)
  SR  < 0  -> followers over-perform (rare, possible in imperfect leader models)

Decomposed into:
  - Behavioral SR   : action-space gap
  - Latent SR       : representation-space gap (novel, latent-space specific)
  - Nash Deviation  : stability of follower joint policy
  - Latent Cone     : geometric alignment check in z-space
"""

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from collections import deque


# ─────────────────────────────────────────────────────────────
# 0. DATA STRUCTURES
# ─────────────────────────────────────────────────────────────

@dataclass
class AgentStep:
    """One timestep snapshot from a single agent."""
    agent_id: int
    state: torch.Tensor          # raw observation
    latent: torch.Tensor         # z extracted from agent's encoder
    action: torch.Tensor         # sampled action (discrete idx or continuous vec)
    q_values: torch.Tensor       # Q(s, a) for all actions  [A]
    value: torch.Tensor          # V(s)  scalar


@dataclass
class EpisodeBuffer:
    """Stores a full episode of multi-agent steps for post-hoc evaluation."""
    leader_steps: List[AgentStep] = field(default_factory=list)
    follower_steps: Dict[int, List[AgentStep]] = field(default_factory=dict)
    leader_latent_objective: List[torch.Tensor] = field(default_factory=list)

    def add_leader(self, step: AgentStep, z_L: torch.Tensor):
        self.leader_steps.append(step)
        self.leader_latent_objective.append(z_L)

    def add_follower(self, step: AgentStep):
        fid = step.agent_id
        if fid not in self.follower_steps:
            self.follower_steps[fid] = []
        self.follower_steps[fid].append(step)


@dataclass
class StackelbergRegretResult:
    """Full evaluation report for one episode."""
    behavioral_sr: float      # action-level gap
    latent_sr: float          # representation-level gap
    nash_deviation: float     # NDI: follower NE stability
    latent_cone_rate: float   # fraction of steps inside leader's latent cone
    composite_sr: float       # weighted aggregate
    per_follower_brg: Dict[int, float] = field(default_factory=dict)  # BRG per agent
    is_aligned: bool = False  # True if composite_sr < threshold


# ─────────────────────────────────────────────────────────────
# 1. LATENT CONE BUILDER
#    Fits a reference manifold from aligned behavior episodes.
#    At eval time, measures whether follower latents stay inside.
# ─────────────────────────────────────────────────────────────

class LatentConeBuilder:
    """
    Collects latent vectors from 'aligned' (oracle/baseline) episodes
    and fits a bounding ellipsoid (Mahalanobis ball) in latent space.

    Usage:
        builder = LatentConeBuilder(latent_dim=64)
        builder.fit(aligned_latent_matrix)   # shape [N, D]
        in_cone = builder.check(eval_latents) # shape [M, D] -> bool [M]
    """

    def __init__(self, latent_dim: int, sigma_scale: float = 2.5):
        self.latent_dim = latent_dim
        self.sigma_scale = sigma_scale   # how many std-devs define the cone boundary
        self.mean: Optional[torch.Tensor] = None
        self.cov_inv: Optional[torch.Tensor] = None
        self.is_fit = False

    def fit(self, aligned_latents: torch.Tensor):
        """
        aligned_latents: [N, D] from oracle/aligned rollouts.
        Computes mean and inverse covariance for Mahalanobis distance.
        """
        N, D = aligned_latents.shape
        self.mean = aligned_latents.mean(dim=0)          # [D]
        centered = aligned_latents - self.mean           # [N, D]
        cov = (centered.T @ centered) / (N - 1)         # [D, D]

        # Add small ridge for numerical stability
        cov = cov + 1e-5 * torch.eye(D, device=cov.device)
        self.cov_inv = torch.linalg.inv(cov)             # [D, D]
        self.is_fit = True

    def mahalanobis_distance(self, latents: torch.Tensor) -> torch.Tensor:
        """
        latents: [M, D]
        returns: [M] Mahalanobis distances from the cone center.
        """
        assert self.is_fit, "Call fit() with aligned latents first."
        diff = latents - self.mean                       # [M, D]
        left = diff @ self.cov_inv                       # [M, D]
        dist = (left * diff).sum(dim=-1).sqrt()          # [M]
        return dist

    def check(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Returns bool [M]: True if inside the aligned latent cone.
        """
        dist = self.mahalanobis_distance(latents)
        return dist < self.sigma_scale

    def cone_membership_rate(self, latents: torch.Tensor) -> float:
        """Scalar: fraction of latent vectors inside the cone."""
        return self.check(latents).float().mean().item()


# ─────────────────────────────────────────────────────────────
# 2. BEHAVIORAL STACKELBERG REGRET
#    V_leader(intended) - V_leader(observed)
#    Requires: leader's value function, intended BR policy,
#              observed follower joint policy.
# ─────────────────────────────────────────────────────────────

class BehavioralStackelbergRegret:
    """
    Computes action-space SR by rolling out:
      (a) what the leader expected (intended BR)
      (b) what the followers actually did (observed)
    and comparing the resulting leader values.

    In practice you need either:
      - A learned value function for the leader, OR
      - Episode returns (cumulative rewards)

    Here we use Q-values as a proxy for the leader's expected value
    under the intended vs observed follower policies.
    """

    def __init__(self, gamma: float = 0.99):
        self.gamma = gamma

    def compute(
        self,
        leader_steps: List[AgentStep],
        intended_br_q_values: List[torch.Tensor],  # Q under intended BR, per step
        observed_follower_joint_q: List[torch.Tensor],  # Q under observed policy
    ) -> float:
        """
        Returns scalar SR.
        intended_br_q_values[t]     : [A] Q-values if followers had played intended BR
        observed_follower_joint_q[t]: [A] Q-values under actual follower behavior
        """
        assert len(intended_br_q_values) == len(observed_follower_joint_q)

        T = len(leader_steps)
        sr_per_step = []

        for t in range(T):
            # Leader's value under intended equilibrium (best Q over actions)
            v_intended = intended_br_q_values[t].max().item()

            # Leader's value under actual follower behavior
            leader_action = leader_steps[t].action
            if leader_action.dim() == 0:
                a_idx = leader_action.long().item()
            else:
                a_idx = leader_action.argmax().item()

            v_observed = observed_follower_joint_q[t][a_idx].item()

            sr_per_step.append(v_intended - v_observed)

        # Discount-weighted average over the episode
        weights = torch.tensor(
            [self.gamma ** t for t in range(T)], dtype=torch.float32
        )
        weights /= weights.sum()
        return float((torch.tensor(sr_per_step) * weights).sum())


# ─────────────────────────────────────────────────────────────
# 3. LATENT STACKELBERG REGRET
#    Measures alignment gap in representation space.
#    A well-aligned follower's latent should be geometrically
#    consistent with the leader's latent objective z_L.
# ─────────────────────────────────────────────────────────────

class LatentStackelbergRegret:
    """
    Latent SR = E_t [ || z_follower(t) - z_L_projected(t) ||_2 ]

    z_L_projected: the point on the follower's latent manifold that
    the leader's z_L 'predicts'. Estimated via a learned projection head
    (trained offline to map z_L -> expected follower z).

    If no projection head is available, falls back to direct cosine distance
    between leader and follower latents (simpler but less calibrated).
    """

    def __init__(
        self,
        latent_dim: int,
        use_projection_head: bool = True,
        device: str = "cpu"
    ):
        self.latent_dim = latent_dim
        self.use_projection_head = use_projection_head
        self.device = device

        if use_projection_head:
            # Shallow MLP: z_L -> expected follower z
            # Train this offline on aligned (leader, follower) latent pairs
            self.proj_head = nn.Sequential(
                nn.Linear(latent_dim, latent_dim * 2),
                nn.ReLU(),
                nn.Linear(latent_dim * 2, latent_dim),
            ).to(device)
        else:
            self.proj_head = None

    def fit_projection_head(
        self,
        leader_latents: torch.Tensor,    # [N, D] from aligned episodes
        follower_latents: torch.Tensor,  # [N, D] corresponding follower latents
        epochs: int = 100,
        lr: float = 1e-3,
    ):
        """
        Supervised training: learn z_L -> z_follower mapping
        from aligned (oracle) episodes.
        """
        assert self.use_projection_head, "projection head not enabled"
        optim = torch.optim.Adam(self.proj_head.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        for _ in range(epochs):
            pred = self.proj_head(leader_latents)
            loss = loss_fn(pred, follower_latents)
            optim.zero_grad()
            loss.backward()
            optim.step()

    def compute(
        self,
        leader_latents: torch.Tensor,    # [T, D]  z_L per timestep
        follower_latents: torch.Tensor,  # [T, D]  z_follower per timestep
    ) -> float:
        """
        Returns scalar latent SR (mean distance between
        projected leader objective and actual follower latent).
        """
        if self.use_projection_head and self.proj_head is not None:
            with torch.no_grad():
                expected_follower_z = self.proj_head(leader_latents)  # [T, D]
        else:
            # Fallback: use raw leader latent as proxy
            expected_follower_z = leader_latents

        # L2 distance in latent space, averaged over episode
        dist = (follower_latents - expected_follower_z).norm(dim=-1)  # [T]
        return dist.mean().item()

    def alignment_angle(
        self,
        leader_latents: torch.Tensor,    # [T, D]
        follower_latents: torch.Tensor,  # [T, D]
    ) -> float:
        """
        Cosine similarity between leader and follower latent directions.
        1 = aligned, -1 = adversarial, 0 = orthogonal/drifted.
        """
        cos = nn.functional.cosine_similarity(leader_latents, follower_latents, dim=-1)
        return cos.mean().item()


# ─────────────────────────────────────────────────────────────
# 4. NASH DEVIATION INDEX
#    Checks if follower joint policy is at a Nash Equilibrium.
#    NDI > 0 means at least one follower can profitably deviate,
#    indicating the system hasn't converged / has drifted.
# ─────────────────────────────────────────────────────────────

class NashDeviationIndex:
    """
    NDI = sum_i max(0, max_a Q_i(a | a_{-i}) - Q_i(a_i | a_{-i}))

    For each follower i:
      - a_i       : their current action
      - a_{-i}    : all other followers' current actions (fixed)
      - max_a Q_i : best they could do by deviating

    High NDI -> unstable equilibrium or post-convergence drift.
    NDI ~ 0  -> followers are at NE (but check if it's the RIGHT NE via SR)
    """

    def compute(self, follower_steps_at_t: List[AgentStep]) -> float:
        """
        follower_steps_at_t: list of AgentStep, one per follower, at ONE timestep.
        Returns NDI scalar for that timestep.
        """
        ndi = 0.0
        for step in follower_steps_at_t:
            # Q_i for the action they actually took
            action_idx = step.action.long().item() if step.action.dim() == 0 \
                         else step.action.argmax().item()
            q_current = step.q_values[action_idx].item()

            # Best possible Q_i if they deviate
            q_best = step.q_values.max().item()

            # Positive deviation potential = misalignment signal
            ndi += max(0.0, q_best - q_current)

        return ndi

    def compute_episode(
        self,
        buffer: EpisodeBuffer,
    ) -> float:
        """Average NDI over all timesteps in an episode."""
        follower_ids = list(buffer.follower_steps.keys())
        T = min(len(buffer.follower_steps[fid]) for fid in follower_ids)
        ndi_per_t = []

        for t in range(T):
            steps_at_t = [buffer.follower_steps[fid][t] for fid in follower_ids]
            ndi_per_t.append(self.compute(steps_at_t))

        return float(np.mean(ndi_per_t))


# ─────────────────────────────────────────────────────────────
# 5. BEST-RESPONSE GAP (per follower)
#    BRG_i = Q_i(BR_i | pi_leader) - Q_i(a_i_observed | pi_leader)
#    Isolates WHICH follower is misaligned.
# ─────────────────────────────────────────────────────────────

class BestResponseGap:
    """
    BRG_i = max_a Q_i(a | z_L) - Q_i(a_i_obs | z_L)

    Zero means follower i IS playing their true BR to the leader.
    Positive means they're leaving value on the table -> misaligned or unconverged.
    """

    def compute_per_follower(
        self,
        buffer: EpisodeBuffer,
    ) -> Dict[int, float]:
        """Returns mean BRG for each follower agent across the episode."""
        brg_per_follower = {}

        for fid, steps in buffer.follower_steps.items():
            brg_t = []
            for step in steps:
                action_idx = step.action.long().item() if step.action.dim() == 0 \
                             else step.action.argmax().item()
                q_obs = step.q_values[action_idx].item()
                q_best = step.q_values.max().item()
                brg_t.append(max(0.0, q_best - q_obs))
            brg_per_follower[fid] = float(np.mean(brg_t))

        return brg_per_follower


# ─────────────────────────────────────────────────────────────
# 6. COMPOSITE EVALUATOR
#    Orchestrates all sub-metrics into a single SR report.
# ─────────────────────────────────────────────────────────────

class StackelbergRegretEvaluator:
    """
    Main evaluation class. Computes the full SR report for an episode.

    Usage:
        evaluator = StackelbergRegretEvaluator(latent_dim=64)
        evaluator.fit_aligned_cone(aligned_leader_latents)
        evaluator.fit_projection_head(aligned_leader_latents, aligned_follower_latents)
        result = evaluator.evaluate(buffer, intended_br_q_values, observed_q_values)
    """

    def __init__(
        self,
        latent_dim: int,
        n_actions: int,
        gamma: float = 0.99,
        sigma_scale: float = 2.5,
        sr_threshold: float = 0.1,      # below this -> considered aligned
        weights: Optional[Dict[str, float]] = None,
        device: str = "cpu",
    ):
        self.latent_dim = latent_dim
        self.n_actions = n_actions
        self.sr_threshold = sr_threshold
        self.device = device

        # Sub-metric modules
        self.cone_builder = LatentConeBuilder(latent_dim, sigma_scale)
        self.behavioral_sr = BehavioralStackelbergRegret(gamma)
        self.latent_sr = LatentStackelbergRegret(latent_dim, device=device)
        self.nash_idx = NashDeviationIndex()
        self.br_gap = BestResponseGap()

        # Composite weighting
        self.weights = weights or {
            "behavioral": 0.35,
            "latent": 0.30,
            "nash": 0.20,
            "cone": 0.15,
        }

    # ── Offline fitting ───────────────────────────────────────

    def fit_aligned_cone(self, aligned_leader_latents: torch.Tensor):
        """Call once with aligned-episode leader latents to build the reference cone."""
        self.cone_builder.fit(aligned_leader_latents)

    def fit_projection_head(
        self,
        aligned_leader_latents: torch.Tensor,
        aligned_follower_latents: torch.Tensor,
        epochs: int = 100,
    ):
        """Trains z_L -> z_follower projection on aligned episodes."""
        self.latent_sr.fit_projection_head(
            aligned_leader_latents, aligned_follower_latents, epochs
        )

    # ── Episode evaluation ────────────────────────────────────

    def evaluate(
        self,
        buffer: EpisodeBuffer,
        intended_br_q_values: List[torch.Tensor],
        observed_follower_joint_q: List[torch.Tensor],
    ) -> StackelbergRegretResult:
        """
        Full evaluation pass over one episode.

        Args:
            buffer                   : EpisodeBuffer with leader + follower steps
            intended_br_q_values     : list[T] of [A] Q-tensors under intended BR
            observed_follower_joint_q: list[T] of [A] Q-tensors under observed policy
        """
        # 1. Behavioral SR
        b_sr = self.behavioral_sr.compute(
            buffer.leader_steps,
            intended_br_q_values,
            observed_follower_joint_q,
        )

        # 2. Latent SR
        leader_lat = torch.stack(buffer.leader_latent_objective)     # [T, D]
        follower_ids = list(buffer.follower_steps.keys())
        T = min(len(buffer.follower_steps[fid]) for fid in follower_ids)

        # Average follower latent across all followers at each timestep
        avg_follower_lat = torch.stack([
            torch.stack([
                buffer.follower_steps[fid][t].latent
                for fid in follower_ids
            ]).mean(dim=0)
            for t in range(T)
        ])                                                            # [T, D]

        l_sr = self.latent_sr.compute(leader_lat[:T], avg_follower_lat)

        # 3. Nash Deviation Index
        ndi = self.nash_idx.compute_episode(buffer)

        # 4. Latent Cone membership rate
        all_follower_lats = torch.cat([
            torch.stack([s.latent for s in steps])
            for steps in buffer.follower_steps.values()
        ])  # [T * n_followers, D]

        cone_rate = (
            self.cone_builder.cone_membership_rate(all_follower_lats)
            if self.cone_builder.is_fit else 0.5
        )

        # 5. Per-follower BRG
        brg = self.br_gap.compute_per_follower(buffer)

        # 6. Composite SR (normalize NDI and cone to [0,1] range then weight)
        ndi_norm = float(np.tanh(ndi))          # squash to (0,1)
        cone_penalty = 1.0 - cone_rate          # 0=all inside, 1=all outside

        composite = (
            self.weights["behavioral"] * max(0.0, b_sr) +
            self.weights["latent"]     * l_sr +
            self.weights["nash"]       * ndi_norm +
            self.weights["cone"]       * cone_penalty
        )

        return StackelbergRegretResult(
            behavioral_sr=b_sr,
            latent_sr=l_sr,
            nash_deviation=ndi,
            latent_cone_rate=cone_rate,
            composite_sr=composite,
            per_follower_brg=brg,
            is_aligned=composite < self.sr_threshold,
        )

    # ── Temporal tracking ─────────────────────────────────────

    def track_drift(
        self,
        results: List[StackelbergRegretResult],
        window: int = 10,
    ) -> Dict[str, List[float]]:
        """
        Given a sequence of episode results, returns rolling-window
        averages for each metric to detect alignment drift over training.
        """
        metrics = {
            "composite_sr": [r.composite_sr for r in results],
            "behavioral_sr": [r.behavioral_sr for r in results],
            "latent_sr": [r.latent_sr for r in results],
            "nash_deviation": [r.nash_deviation for r in results],
            "cone_rate": [r.latent_cone_rate for r in results],
        }

        smoothed = {}
        for key, vals in metrics.items():
            buf = deque(maxlen=window)
            rolled = []
            for v in vals:
                buf.append(v)
                rolled.append(float(np.mean(buf)))
            smoothed[key] = rolled

        return smoothed


# ─────────────────────────────────────────────────────────────
# 7. MINIMAL USAGE EXAMPLE
# ─────────────────────────────────────────────────────────────

def make_mock_agent_step(
    agent_id: int,
    latent_dim: int,
    n_actions: int,
    aligned: bool = True,
) -> AgentStep:
    """Helper: creates a synthetic AgentStep for testing."""
    latent = torch.randn(latent_dim)
    if not aligned:
        latent = latent + 3.0  # shift far from aligned cone

    q_values = torch.randn(n_actions)
    action = torch.tensor(q_values.argmax().item())  # greedy if aligned

    if not aligned:
        # Misaligned: pick a suboptimal action
        worst = q_values.argmin().item()
        action = torch.tensor(worst)

    return AgentStep(
        agent_id=agent_id,
        state=torch.randn(8),
        latent=latent,
        action=action,
        q_values=q_values,
        value=q_values.max(),
    )


def demo():
    LATENT_DIM = 16
    N_ACTIONS = 4
    N_FOLLOWERS = 3
    EPISODE_LEN = 20

    evaluator = StackelbergRegretEvaluator(
        latent_dim=LATENT_DIM,
        n_actions=N_ACTIONS,
        gamma=0.99,
        sigma_scale=2.5,
        sr_threshold=0.15,
    )

    # ── Step 1: fit on aligned (oracle) episodes ──────────────
    N_ALIGNED = 200
    aligned_leader_lats = torch.randn(N_ALIGNED, LATENT_DIM)
    aligned_follower_lats = aligned_leader_lats + 0.1 * torch.randn(N_ALIGNED, LATENT_DIM)

    evaluator.fit_aligned_cone(aligned_leader_lats)
    evaluator.fit_projection_head(aligned_leader_lats, aligned_follower_lats, epochs=50)

    # ── Step 2: run an eval episode (misaligned) ──────────────
    buffer = EpisodeBuffer()

    for t in range(EPISODE_LEN):
        leader_step = make_mock_agent_step(0, LATENT_DIM, N_ACTIONS, aligned=True)
        z_L = torch.randn(LATENT_DIM)
        buffer.add_leader(leader_step, z_L)

        for fid in range(1, N_FOLLOWERS + 1):
            misaligned = (fid == 2)  # follower 2 is the bad actor
            f_step = make_mock_agent_step(fid, LATENT_DIM, N_ACTIONS, aligned=not misaligned)
            buffer.add_follower(f_step)

    intended_q = [torch.randn(N_ACTIONS) for _ in range(EPISODE_LEN)]
    observed_q  = [q - 0.5 * torch.rand(N_ACTIONS) for q in intended_q]

    # ── Step 3: evaluate ─────────────────────────────────────
    result = evaluator.evaluate(buffer, intended_q, observed_q)

    print("=" * 50)
    print("  Stackelberg Regret Evaluation Report")
    print("=" * 50)
    print(f"  Behavioral SR    : {result.behavioral_sr:.4f}")
    print(f"  Latent SR        : {result.latent_sr:.4f}")
    print(f"  Nash Deviation   : {result.nash_deviation:.4f}")
    print(f"  Latent Cone Rate : {result.latent_cone_rate:.4f}")
    print(f"  Composite SR     : {result.composite_sr:.4f}")
    print(f"  Is Aligned?      : {result.is_aligned}")
    print()
    print("  Per-Follower Best-Response Gap:")
    for fid, brg in result.per_follower_brg.items():
        flag = " <-- misaligned" if brg > 0.3 else ""
        print(f"    Follower {fid}: BRG = {brg:.4f}{flag}")
    print("=" * 50)


if __name__ == "__main__":
    demo()
