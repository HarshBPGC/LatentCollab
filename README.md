# LatentCollab

LatentCollab is a research prototype for evaluating and repairing alignment drift in latent multi-agent collaboration. It combines three components:

- **Alignment Quality Index (AQI):** scores candidate agents across harmlessness, helpfulness, honesty, consistency, and instruction fidelity.
- **Latent steering:** learns a lightweight adapter that moves follower hidden states toward a selected baseline agent's latent region.
- **Stackelberg Regret:** evaluates whether follower behavior and representations remain aligned with the leader's intended collaborative equilibrium.

The current implementation uses PettingZoo's cooperative navigation environment as a controlled testbed for the full AQI -> steering -> regret evaluation loop.

## Repository Structure

```text
.
|-- alignment_probes.json       # Probe set used by AQI scoring
|-- aqi_module.py               # AQI data structures, scorers, and pipeline
|-- latent_steering.py          # Linear latent steering adapter
|-- stackelberg_regret_eval.py  # Behavioral and latent Stackelberg Regret metrics
|-- pettingzoo_sr_eval.py       # End-to-end PettingZoo evaluation harness
|-- generate_plots.py           # Generates project figures
|-- figures/                    # Generated visualizations and poster assets
`-- requirements.txt            # Python dependencies
```

## Setup

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

Run the full PettingZoo evaluation pipeline:

```bash
python pettingzoo_sr_eval.py
```

Generate all figures:

```bash
python generate_plots.py
```

Generated plots are saved to `figures/`.

## Method Overview

1. **Score candidate agents with AQI.** Each model/profile is evaluated on alignment probes, and the highest-scoring agent becomes the leader.
2. **Calibrate the aligned latent region.** Aligned rollouts define the leader's latent reference manifold.
3. **Steer follower latents.** A lightweight identity-regularized adapter maps follower hidden states toward the leader region.
4. **Evaluate collaboration.** Composite Stackelberg Regret combines behavioral regret, latent regret, Nash deviation, and latent cone membership.

## Notes

This project currently uses small PettingZoo agents as a simulation scaffold. A natural next step is to replace the symbolic profiles with open-source LLM agents, extract their hidden states, and evaluate latent steering on real collaborative reasoning or tool-use tasks.
