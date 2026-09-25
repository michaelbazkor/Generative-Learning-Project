# Curvature-Aware Step-Size Adaptation and Guidance Damping for OT Flow Matching

Training-free sampler that uses local trajectory acceleration to co-adapt ODE step sizes and Classifier-Free Guidance scales in Optimal Transport Conditional Flow Matching (OT-CFM).

## Documents

- [`docs/proposal.pdf`](docs/proposal.pdf) — project proposal
- [`docs/experiment_plan.pdf`](docs/experiment_plan.pdf) — initial experiment plan (not treated as fixed)
- [`docs/theory.md`](docs/theory.md) — proofs for material acceleration, equal-error steps, affine CFG
- [`REPORT.md`](REPORT.md) — chronological decisions and results

## Setup

```bash
pip install -r requirements.txt
# Optional CUDA (RTX 3050 / similar):
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## Reproduce

```bash
python tests/test_identities.py
python scripts/run_experiments.py --all
# Or stage-by-stage:
python scripts/run_experiments.py --stage coupling
python scripts/run_experiments.py --stage estimator
python scripts/run_experiments.py --stage step
python scripts/run_experiments.py --stage guidance
python scripts/run_experiments.py --stage factorial
python scripts/run_experiments.py --stage fmnist
python scripts/run_experiments.py --stage cifar
python scripts/run_experiments.py --stage report
```

## Layout

- `src/` — data, models, CFM loss, solvers, metrics, train/eval
- `scripts/run_experiments.py` — staged decision pipeline
- `results/json/` — raw metrics and decisions
- `results/figures/` — trajectory, W2, acceleration, and sample plots

## Method sketch

At each Heun step, evaluate conditional and unconditional velocities, form
$a(w) = a_{\emptyset} + w\cdot(a_c - a_{\emptyset})$ from the predictor–corrector pair (no extra NFE),
choose $w_{\mathrm{eff}}$ (fixed / $\gamma$-damped / acceleration-capped), then set
$\Delta t = \min(\eta / (\Vert a \Vert_{\mathrm{rms}}^{1/2} + \varepsilon),\, 1-t)$.
