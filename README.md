# Inverse Study Tutorial

A hands-on, notebook-based course on **inverse problems**: how to recover hidden
parameters or fields from indirect, noisy measurements, and why the obvious
recipe usually fails.

Every tutorial is self-contained, runs in seconds on a laptop (NumPy + Matplotlib only),
and checks each formula numerically before trusting it.

## Roadmap

| # | Module | Status |
|---|---|---|
| 1 | **Why inverse problems are hard**: forward vs. inverse, noise amplification, a first stable fix by least squares | Available |
| 2 | Least squares in depth: normal equations, conditioning, SVD, Tikhonov / L-curve regularisation | Available |
| 3 | Adjoint methods: gradients of PDE-constrained objectives | Planned |
| 4 | 4D-Var: variational data assimilation | Planned |
| 5 | Kalman filters: KF, EKF, EnKF | Planned |
| 6 | PINNs for inverse problems | Planned |
| 7 | Full-waveform inversion (FWI) | Planned |

## Module 1: Why inverse problems are hard (available)

**Running example.** A rod heated uniformly, $-\alpha T''(x) = q$ with fixed end
temperatures. We measure $T(x)$ and want the conductivity $\alpha$.
The naive inverse $\alpha = -q / T''$ is exact on clean data and meaningless on noisy data.

| Notebook | Question | Answer in one line |
|---|---|---|
| [01_forward_and_clean_FDM](01_forward_and_clean_FDM.ipynb) | Does the finite-difference inverse work at all? | Yes on clean data, to about 1e-14. |
| [02_noise_breaks_naive_inverse](02_noise_breaks_naive_inverse.ipynb) | What breaks with noise? | Noise in $T''$ scales as $\sqrt{6}\,\sigma/\Delta x^2$, so finer grids make it worse. |
| [03_frequency_view_and_global_fix](03_frequency_view_and_global_fix.ipynb) | Why, and what is the first fix? | Differentiation is a high-pass amplifier; fitting all data at once by least squares is stable. |

**What you will learn**

- The difference between a stable forward map and an unstable inverse map.
- Why 1% measurement noise can be about 30x the signal inside a second-difference stencil.
- Why refining the mesh *hurts* the pointwise inverse, and why even round-off follows the same law.
- The frequency-domain explanation (gain up to $4/\Delta x^2$).
- A global least-squares fit that improves as data are added, and the honest caveat:
  it works because the model form was assumed. That is the motivation for the later modules.

Each notebook ends with exercises.

## Getting started

```bash
git clone https://github.com/AGN000/InverseStudyTutorial.git
cd InverseStudyTutorial
pip install -r requirements.txt
jupyter lab
```

Open the notebooks in order. They are already executed, so GitHub shows the plots without running anything.

## License

MIT. See [LICENSE](LICENSE).
