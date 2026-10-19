# Inverse Study Tutorial

A hands-on, notebook-based course on **inverse problems**: how to recover hidden
parameters or fields from indirect, noisy measurements, and why the obvious
recipe usually fails.

Every tutorial is self-contained and checks each formula numerically before
trusting it. Most need only NumPy, SciPy and Matplotlib and run in seconds on a
laptop; Tutorial 7 additionally needs JAX (and `ffmpeg` if you want to save its
animation).

## Roadmap

| # | Notebook | Topic | Status |
|---|---|---|---|
| 1-3 | `01`, `02`, `03` | **Why inverse problems are hard**: forward vs. inverse, noise amplification, a first stable fix by least squares | Available |
| 4 | [04_Least_Squares_Regularisation](04_Least_Squares_Regularisation.ipynb) | Normal equations, conditioning, SVD, Picard plot, TSVD / Tikhonov, L-curve | Available |
| 5 | [Tutorial_05_Adjoint_Methods_PDE_Inverse_Problems_revised](Tutorial_05_Adjoint_Methods_PDE_Inverse_Problems_revised.ipynb) | Adjoint methods: gradients of PDE-constrained objectives | Available |
| 6 | [06_pressure_from_PIV](06_pressure_from_PIV.ipynb) | Pressure from PIV velocity: Poisson solve, noise blow-up (synthetic + OpenPIV data) | Available |
| 7 | [07_shape_from_wall_pressure_Euler_JAX](07_shape_from_wall_pressure_Euler_JAX.ipynb) | Wall shape from surface pressure: 2D Euler solver in JAX, automatic-differentiation gradients, L-BFGS-B | Available |
| 8 | [08_numerical_crime_diffusion_adjoint](08_numerical_crime_diffusion_adjoint.ipynb) | Numerical diffusion of upwind vs. CFL, the inverse crime, hand-coded discrete adjoint | Available |
| 9 | [09_4DVar_cost_function_and_linear_case](09_4DVar_cost_function_and_linear_case.ipynb) | 4D-Var 1/5: cost function, Kalman gain, 4D-Var = Kalman filter in the linear case | Available |
| 10 | [10_4DVar_tangent_linear_and_adjoint](10_4DVar_tangent_linear_and_adjoint.ipynb) | 4D-Var 2/5: Lorenz-63, tangent-linear and adjoint models, dot-product and Taylor tests | Available |
| 11 | [11_4DVar_twin_experiment_and_window_length](11_4DVar_twin_experiment_and_window_length.ipynb) | 4D-Var 3/5: twin experiment, window length vs non-convexity, cycling | Available |
| 12 | [12_4DVar_background_covariance_and_observations](12_4DVar_background_covariance_and_observations.ipynb) | 4D-Var 4/5: Lorenz-96, 3D-Var vs 4D-Var, choosing B, observation density | Available |
| 13 | [13_4DVar_model_error](13_4DVar_model_error.ipynb) | 4D-Var 5/5: imperfect models, the inverse crime in data assimilation, chi-square diagnostic | Available |
| 14 | Kalman filters | KF, EKF, EnKF | Planned |
| 15 | PINNs | PINNs for inverse problems | Planned |
| 16 | FWI | Full-waveform inversion | Planned |

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

## Tutorials 4-13 at a glance

| Notebook | Question | Answer in one line |
|---|---|---|
| 04 | How do we stabilise an ill-conditioned linear inverse problem? | Truncate or damp the small singular values (TSVD, Tikhonov) and pick the strength with the L-curve. |
| 05 | How do we get gradients when there are many unknowns? | One adjoint solve gives the whole gradient, verified against finite differences. |
| 06 | Can we get pressure from measured velocity? | Yes via a Poisson equation, but differentiating noisy velocity amplifies noise. |
| 07 | Can we recover a wall shape from its surface pressure? | Yes: autodiff through a 2D Euler solver recovers the shape to about 0.1% RMS, with caveats listed in the notebook. |
| 08 | What does a first-order solver do to an inverse problem? | Upwind diffusion grows as CFL falls and biases the fitted parameters; data from the same solver (inverse crime) hides it. |
| 09-13 | How do we combine a model and noisy data over a time window? | Minimise a prior-plus-misfit cost with adjoint gradients; keep windows short and cycle; get B right; beware model error. |

**Tutorials 9 to 13 (4D-Var) are five short notebooks; read them in order.** Each is self-contained and runs on its own. Tutorial 9 is linear algebra only; 10 to 13 use the Lorenz-63 and Lorenz-96 models and a hand-coded discrete adjoint (no JAX needed). Tutorial 12 takes a few minutes to run because it performs cycled assimilation experiments.

## Getting started

```bash
git clone https://github.com/AGN000/InverseStudyTutorial.git
cd InverseStudyTutorial
pip install -r requirements.txt
jupyter lab
```

Open the notebooks in order. Tutorial 7 needs `pip install jax`. They are already executed, so GitHub shows the plots without running anything.

## License

MIT. See [LICENSE](LICENSE).
