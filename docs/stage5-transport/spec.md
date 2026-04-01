# Stage 5: Transport & Power Balance

## Overview

Stage 5 evolves radial density and temperature profiles using neoclassical and
turbulent fluxes, and computes whole-device fusion-power metrics.  This is the
final stage of the forward pass, producing transport-consistent profiles and the
headline numbers ($P_\text{fus}$, $Q$).

**JAX-first priority:** NEOPAX is the primary code (JAX-native, uses diffrax ODE
solver).  Trinity3D is the traditional alternative with mature GX+SFINCS
coupling.

**Position in pipeline:** Receives $D_{ij}$ database from Stage 3 (MONKES),
turbulent fluxes from Stage 4 (SPECTRAX-GK), and geometry from Stage 1/2.
Produces the forward-pass output: updated $n(r)$, $T(r)$, $E_r(r)$,
$P_\text{fus}$, $Q$.

**Important note on NEOPAX turbulence coupling:** NEOPAX has
turbulence-coupling utilities, but the public examples center on the
neoclassical reduced model consuming MONKES $D_{ij}$.  Coupling SPECTRAX-GK
turbulent flux into NEOPAX is a coordination point with the Stage 4 owner.  The
alternative path (GX -> Trinity3D) has mature, tested turbulence coupling.

**Outer-loop handoff (future, not forward pass):** Updated pressure and current
profiles feed back to Stage 1 for the next iteration.  This loop closure is NOT
part of the initial forward-pass goal.

Reference: `stellarator_workflow.tex`, Sections 4.8--4.9;
`stellarator_io_reference.tex`, Sections 3.11--3.12.

---

## Codes

### NEOPAX (Primary JAX)

- **Repository:** <https://github.com/uwplasma/NEOPAX>
- **Language:** Python / JAX
- **Role:** Reduced neoclassical transport and profile evolution using MONKES
  $D_{ij}$ databases.  Uses diffrax for JAX-native ODE integration.

### Trinity3D (Alternative)

- **Repository:** <https://bitbucket.org/gyrokinetics/t3d>
- **Documentation:** (see Trinity3D docs)
- **Language:** Python
- **Role:** Global transport solver coupling GX turbulence and SFINCS
  neoclassical fluxes.  Implicit linearized time advance.

### Installation & Platform

<!-- OWNER COMPLETES: Document installation instructions for NEOPAX and
     Trinity3D, including:
     - Python version requirements and virtual-environment setup
     - JAX installation (CPU vs GPU) and diffrax version pins
     - Any Fortran/C dependencies for Trinity3D (GX, SFINCS)
     - Platform-specific notes (Linux cluster, macOS dev, Docker base image)
     - How to verify a working installation (e.g., run a smoke test) -->

---

## Input Specification

### NEOPAX Inputs

| Field | Type | Description | Source |
|-------|------|-------------|--------|
| `wout_*.nc` | NetCDF | VMEC equilibrium geometry | Stage 1 |
| `boozmn_*.nc` | NetCDF | Boozer-coordinate equilibrium | Stage 2 |
| $D_{ij}$ database | HDF5 | Monoenergetic transport coefficients | Stage 3 (MONKES) |

**$D_{ij}$ database fields consumed by NEOPAX reader:**

| Field | Type | Description |
|-------|------|-------------|
| `D11` | array | Monoenergetic coefficient (1,1) |
| `D13` | array | Monoenergetic coefficient (1,3) |
| `D33` | array | Monoenergetic coefficient (3,3) |
| `Er` | array | Radial electric field grid |
| `Er_tilde` | array | Normalized $E_r$ |
| `drds` | array | Radial coordinate Jacobian |
| `rho` | array | Radial coordinate |
| `nu_v` | array | Collisionality grid |

**Optional profile initialization:** NTSS-like HDF5 files with arrays: `r`,
`Er`, `Te`, `ne`, `Pressure`, `I_bs`, and related transport quantities.

### Trinity3D Inputs (Alternative)

| Field | Type | Description | Source |
|-------|------|-------------|--------|
| TOML config (`*.in`) | file | Groups: `[grid]`, `[time]`, `[[model]]`, `[[species]]`, `[geometry]`, `[physics]`, `[log]` | User-specified |
| `wout_*.nc` | via `[geometry]` | VMEC geometry | Stage 1 |
| `gx_template` | via `[[model]]` | GX input template for turbulence model | Stage 4 (GX) |
| `gx_outputs` | via `[[model]]` | GX flux outputs location | Stage 4 (GX) |
| SFINCS fluxes | via `[[model]]` | Neoclassical fluxes | Stage 3 (SFINCS) |

### Input Validation

<!-- OWNER COMPLETES: Define input validation rules, including:
     - Required fields, expected shapes/dimensions, and dtype constraints
       for the D_ij HDF5 database (e.g., D11, D13, D33 must share the same
       shape; rho must be monotonically increasing on [0, 1])
     - Sanity checks on physical ranges (e.g., Er grid bounds, nu_v > 0)
     - Validation that wout and boozmn files are consistent (same equilibrium)
     - Trinity3D TOML schema validation (required keys, type checks)
     - Error messages and exit behavior when validation fails -->

---

## Output Specification

### NEOPAX Outputs

**Core API returns (in-memory):**

| Field | Type | Description | Used As |
|-------|------|-------------|---------|
| `Lij` | 2D array | Thermal transport matrix | Transport analysis |
| `Gamma` | array (per species) | Particle flux | Objective / transport |
| `Q` | array (per species) | Heat flux | Objective / transport |
| `Upar` | array (per species) | Parallel flow | Diagnostic |

**HDF5 outputs (minimal):**

| Field | Type | Description |
|-------|------|-------------|
| `rho` | 1D array | Radial coordinate |
| `Er` | 1D array | Ambipolar radial electric field |
| `Jboots` | 1D array | Bootstrap current density |

**HDF5 outputs (full NTSS-style):**

| Field | Type | Description | Used As |
|-------|------|-------------|---------|
| `Pressure` | 1D array | Total pressure profile | **Outer-loop handoff** |
| `FluxNeo` | 1D array | Neoclassical particle flux | Diagnostic |
| `FluxQe` | 1D array | Electron heat flux | Diagnostic |
| `FluxQi` | 1D array | Ion heat flux | Diagnostic |
| `AmbiFlux` | 1D array | Ambipolar flux | Diagnostic |
| `J_bs` | 1D array | Bootstrap current density | **Outer-loop handoff** |
| `I_bs` | scalar | Total bootstrap current | Objective |
| `I_tor` | scalar | Total toroidal current | Objective |
| `beta` | 1D array | Local beta | Objective |
| `Te` | 1D array | Electron temperature profile | **Forward-pass output** |
| `TD` | 1D array | Deuterium temperature | Forward-pass output |
| `Tt` | 1D array | Tritium temperature | Forward-pass output |
| `ne` | 1D array | Electron density profile | **Forward-pass output** |
| `nD` | 1D array | Deuterium density | Forward-pass output |
| `nHe` | 1D array | Helium density (ash) | Forward-pass output |

### Trinity3D Outputs (Alternative)

**NetCDF / ADIOS2 / numpy-log outputs:**

Geometry group:

| Field | Description |
|-------|-------------|
| `B0` | Reference magnetic field |
| `Btor` | Toroidal field |
| `a_minor` | Minor radius |
| `R_major` | Major radius |
| `area` | Flux surface area |
| `grho` | Geometric factor |

Profile histories:

| Field | Description |
|-------|-------------|
| `n_e` | Electron density vs ($\rho$, time) |
| `T_e` | Electron temperature vs ($\rho$, time) |
| `n_H` | Hydrogen density vs ($\rho$, time) |
| `T_H` | Hydrogen temperature vs ($\rho$, time) |

Flux histories:

| Field | Description |
|-------|-------------|
| `pflux_*` | Particle flux (total + per-model) |
| `qflux_*` | Heat flux (total + per-model) |

Gradients and sources:

| Field | Description |
|-------|-------------|
| `aLn_*` | Density gradient scale lengths |
| `aLT_*` | Temperature gradient scale lengths |
| `Sn_*` | Particle sources |
| `Sp_*` | Power sources |

Device metrics:

| Field | Description | Used As |
|-------|-------------|---------|
| `beta_vol` | Volume-averaged beta | Objective |
| `Paux_MW` | Auxiliary power (MW) | Objective |
| `Palpha_int_MW` | Alpha heating power (MW) | Diagnostic |
| `Pfus_MW` | Fusion power (MW) | **Key forward-pass output** |
| `Qfus` | Fusion gain $Q = P_\text{fus} / P_\text{aux}$ | **Key forward-pass output** |

### Subset: Forward-Pass Output

The essential forward-pass outputs are: $n_s(r)$, $T_s(r)$, $E_r(r)$, and
whole-device metrics $P_\text{fus}$ and $Q$.

### Subset: Outer-Loop Handoff (Future)

Updated pressure $p(r) = \sum_s n_s(r)\, T_s(r)$ and bootstrap-current
profiles, to be fed back to Stage 1.  NOT part of the initial forward-pass
scope.

### Outputs Used as Objectives

Ambipolar $E_r$, bootstrap current, toroidal current, neoclassical fluxes,
$P_\text{fus}$, $Q$, $\beta$, transport-consistent profiles.

### Output Validation

<!-- OWNER COMPLETES: Define output validation rules, including:
     - Physical sanity checks (e.g., n_s > 0, T_s > 0, P_fus >= 0, Q >= 0)
     - Profile monotonicity or smoothness expectations
     - Conservation checks (e.g., particle and energy balance residuals
       below a tolerance)
     - Expected output shapes and consistency between arrays (e.g., all
       profile arrays share the same rho grid)
     - Validation of HDF5/NetCDF metadata (units, coordinate labels)
     - Comparison against reference cases for regression testing -->

---

## Governing Equations

### NEOPAX: Reduced Neoclassical Transport

Thermal transport coefficients assembled into $L_{ij}$ matrix.  Particle flux,
heat flux, and parallel flow:

$$\Gamma_a = -n_a\bigl(L_{11}A_1 + L_{12}A_2 + L_{13}A_3\bigr)$$

$$Q_a = -T_a\, n_a\bigl(L_{21}A_1 + L_{22}A_2 + L_{23}A_3\bigr)$$

$$U_{\parallel a} = -n_a\bigl(L_{31}A_1 + L_{32}A_2 + L_{33}A_3\bigr)$$

Ambipolar $E_r$ from charge-flux imbalance:

$$S_{E_r} \propto -\Gamma_e + \Gamma_D + \Gamma_T$$

Profile evolution integrated with diffrax (JAX ODE solver).

### Trinity3D: 1D Conservation Laws

$$\frac{\partial n_s}{\partial\tau} + \mathcal{G}(\rho)\frac{\partial F_{n,s}}{\partial\rho} = S_{n,s}$$

$$\frac{\partial p_s}{\partial\tau} + \mathcal{G}(\rho)\frac{\partial F_{p,s}}{\partial\rho} = \frac{2}{3}S_{p,s}$$

Implicit linearized time advance:

$$(d_1 I + \alpha\Psi)\,y^{m+1} = -d_0 y^m - d_{-1}y^{m-1} + \alpha\Psi y^m - \alpha\bigl[G(F^+ - F^-) - S\bigr] - (1-\alpha)\bigl[G(F^+_m - F^-_m) - S_m\bigr]$$

Flux Jacobians obtained by finite-differencing perturbed GX runs.

### Fusion Power (both codes)

$$P_\text{fus} = \int dV\; n_D\, n_T\, \langle\sigma v\rangle_{DT}\, E_{DT}$$

with $E_{DT} \sim 17.6$ MeV and Bosch--Hale thermal reactivity.

Reference: `stellarator_workflow.tex`, Sections 4.8--4.9.

---

## Convergence & Validity

<!-- OWNER COMPLETES: Document convergence criteria and validity checks:
     - NEOPAX: ODE solver tolerances (rtol, atol for diffrax), steady-state
       convergence criterion (e.g., max relative change in profiles < threshold),
       time-step strategy
     - Trinity3D: implicit time-step limits, CFL-like constraints, convergence
       of the linearized advance, number of GX perturbation runs per step
     - Ambipolar E_r root-finding convergence (tolerance, max iterations)
     - Physical validity: profiles remain positive, energy balance closes,
       fusion power consistent with profile integrals
     - Known failure modes and how to detect them (e.g., bifurcation in E_r,
       stiff transport leading to solver divergence) -->

---

## API Documentation

<!-- OWNER COMPLETES: Document the Python API for both codes:
     - NEOPAX: key functions/classes, their signatures, and usage patterns
       (e.g., how to load D_ij database, configure species, run profile
       evolution, extract Lij/Gamma/Q)
     - Trinity3D: how to programmatically configure and launch a run, read
       outputs, extract device metrics
     - StellaForge adapter interface: the wrapper function(s) that Stage 5
       exposes to the pipeline orchestrator, with input/output contracts
     - Example call sequences for the forward pass -->

---

## Scripts & Workflows

<!-- OWNER COMPLETES: Document the runnable scripts and workflow entry points:
     - NEOPAX: main driver script, CLI arguments, example invocations
     - Trinity3D: main driver script, TOML config examples, example invocations
     - StellaForge adapter script that wires Stage 3/4 outputs into Stage 5
     - End-to-end forward-pass example (from D_ij database to P_fus, Q)
     - Any pre-processing scripts (e.g., converting Stage 3 output format
       to NEOPAX-expected HDF5 layout)
     - Post-processing / visualization scripts for profiles and metrics -->

---

## W&B Tracking

<!-- OWNER COMPLETES: Configure Weights & Biases logging for Stage 5.
     - Project: stellaforge-stage5-transport
     - Key metrics to log per run:
       * P_fus (MW), Q (fusion gain)
       * Profile evolution: n_e(rho), T_e(rho), E_r(rho) at selected timesteps
       * Bootstrap current I_bs
       * Convergence residuals vs iteration/time
       * Wall-clock time and solver statistics
     - Artifacts to log: final HDF5 output, input config, D_ij database hash
     - Run naming convention and tagging scheme
     - Dashboard layout for comparing runs across equilibria -->

---

## Container Specification (Phase 2)

<!-- OWNER COMPLETES: Define the Docker container for Stage 5.
     - Base image (e.g., python:3-slim for CPU, nvidia/cuda:12.x for GPU JAX)
     - System-level dependencies
     - Python environment: JAX, diffrax, jaxlib, h5py, netCDF4, etc.
     - Trinity3D dependencies if included in the same container
     - Entry point and expected volume mounts (input dir, output dir)
     - Environment variables (e.g., JAX_PLATFORM_NAME, XLA flags)
     - Resource requirements: CPU/GPU, memory estimates for typical runs
     - Health check / smoke test command -->

---

## Tests (Phase 2)

<!-- OWNER COMPLETES: Define the test plan for Stage 5.
     - Unit tests:
       * L_ij matrix assembly from D_ij coefficients
       * Flux computations (Gamma, Q, Upar) against analytical limits
       * Fusion power integral against known profiles
       * Ambipolar E_r root-finding on synthetic data
     - Integration tests:
       * MONKES D_ij database (Stage 3 output) is correctly loaded and
         consumed by NEOPAX reader -- verify shapes, coordinate grids,
         and that computed fluxes match a reference case
       * SPECTRAX-GK turbulent flux coupling: verify that Stage 4 fluxes
         are correctly ingested and combined with neoclassical fluxes
       * End-to-end: Stage 1/2/3 outputs -> Stage 5 -> profiles and P_fus
     - Regression tests:
       * Known-good output files for a reference equilibrium
       * Tolerance bounds for P_fus, Q, and profile norms
     - Performance tests:
       * Wall-clock benchmarks for typical grid sizes
       * Memory usage bounds -->

---

## Claude Skills

<!-- OWNER COMPLETES: Define Claude Code skills for Stage 5 development.
     - Skill for running NEOPAX on a given equilibrium with default settings
     - Skill for comparing transport results between two runs
     - Skill for diagnosing common failure modes (E_r bifurcation, solver
       divergence, negative profiles)
     - Skill for generating summary plots of profile evolution
     - Any stage-specific linting or validation skills -->
