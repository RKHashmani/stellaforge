# Stage 4: Turbulence

## Overview

Stage 4 solves the gyrokinetic equations to compute turbulent transport. The primary outputs -- heat and particle fluxes -- are both optimization objectives (to minimize) AND direct transport inputs for Stage 5.

**JAX-first priority:** SPECTRAX-GK is the primary code (JAX-native, differentiable). GX and GENE are traditional alternatives added later.

**Position in pipeline:** Receives geometry from Stage 1/2. Runs in parallel with Stage 3 (Neoclassical). Outputs feed Stage 5 (Transport).

**Important coordination point:** The coupling between SPECTRAX-GK output and NEOPAX (Stage 5) is less mature than the GX-Trinity3D coupling. NEOPAX has turbulence-coupling utilities but the public examples focus on the neoclassical reduced model. The Stage 4 and 5 owners must coordinate on this interface.

Reference: `stellarator_workflow.tex`, Section 4.7; `stellarator_io_reference.tex`, Sections 3.9-3.10.

## Codes

### SPECTRAX-GK (Primary JAX)
- **Repository:** https://github.com/uwplasma/SPECTRAX-GK
- **Language:** Python/JAX
- **Role:** JAX-native gyrokinetic solver for differentiable turbulence calculations

### GX (Alternative)
- **Repository:** https://bitbucket.org/gyrokinetics/gx
- **Language:** Fortran/CUDA
- **Role:** GPU-native gyrokinetic code, mature coupling with Trinity3D

### GENE / GENE-3D (Alternative)
- **Website:** https://genecode.org
- **Language:** Fortran
- **Role:** High-fidelity grid-based Eulerian gyrokinetic code

### Installation & Platform

<!-- OWNER COMPLETES: Document installation instructions for SPECTRAX-GK (primary), including:
     - Python/JAX version requirements and GPU/TPU backend setup
     - pip/conda/pixi install steps
     - Known platform issues (e.g., JAX on macOS ARM, CUDA version constraints)
     - Any dependency conflicts with other StellaForge stages
     - For GX: build instructions (Fortran compiler, CUDA toolkit, MPI)
     - For GENE: license and access process, build system requirements
     - Verified platform matrix (OS, GPU, JAX version combinations that are tested) -->

## Input Specification

### SPECTRAX-GK Inputs

| Field | Type | Description | Source |
|-------|------|-------------|--------|
| TOML config | file | Grid, geometry, physics toggles, time integration | User-specified |
| Geometry | analytic or `*.eik.nc` | Magnetic geometry (can be VMEC-derived) | Stage 1/2 |
| Species profiles | in config | Density, temperature, gradients per species | User-specified |
| Collisionality | in config | Collision parameters | User-specified |
| Beta | in config | Electromagnetic parameter | User-specified |

Required input fields:
- `species`: Defines the physical properties of the active plasma species.
     - Default: ion with `charge=1.0, mass=1.0, density=1.0, temperature=1.0, tprim=2.49, fprim=0.8, nu=0.0, kinetic=True`. (The code seems to have a default. But does not seem to run with not specifying it.)
- `geometry`: Specifies the magnetic equilibrium and flux surface geometry.
- `physics`: Sets the global physical assumptions for the plasma.
- `run`: Configures the execution mode.

Optional input fields:
- `grid`: Defines the resolution of the simulated phase-space.

     - Defaults: `Nx=48, Ny=48, Nz=64, Lx=62.8, Ly=62.8, boundary="periodic", jtwist=None, non_twist=False, kxfac=1.0, z_min=-pi, z_max=pi, y0=None, ntheta=None, nperiod=None, zp=None`

- `time`: Specifies time configurations.
     - Defaults: `t_max=100.0, dt=0.1, method="rk2", sample_stride=1, diagnostics_stride=1, diagnostics=True, save_state=False, checkpoint=False, implicit_restart=20, implicit_preconditioner=None, implicit_solve_method="batched", use_diffrax=True, diffrax_solver="Dopri8", diffrax_adaptive=False, diffrax_rtol=1e-5, diffrax_atol=1e-7, diffrax_max_steps=4096, progress_bar=False, fixed_dt=True, dt_min=1e-7, dt_max=None, cfl=0.9, cfl_fac=None, collision_split=False, collision_scheme="implicit", gx_real_fft=True, nonlinear_dealias=True, laguerre_nonlinear_mode="grid"`

- `init`: Controls how the initial perturbation is built.
     - Defaults: `init_field="density", init_amp=1e-5, init_single=True, random_seed=22, gaussian_init=False, gaussian_width=0.5, gaussian_envelope_constant=1.0, gaussian_envelope_sine=0.0, kpar_init=0.0, init_file=None, init_file_scale=1.0, init_file_mode="replace", init_electrons_only=False`
- `collisions`: Configures the collision operator. Controls collision, hypercollision, and end-damping parameters.
     - Defaults: `nu_hermite=1.0, nu_laguerre=2.0, nu_hyper=0.0, p_hyper=4.0, nu_hyper_l=0.0, nu_hyper_m=1.0, nu_hyper_lm=0.0, p_hyper_l=6.0, p_hyper_m=None, p_hyper_lm=6.0, D_hyper=0.0, p_hyper_kperp=2.0, hypercollisions_const=0.0, hypercollisions_kz=1.0, damp_ends_amp=0.1, damp_ends_widthfrac=0.125, damp_ends_scale_by_dt=False`. Note `p_hyper_m=None` is not a hard numeric default; the runtime follows the GX fallback min(20, Nm/2) when it is omitted.
- `normalization`: Sets the reference units used to non-dimensionalize the equations. 
     - Defaults: `contract="cyclone", rho_star=None, omega_d_scale=None, omega_star_scale=None, diagnostic_norm="gx", flux_scale=1.0, wphi_scale=1.0`
- `terms`: Controls which RHS terms are enabled, as multiplicative weights.
     - Defaults: `streaming=1.0, mirror=1.0, curvature=1.0, gradb=1.0, diamagnetic=1.0, , collisions=1.0, hypercollisions=1.0, hyperdiffusion=0.0, end_damping=1.0, apar=1.0, bpar=1.0, nonlinear=0.0`
- `experts`: Advanced special-purpose controls.
     - Defaults: `fixed_mode=False, iky_fixed=None, ikx_fixed=None, dealias_kz=False`


### GX Inputs (Alternative)

| Field | Type | Description | Source |
|-------|------|-------------|--------|
| `run_name.in` | Input file | Geometry, species, domain, time stepping, diagnostics, resolution | User-specified |
| VMEC geometry | via geometry module | Field-line geometry from wout | Stage 1 |
| `omega=true` | flag | Enable growth-rate diagnostics | Config |
| `fluxes=true` | flag | Enable flux diagnostics | Config |

### GENE Inputs (Alternative)

Installation-dependent. Key physics contract: geometry from VMEC/Boozer, species profiles/gradients, collisionality, electromagnetic parameters, numerical grid settings.

### Input Validation

- SPECTRAX-GK input validated.

<!-- OWNER COMPLETES: After running the code, validate the input tables above against actual code behavior. Document:
     - Which TOML config fields are required vs. optional for SPECTRAX-GK
     - Default values for optional fields (especially grid resolution, time step, physics toggles)
     - How geometry is loaded: analytic specification vs. eik.nc file path in config
     - Species profile format: exact key names, units, normalization conventions
     - Any fields the TeX spec missed or that are deprecated
     - Differences between SPECTRAX-GK and GX input conventions for the same physics
     - For GX: validate the .in file format and geometry module interface -->

## Output Specification

### SPECTRAX-GK Outputs

| Field | Type | Description | Used As |
|-------|------|-------------|---------|
| `t` | 1D array (time) | Simulation time | Time axis / Independent variable |
| `dt` | 1D array (time) | Time step size | Diagnostic |
| `gamma` | 1D array (time) | Growth rate time trace | Objective / screening / Convergence check |
| `omega` | 1D array (time) | Frequency time trace | Diagnostic / Convergence check |
| `Wg` | 1D array (time) | Free energy (g) trace | Diagnostic |
| `Wphi` | 1D array (time) | Free energy (phi) trace | Diagnostic |
| `Wapar` | 1D array (time) | Free energy (A_parallel) trace | Diagnostic |
| `energy` | 1D array (time) | Free energy trace | Diagnostic |
| `heat_flux` | 1D array (time) | Heat flux time trace | **Transport input** |
| `particle_flux` | 1D array (time) | Particle flux time trace | **Transport input** |
| `heat_flux_s0` | 1D array (time) | Species-resolved particle flux time trace | Diagnostic |
| `particle_flux_s0` | 1D array (time) | Species-resolved particle flux time trace |  Diagnostic |

Output in CSV files, along with a json file that records the info for only last time step.

<!-- Optional CSV output: time, growth rate, frequency, free energy, species-resolved heat and particle flux (`heat_flux_s0` and `particle_flux_s0`). -->

The natural downstream contract is the same as GX: turbulent heat and particle flux (steady-state values).

### GX Outputs (Alternative)

| File | Description |
|------|-------------|
| `run_name.out.nc` | Linear run output |
| `run_name.nc` | Nonlinear run output |
| `run_name.big.nc` | Saved field diagnostics |
| `run_name.restart.nc` | Restart data |

Key NetCDF groups: `Grids`, `Geometry`, `Diagnostics`, `Inputs`

| Field | Location | Description | Used As |
|-------|----------|-------------|---------|
| `ParticleFlux_st` | `Diagnostics/` | Particle flux (species, time) | **Transport input** (Trinity3D) |
| `HeatFlux_st` | `Diagnostics/` | Heat flux (species, time) | **Transport input** (Trinity3D) |
| `pflux` | `Fluxes/` | Particle flux (alternative location) | Transport input |
| `qflux` | `Fluxes/` | Heat flux (alternative location) | Transport input |
| `ParticleFlux_zst` | `Diagnostics/` | Zeta-resolved particle flux (stellarator) | Transport input |
| `HeatFlux_zst` | `Diagnostics/` | Zeta-resolved heat flux (stellarator) | Transport input |
| `omega_v_time` | `Special/` | Linear growth rate vs time | Screening |

GX spectral representation: Hermite-Laguerre velocity-space basis:

$$h_s = \sum_{\ell,m,k_x,k_y} \hat{h}_{s,\ell,m}(z,t)\, e^{i(k_x x + k_y y)} H_m\left(\frac{v_\parallel}{v_{ts}}\right) L_\ell\left(\frac{v_\perp^2}{v_{ts}^2}\right) F_{Ms}$$

### GENE Outputs (Alternative)

Installation-dependent filenames. Key outputs: linear growth rates, real frequencies, eigenfunctions, nonlinear species heat/particle fluxes, spectra, time histories.

### Subset Handed to Next Stage

For transport coupling, the critical handoff is the **turbulent flux vector** (steady-state heat and particle flux per species). For screening, only linear gamma and omega may be retained.

Trinity3D obtains flux Jacobians by rerunning GX on perturbed gradients and finite-differencing.

### Outputs Used as Objectives

- Linear gamma, omega: rapid screening
- Nonlinear heat flux, particle flux: high-fidelity design objectives
- Heat flux is BOTH an objective AND a transport input (dual-role output)

### Output Validation

- SPECTRAX-GK output valified.

<!-- OWNER COMPLETES: Run actual gyrokinetic calculations and verify the output fields listed above. Document:
     - Exact output format for SPECTRAX-GK: file type (HDF5, NetCDF, CSV, in-memory), field names, array shapes
     - Units and normalization conventions for gamma, omega, heat flux, particle flux
     - How to extract steady-state flux values from time traces (averaging window, convergence criteria)
     - For GX: verify NetCDF group structure and field names against actual output files
     - Any additional output fields not listed here
     - Shape discrepancies or differences between linear and nonlinear run outputs
     - How species-resolved outputs are indexed (species ordering convention) -->

## Governing Equations

Generic delta-f gyrokinetic equation:

$$\frac{\partial h_s}{\partial t} + v_\parallel \mathbf{b}\cdot\nabla h_s + \mathbf{v}_{Ds}\cdot\nabla h_s + \mathbf{v}_E\cdot\nabla h_s - C[h_s] = -\frac{Z_s e F_{Ms}}{T_s}\frac{\partial\langle\chi\rangle}{\partial t} - \mathbf{v}_\chi\cdot\nabla F_{Ms}$$

Closed by quasineutrality and (for electromagnetic calculations) appropriate field equations.

Reference: `stellarator_workflow.tex`, Section 4.7.

## Convergence & Validity

<!-- OWNER COMPLETES: Document the following after running the code:
     - What stellarator geometries are tested and known to work (e.g., Landreman-Paul QA/QH, W7-X, NCSX)
     - Linear vs. nonlinear convergence criteria: how to determine when a linear growth rate is converged, when nonlinear fluxes have reached a statistical steady state
     - Resolution requirements: velocity-space (Hermite/Laguerre modes for GX, equivalent for SPECTRAX-GK), spatial (kx, ky, z grids), and time step constraints
     - Known failure modes: geometries that cause numerical instability, parameter regimes that are problematic
     - Comparison between SPECTRAX-GK and GX results for benchmark cases (if available)
     - Cost estimates: typical wall-clock time for linear scans vs. nonlinear runs -->

## API Documentation

<!-- OWNER COMPLETES: Document the following:
     - Key entry-point functions for SPECTRAX-GK with full signatures (Python/JAX)
     - How to run a single linear calculation programmatically
     - How to run a nonlinear flux calculation programmatically
     - How to extract growth rates and fluxes from the output object
     - JAX differentiation: how to obtain gradients of gamma or flux with respect to inputs
     - For GX: Python wrapper interface (if any), or command-line invocation pattern
     - Configuration parameters and their effects on physics fidelity vs. cost -->

## Scripts & Workflows

<!-- OWNER COMPLETES: Provide the following:
     - How to run SPECTRAX-GK standalone (CLI and Python)
     - Example: linear growth rate scan over a range of ky values
     - Example: nonlinear flux calculation for a given equilibrium
     - How to convert Stage 1/2 output (wout or Boozer) into SPECTRAX-GK geometry input
     - Common debugging workflows (e.g., diagnosing non-convergence, energy conservation checks)
     - How to visualize growth rate spectra and flux time traces
     - For GX: equivalent standalone workflow examples -->

## W&B Tracking

<!-- OWNER COMPLETES: Set up and document:
     - W&B project: stellaforge-stage4-turbulence
     - What metrics to log: growth rates (gamma, omega) vs. ky, flux time traces, steady-state flux values, runtime, resolution parameters
     - Key dashboard panels: growth rate spectrum, flux convergence, cost vs. fidelity
     - Run naming convention
     - How to tag linear-only vs. nonlinear runs
     - Logging of input geometry metadata (which equilibrium, which flux surface) -->

## Container Specification (Phase 2)

<!-- OWNER COMPLETES: Define the following during Phase 2:
     - Base image and key dependencies (JAX version, GPU drivers, etc.)
     - Dockerfile entry point for SPECTRAX-GK
     - Expected volume mounts (input geometry dir, output dir, config dir)
     - Environment variables (e.g., JAX platform, GPU memory settings)
     - Resource requirements: GPU type/memory for nonlinear runs, CPU fallback for linear runs
     - For GX container: Fortran/CUDA build layer, MPI configuration
     - Multi-code container strategy: separate images per code or combined -->

## Tests (Phase 2)

<!-- OWNER COMPLETES: Write the following during Phase 2:
     - Unit tests for mathematical invariants (e.g., energy conservation in collisionless limit, flux-surface averaging properties)
     - Regression tests: known-good growth rates and fluxes for benchmark stellarator cases, compared with tolerances
     - Benchmark: compare SPECTRAX-GK results against published GX or GENE results for the same geometry
     - Integration test: Stage 1/2 geometry output can be loaded and produces valid growth rates
     - Integration test with Stage 5: SPECTRAX-GK flux output can be consumed by NEOPAX turbulence coupling (this is the critical cross-stage test -- coordinate with Stage 5 owner)
     - Acceptance criteria: definition of "done" for this stage (linear AND nonlinear capabilities) -->

## Claude Skills

<!-- OWNER COMPLETES: Create the following Claude skills:
     - Dev skill: how to run SPECTRAX-GK, interpret growth rates and fluxes, debug convergence, understand the delta-f gyrokinetic formulation, navigate the codebase
     - Operational skill: how to build the container, run the test suite, validate outputs against known-good results, set up GX as an alternative backend
     - Cross-stage skill: how to coordinate the Stage 4 -> Stage 5 handoff, especially the SPECTRAX-GK -> NEOPAX turbulence coupling interface -->
