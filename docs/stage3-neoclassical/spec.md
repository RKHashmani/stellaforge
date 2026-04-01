# Stage 3: Neoclassical Transport

## Overview

Stage 3 computes neoclassical transport properties from the Boozer-coordinate equilibrium. It contains three sub-stages that can run in parallel, each serving a distinct role:

1. **NEO / NEO_JAX** -- Computes effective ripple (epsilon_eff), a screening/optimization diagnostic. **NOT a transport state variable** -- does not feed into profile evolution.
2. **SFINCS / sfincs_jax** -- Solves the full drift-kinetic equation for neoclassical particle flux, heat flux, bootstrap current, and ambipolar E_r. Direct transport input for Trinity3D.
3. **MONKES** -- Builds a monoenergetic D_ij transport coefficient database. Direct input for NEOPAX (Stage 5).

**Critical distinction:** NEO's epsilon_eff is a screening metric used for ranking configurations. SFINCS fluxes and MONKES D_ij coefficients are actual transport inputs consumed by Stage 5.

**Position in pipeline:** NEO_JAX and MONKES receive `boozmn_*.nc` from Stage 2 (Boozer). sfincs_jax receives `wout_*.nc` directly from Stage 1 (Equilibrium). The three sub-stages fan in to Stage 5 (Transport). Stage 3 runs in parallel with Stage 4 (Turbulence).

**Reference:** `stellarator_workflow.tex`, Sections 4.4-4.6; `stellarator_io_reference.tex`, Sections 3.4-3.6.

---

## Sub-Stage 3a: NEO / NEO_JAX (Effective Ripple)

### Codes

**NEO_JAX (Primary JAX):** <https://github.com/uwplasma/NEO_JAX>

**NEO (Legacy, part of STELLOPT):** <https://github.com/PrincetonUniversity/STELLOPT>

### Installation & Platform

<!-- OWNER COMPLETES: Document installation steps for NEO_JAX and legacy NEO.
     Include:
     - Python version and JAX version requirements for NEO_JAX
     - Fortran compiler and build instructions for NEO (from STELLOPT suite)
     - pip/conda/pixi install commands for NEO_JAX
     - Any platform-specific notes (macOS ARM, Linux GPU, etc.)
     - How to verify a successful installation (e.g., a smoke-test command)
     - Known dependency conflicts with other sub-stages -->

### Input Specification

| Field | Type | Description | Source |
|-------|------|-------------|--------|
| `boozmn_*.nc` | NetCDF file | Boozer-coordinate equilibrium | Stage 2 |
| `neo_in.*` / `neo_param.*` | Control file | Surface list, angular resolution (theta_n, phi_n), Fourier cutoffs, MC controls, accuracy targets, current calculation switch (CALC_CUR) | User-specified |

#### Input Validation

<!-- OWNER COMPLETES: Document input validation checks for NEO / NEO_JAX.
     Include:
     - Required fields in boozmn_*.nc and how to verify they exist
     - Sanity checks on angular resolution parameters (theta_n, phi_n)
     - Surface index validation (must be within the Boozer radial grid range)
     - Fourier cutoff validation relative to Boozer spectrum resolution
     - CALC_CUR switch: when is it appropriate to enable/disable?
     - Error messages or exit codes for each validation failure
     - Differences between NEO_JAX and legacy NEO input requirements -->

### Output Specification

**Primary output:** `neo_out.*` and `neolog.*`

| Field | Type | Description | Used As |
|-------|------|-------------|---------|
| `epstot` | 1D array (per surface) | epsilon_eff^{3/2} (total effective ripple) | **Screening objective only** |
| `epspar` | 1D array | Parallel epsilon | Diagnostic |
| `reff` | 1D array | Effective radius | Diagnostic |
| `iota` | 1D array | Rotational transform | Cross-check |
| `b_ref` | scalar | Reference magnetic field | Normalization |
| `r_ref` | scalar | Reference radius | Normalization |
| `ctrone` | 1D array | Contribution from one class | Diagnostic |
| `ctrtot` | 1D array | Total contribution | Diagnostic |
| `bareph` | 1D array | Parallel epsilon (bar) | Diagnostic |
| `barept` | 1D array | Perpendicular epsilon (bar) | Diagnostic |
| `yps` | 1D array | Normalized toroidal flux | Coordinate |

**NEO_JAX result objects:** `epsilon_effective`, `epsilon_effective_by_class`

Optional outputs (if CALC_CUR=1): `neo_cur.*`, `current.dat`, `conver.dat`, `diagnostic.dat`, `diagnostic_add.dat`, `diagnostic_bigint.dat`

**Role:** Screening/optimization diagnostic. Usually NO direct transport consumer. epsilon_eff is NOT what Trinity3D or NEOPAX advances in time.

#### Output Validation

<!-- OWNER COMPLETES: Document output validation checks for NEO / NEO_JAX.
     Include:
     - Verify neo_out.* contains all expected fields with correct shapes
     - Check that epstot is finite, non-NaN, and non-negative on all surfaces
     - Physical range checks: epsilon_eff^{3/2} should be in [0, ~0.5] for reasonable configs
     - Verify iota cross-checks against Stage 2 boozmn values within tolerance
     - Check per-class contributions sum to epstot
     - Convergence check: epstot should not change significantly with increased resolution
     - Cross-code validation: NEO_JAX vs. legacy NEO agreement within tolerance
     - Describe how validation results are logged (stdout, W&B, JSON summary) -->

### Governing Equations

Field-line integrals:

$$y_2 = \int d\phi\, B^{-2}, \quad y_3 = \int d\phi\, |\nabla\psi| B^{-2}, \quad y_4 = \int d\phi\, K_G B^{-3}$$

Trapped-particle integrals:

$$I_f = \int d\phi\, \sqrt{1 - \frac{B}{B_0 \eta}}\, B^{-2}$$

$$H_f = \int d\phi\, \sqrt{1 - \frac{B}{B_0 \eta}} \left(\frac{4}{B/B_0} - \frac{1}{\eta}\right) \frac{K_G}{\sqrt{\eta}} B^{-2}$$

Class-resolved effective ripple:

$$\epsilon_{\text{eff}}^{3/2}(m) = C_\epsilon \frac{y_2}{y_3^2} \text{BigInt}(m), \quad C_\epsilon = \frac{\pi R_0^2 \Delta\eta}{8\sqrt{2}}$$

Total `epstot` is the sum over classes.

**Reference:** `stellarator_workflow.tex`, Section 4.4.

### Convergence & Validity

<!-- OWNER COMPLETES: Document convergence behavior and validity criteria for NEO / NEO_JAX.
     Include:
     - How to assess convergence with respect to angular resolution (theta_n, phi_n)
     - Convergence with respect to Fourier cutoffs
     - How number of trapped-particle classes affects results
     - Known failure modes (e.g., near-axis equilibria, very low aspect ratio)
     - Recommended resolution settings for production runs vs. quick screening
     - How to detect divergence or poor convergence at runtime
     - Comparison between NEO_JAX and legacy NEO for validation -->

### API Documentation

<!-- OWNER COMPLETES: Document the Python API for NEO_JAX.
     Include:
     - Main entry point function(s) with full signatures and type hints
     - How to call from Python with a boozmn_*.nc file path
     - How to call from Python with an in-memory Boozer object (JAX arrays)
     - Return type and how to access epsilon_effective and per-class results
     - How to write results to neo_out.* files
     - Example code snippet for a minimal end-to-end call
     - Any configuration options beyond the control file parameters -->

### Scripts & Workflows

<!-- OWNER COMPLETES: Document the scripts for running Sub-Stage 3a.
     Include:
     - CLI invocation for NEO_JAX (command, required args, optional args)
     - CLI invocation for legacy NEO (command, required args, optional args)
     - Example neo_in control file with comments explaining each parameter
     - How to run on a single boozmn file
     - How to run in batch mode on multiple equilibria
     - Output directory convention: {run_dir}/stage3_neoclassical/neo/
     - How results connect to optimization (screening use, NOT Stage 5 input) -->

---

## Sub-Stage 3b: SFINCS / sfincs_jax (Full Neoclassical)

### Codes

**sfincs_jax (Primary JAX):** <https://github.com/uwplasma/sfincs_jax>

**SFINCS (Legacy):** <https://github.com/landreman/sfincs>

### Installation & Platform

<!-- OWNER COMPLETES: Document installation steps for sfincs_jax and legacy SFINCS.
     Include:
     - Python version and JAX version requirements for sfincs_jax
     - Fortran/PETSc/MPI build requirements for legacy SFINCS
     - pip/conda/pixi install commands for sfincs_jax
     - PETSc version constraints and configuration flags for legacy SFINCS
     - Any platform-specific notes (macOS ARM, Linux GPU, etc.)
     - How to verify a successful installation (e.g., a smoke-test command)
     - Known dependency conflicts with other sub-stages -->

### Input Specification

| Field | Type | Description | Source |
|-------|------|-------------|--------|
| `wout_*.nc` | NetCDF file | VMEC equilibrium output (referenced via `equilibriumFile` in the input file) | Stage 1 |
| `input.*` | Fortran-style text file | Configuration with species, gradients, resolution, and `equilibriumFile` path | User-specified |

Key namelist parameters: species charges/masses, $\hat{n}_s$, $\hat{T}_s$, their gradients, collision model, E_r guess, Phi1 switches, numerical resolution.

#### Input Validation

<!-- OWNER COMPLETES: Document input validation checks for SFINCS / sfincs_jax.
     Include:
     - Required namelist groups and their mandatory fields
     - Species parameter validation (charges, masses, density/temperature consistency)
     - Gradient sanity checks (sign conventions, physical range)
     - Resolution parameter validation (Ntheta, Nzeta, Nxi, Nx ranges)
     - Geometry source validation (verify boozmn file is readable and has required fields)
     - E_r initial guess: guidance on choosing reasonable values
     - Phi1 switch: when to enable and implications for output fields
     - Error messages or exit codes for each validation failure
     - Differences between sfincs_jax and legacy SFINCS input handling -->

### Output Specification

**Primary output:** `sfincsOutput.h5` (HDF5)

| Field | Type | Description | Used As |
|-------|------|-------------|---------|
| `particleFlux_vm_rN` | array (per species) | Particle flux (vm normalization, rN coord) | **Transport input** (Trinity3D) |
| `heatFlux_vm_rN` | array (per species) | Heat flux (vm normalization, rN coord) | **Transport input** (Trinity3D) |
| `particleFlux_vd_rN` | array | Particle flux (with Phi1 enabled) | Transport input (alt) |
| `heatFlux_vd_rN` | array | Heat flux (with Phi1 enabled) | Transport input (alt) |
| `FSABjHat` | array | Flux-surface averaged bootstrap current | Transport/equilibrium feedback |
| `FSABFlow` | array | Flux-surface averaged flow | Diagnostic |
| `Phi1Hat` | array | First-order electrostatic potential | Diagnostic |
| `transportMatrix` | 2D array | Full transport matrix | Analysis |

Also: matching momentum-flux arrays, classical fluxes, optional full-f/delta-f exports, `*_vs_x` lineouts. sfincs_jax additionally exposes in-memory result dicts and can write `.npy` state vectors.

**Handoff to Trinity3D:** The Trinity3D adapter reads `particleFlux_vm_rN` and `heatFlux_vm_rN` when `includePhi1` is off, and `particleFlux_vd_rN` / `heatFlux_vd_rN` when on.

#### Output Validation

<!-- OWNER COMPLETES: Document output validation checks for SFINCS / sfincs_jax.
     Include:
     - Verify sfincsOutput.h5 contains all required fields with correct shapes
     - Check that fluxes are finite and non-NaN for all species
     - Ambipolarity check: sum of Z_s * particleFlux across species should be near zero
     - Bootstrap current sign and magnitude sanity check
     - Transport matrix symmetry checks (Onsager symmetry: L_ij = L_ji)
     - Physical range checks for fluxes (order-of-magnitude expectations)
     - Convergence check: fluxes should not change significantly with increased resolution
     - Cross-code validation: sfincs_jax vs. legacy SFINCS agreement within tolerance
     - Verify Phi1 outputs are present when includePhi1 is enabled
     - Describe how validation results are logged (stdout, W&B, JSON summary) -->

### Governing Equations

First-order drift-kinetic equation:

$$(v_\parallel \mathbf{b} + \frac{d\Phi_0}{dr}\frac{\mathbf{B}\times\nabla r}{B^2})\cdot\nabla f_{s1} + [\text{mirror/drift terms}]\frac{\partial f_{s1}}{\partial\xi} - (\mathbf{v}_{ms}\cdot\nabla r)\frac{Z_s e}{2T_s x_s}\frac{d\Phi_0}{dr}\frac{\partial f_{s1}}{\partial x_s}$$

$$+ (\mathbf{v}_{ms}\cdot\nabla r)\left[\frac{1}{n_s}\frac{dn_s}{dr} + \frac{Z_s e}{T_s}\frac{d\Phi_0}{dr} + (x_s^2 - \frac{3}{2})\frac{1}{T_s}\frac{dT_s}{dr}\right]f_{sM} = C_s[f_{s1}] + S_s$$

Collision operator: $C_s[f_s] = \sum_b C_{sb}^l[f_s, f_b]$ with Lorentz, energy-diffusion, and field-particle components.

When Phi1 is included, coupled to quasineutrality.

**Reference:** `stellarator_workflow.tex`, Section 4.5.

### Convergence & Validity

<!-- OWNER COMPLETES: Document convergence behavior and validity criteria for SFINCS / sfincs_jax.
     Include:
     - Resolution parameters and their convergence behavior (Ntheta, Nzeta, Nxi, Nx)
     - Typical convergence order: which parameter needs the most points?
     - How to perform a convergence scan (vary one parameter at a time)
     - Known failure modes (e.g., very low collisionality, strong E_r shear)
     - Impact of collision model choice on results
     - When Phi1 is needed vs. when it can be neglected
     - Recommended resolution settings for production runs vs. quick screening
     - How to detect solver failure or poor convergence at runtime
     - Comparison between sfincs_jax and legacy SFINCS for validation -->

### API Documentation

<!-- OWNER COMPLETES: Document the Python API for sfincs_jax.
     Include:
     - Main entry point function(s) with full signatures and type hints
     - How to call from Python with a boozmn_*.nc file path
     - How to call from Python with an in-memory Boozer object (JAX arrays)
     - Return type and how to access flux, bootstrap current, and transport matrix results
     - How to write results to sfincsOutput.h5
     - How to export .npy state vectors for warm-start / checkpointing
     - Example code snippet for a minimal end-to-end call
     - Namelist generation utilities (if any)
     - How to perform an E_r scan programmatically -->

### Scripts & Workflows

<!-- OWNER COMPLETES: Document the scripts for running Sub-Stage 3b.
     Include:
     - CLI invocation for sfincs_jax (command, required args, optional args)
     - CLI invocation for legacy SFINCS (command, MPI launch, required args)
     - Example input.namelist with comments explaining each group and key parameters
     - How to run on a single surface
     - How to run a radial scan (multiple surfaces)
     - How to run an E_r scan for ambipolar root-finding
     - Output directory convention: {run_dir}/stage3_neoclassical/sfincs/
     - How results connect to Stage 5 (Trinity3D adapter reads which fields) -->

---

## Sub-Stage 3c: MONKES (Monoenergetic Coefficients)

### Codes

**MONKES (JAX-native):** <https://github.com/f0uriest/monkes>

### Installation & Platform

<!-- OWNER COMPLETES: Document installation steps for MONKES.
     Include:
     - Python version and JAX version requirements
     - pip/conda/pixi install commands
     - Any platform-specific notes (macOS ARM, Linux GPU, etc.)
     - How to verify a successful installation (e.g., a smoke-test command)
     - Known dependency conflicts with other sub-stages
     - DESC dependency: is DESC required as a geometry backend? -->

### Input Specification

| Field | Type | Description | Source |
|-------|------|-------------|--------|
| Field object | DESC equilibrium or equivalent | Magnetic field representation at a single surface | Stage 1 (DESC) or Stage 2 |
| Species Maxwellians | in-memory | Species definitions | User-specified |
| E_r | scalar (float) | Radial electric field | User-specified / scan |
| Speed / nu | scalar (float) | Particle speed / collisionality | User-specified / scan |
| Pitch-angle resolution | int | Number of Legendre modes | User-specified |

In database-building mode: the solve is repeated over collisionality (nu_v) and E_r grids.

#### Input Validation

<!-- OWNER COMPLETES: Document input validation checks for MONKES.
     Include:
     - Field object validation: required attributes and methods
     - Species Maxwellian parameter validation (temperature, density, charge, mass)
     - E_r range and grid spacing guidance for database building
     - Collisionality (nu_v) range and grid spacing guidance for database building
     - Pitch-angle resolution: minimum and recommended values
     - Surface selection: how to specify which flux surface to compute on
     - Error messages or exit codes for each validation failure
     - How MONKES handles DESC equilibria vs. boozmn-based equilibria -->

### Output Specification

**Core output:** D_ij transport matrix (in-memory or HDF5)

| Field | Type | Description | Used As |
|-------|------|-------------|---------|
| `D11` | array | Monoenergetic transport coefficient (1,1) | **NEOPAX input** |
| `D13` | array | Monoenergetic transport coefficient (1,3) | **NEOPAX input** |
| `D33` | array | Monoenergetic transport coefficient (3,3) | **NEOPAX input** |
| `D12`, `D21`, `D22`, `D23`, `D31`, `D32` | arrays | Remaining matrix elements | Analysis |
| `Er` | array | Radial electric field grid | **NEOPAX input** |
| `Er_tilde` | array | Normalized E_r | **NEOPAX input** |
| `drds` | array | Radial coordinate Jacobian | **NEOPAX input** |
| `rho` | array | Radial coordinate | **NEOPAX input** |
| `nu_v` | array | Collisionality grid | **NEOPAX input** |
| `f` | array | Perturbed distribution function | Analysis only |
| `s` | array | Source vector | Analysis only |

**Key handoff:** NEOPAX's database reader consumes the **reduced subset**: D11, D13, D33, Er, Er_tilde, drds, rho, nu_v. The full distribution f and source s are NOT part of that handoff.

#### Output Validation

<!-- OWNER COMPLETES: Document output validation checks for MONKES.
     Include:
     - Verify D_ij matrix contains all required fields with correct shapes
     - Check that D11, D13, D33 are finite and non-NaN across the full (nu_v, Er) grid
     - Physical range checks: D11 > 0 (positive-definite diffusion), D33 > 0
     - Onsager symmetry check: D_ij = D_ji within tolerance
     - Check limiting behavior: D11 ~ 1/nu at low collisionality (1/nu regime)
     - Check plateau regime behavior at intermediate collisionality
     - Verify grid coverage: nu_v and Er grids must span the range needed by NEOPAX
     - Database completeness check: no missing grid points
     - Convergence check: D_ij should not change significantly with increased Legendre resolution
     - Describe how validation results are logged (stdout, W&B, JSON summary) -->

### Governing Equations

Legendre-expanded pitch-angle operators:

$$L_k(f) = \frac{k}{2k-1}\left[\mathbf{b}\cdot\nabla f + \frac{k-1}{2}\frac{\mathbf{b}\cdot\nabla B}{B}f\right]$$

$$U_k(f) = \frac{k+1}{2k+3}\left[\mathbf{b}\cdot\nabla f - \frac{k+2}{2}\frac{\mathbf{b}\cdot\nabla B}{B}f\right]$$

$$D_k(f) = -\frac{\hat{E}_r}{\psi_r\langle B^2\rangle}\mathbf{B}\times\nabla\psi\cdot\nabla f + \frac{k(k+1)}{2}\hat\nu f$$

The monoenergetic coefficients are assembled into:

$$D_{ij} = \begin{pmatrix} D_{11} & D_{12} & D_{13} \\ D_{21} & D_{22} & D_{23} \\ D_{31} & D_{32} & D_{33} \end{pmatrix}$$

**Reference:** `stellarator_workflow.tex`, Section 4.6.

### Convergence & Validity

<!-- OWNER COMPLETES: Document convergence behavior and validity criteria for MONKES.
     Include:
     - How to assess convergence with respect to Legendre mode count
     - How to assess convergence with respect to spatial resolution (theta, zeta)
     - Typical convergence behavior across different collisionality regimes
     - Known failure modes (e.g., extremely low collisionality, very strong E_r)
     - Recommended resolution settings for production database building vs. quick checks
     - How to detect solver failure or poor convergence at runtime
     - Validation against published monoenergetic coefficient databases -->

### API Documentation

<!-- OWNER COMPLETES: Document the Python API for MONKES.
     Include:
     - Main entry point function(s) with full signatures and type hints
     - How to construct the Field object from a DESC equilibrium
     - How to construct the Field object from a boozmn_*.nc file (if supported)
     - How to specify species and scanning parameters
     - Return type and how to access D_ij matrix elements
     - How to build a full (nu_v, Er) database programmatically
     - How to write the database to HDF5 in the format NEOPAX expects
     - Example code snippet for a minimal single-point solve
     - Example code snippet for building a full database -->

### Scripts & Workflows

<!-- OWNER COMPLETES: Document the scripts for running Sub-Stage 3c.
     Include:
     - CLI invocation for MONKES (command, required args, optional args)
     - How to run a single-point solve
     - How to run a database-building scan over (nu_v, Er) grids
     - Grid spacing recommendations for the database
     - Output directory convention: {run_dir}/stage3_neoclassical/monkes/
     - Output file naming convention for the database
     - How results connect to Stage 5 (NEOPAX database reader interface) -->

---

## Stage-Wide Sections

The following sections apply to Stage 3 as a whole, covering all three sub-stages.

---

## W&B Tracking

**Project:** `stellaforge-stage3-neoclassical`

<!-- OWNER COMPLETES: Document the Weights & Biases tracking setup for all three sub-stages.
     Include:
     - Metric groups: separate metric prefixes or groups for NEO_JAX, sfincs_jax, and MONKES
     - NEO_JAX metrics: epstot per surface, per-class contributions, wall-clock time,
       resolution parameters used, convergence indicators
     - sfincs_jax metrics: particle flux, heat flux, bootstrap current per species,
       E_r root, solver iterations, wall-clock time, resolution parameters
     - MONKES metrics: D11/D13/D33 over (nu_v, Er) grid, solver iterations per grid point,
       wall-clock time, Legendre resolution used
     - Artifacts to log per sub-stage: output files, convergence plots, input configs
     - Run naming convention (should encode which sub-stage and which equilibrium)
     - Cross-code comparison panels: NEO_JAX vs. NEO, sfincs_jax vs. SFINCS
     - Dashboard layout recommendations (sub-stage tabs or grouped panels)
     - Integration with Stage 2 W&B project for traceability
     - Integration with Stage 5 W&B project for end-to-end tracking -->

---

## Container Specification (Phase 2)

<!-- OWNER COMPLETES: Document the Docker container(s) for Stage 3.
     Key design decision: one container for all three sub-stages or separate containers?
     For each container, include:
     - Base image (e.g., python:3-slim for CPU, nvidia/cuda:12.x for GPU JAX)
     - Key dependencies and their pinned versions (JAX, DESC if needed by MONKES, etc.)
     - Entry point command(s) -- one per sub-stage or a dispatcher
     - Volume mount expectations:
       - Input from {run_dir}/stage2_boozer/ (boozmn_*.nc)
       - Output to {run_dir}/stage3_neoclassical/neo/, sfincs/, monkes/ respectively
     - Environment variables (e.g., JAX_PLATFORM_NAME, W&B API key)
     - Health check or smoke-test command per sub-stage
     - Expected image size
     - GPU vs. CPU variants
     - Parallelism: how to launch all three sub-stages concurrently within the pipeline -->

---

## Tests (Phase 2)

<!-- OWNER COMPLETES: Document the test plan for Stage 3 (all sub-stages).

     NEO_JAX unit tests:
     - epsilon_eff computation for known analytical cases
     - Field-line integral accuracy
     - Trapped-particle class identification
     - Regression tests against known-good neo_out files for reference equilibria
     - Cross-code validation: NEO_JAX vs. legacy NEO agreement within tolerance

     sfincs_jax unit tests:
     - Transport matrix symmetry (Onsager relations)
     - Ambipolarity of particle fluxes
     - Bootstrap current sign and magnitude for known cases
     - Regression tests against known-good sfincsOutput.h5 for reference equilibria
     - Cross-code validation: sfincs_jax vs. legacy SFINCS agreement within tolerance

     MONKES unit tests:
     - D_ij positive-definiteness (D11 > 0, D33 > 0)
     - Onsager symmetry of coefficient matrix
     - Limiting behavior in 1/nu and plateau regimes
     - Regression tests against published monoenergetic coefficient databases
     - Convergence with Legendre mode count

     Integration tests (Stage 3 -> Stage 5):
     - Verify NEO_JAX outputs can be read by downstream screening/optimization tools
     - Verify sfincsOutput.h5 can be read by Trinity3D adapter
     - Verify MONKES D_ij database can be read by NEOPAX database reader
     - Verify the reduced NEOPAX subset (D11, D13, D33, Er, Er_tilde, drds, rho, nu_v)
       is complete and correctly formatted

     Edge case tests:
     - Single surface, all surfaces, minimal resolution
     - Extreme collisionality values
     - Zero E_r and strong E_r limits

     Performance benchmarks:
     - Wall-clock time for standard test cases per sub-stage
     - Database-building time for MONKES at production grid density -->

---

## Claude Skills

<!-- OWNER COMPLETES: Document the Claude Code skills for Stage 3.

     Dev skills (one per sub-stage):
     - NEO_JAX dev skill: running NEO_JAX, debugging epsilon_eff computations,
       interpreting per-class contributions, convergence analysis, comparing
       NEO_JAX vs. legacy NEO results
     - sfincs_jax dev skill: running sfincs_jax, debugging drift-kinetic solves,
       interpreting flux and bootstrap current results, E_r root-finding,
       convergence analysis, comparing sfincs_jax vs. legacy SFINCS results
     - MONKES dev skill: running MONKES, building D_ij databases, debugging
       solver convergence, interpreting coefficient behavior across regimes,
       validating database completeness for NEOPAX

     Operational skills (one per sub-stage or combined):
     - Running the container(s), validating outputs, troubleshooting failures
     - Verifying that outputs are valid inputs for Stage 5 consumers
     - Monitoring database-building progress for MONKES

     For each skill, specify:
     - What context the skill needs (file paths, config, common error patterns)
     - Skill trigger conditions (when should each skill activate?)
     - Key domain knowledge the skill should encode (physics, numerics, common pitfalls) -->
