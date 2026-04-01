# Stage 2: Boozer Transform

## Overview

Transforms a VMEC-style equilibrium into Boozer coordinates. Boozer coordinates are the standard system for neoclassical analysis because the magnetic field takes a particularly simple covariant form: the covariant components $I(\psi)$ and $G(\psi)$ are flux-surface functions, not angle-dependent. This stage is a coordinate-transformation step, not a physics solver.

**Position in pipeline:** Receives `wout_*.nc` from Stage 1 (Equilibrium). Outputs `boozmn_*.nc` consumed by Stage 3 (Neoclassical) and indirectly by Stage 4/5.

**Reference:** `stellarator_workflow.tex`, Section 4.3.

---

## Codes

### booz_xform_jax (Primary JAX)

- **Repository:** <https://github.com/uwplasma/boozx>
- **Language:** Python/JAX
- **Notes:** Also accepts in-memory wout-like objects, enabling a fully differentiable Stage 1 -> Stage 2 path without writing an intermediate NetCDF file.

### BOOZ_XFORM (Legacy)

- **Repository:** <https://github.com/hiddenSymmetries/booz_xform>
- **Language:** Fortran/Python
- **Notes:** Standard legacy tool used by the stellarator-optimization community.

### Installation & Platform

<!-- OWNER COMPLETES: Document the installation steps for booz_xform_jax and BOOZ_XFORM.
     Include:
     - Python version and JAX version requirements for booz_xform_jax
     - Fortran compiler and build instructions for BOOZ_XFORM
     - pip/conda/pixi install commands for both codes
     - Any platform-specific notes (macOS ARM, Linux GPU, etc.)
     - How to verify a successful installation (e.g., a smoke-test command) -->

---

## Input Specification

| Field | Type | Description | Source |
|-------|------|-------------|--------|
| `wout_*.nc` | NetCDF file | Full VMEC equilibrium output | Stage 1 |

The JAX version (booz_xform_jax) does not use a separate `in_booz.*` control file. Resolution (mboz, nboz) and surface selection are specified via the Python API. The legacy BOOZ_XFORM uses an `in_booz.*` control file.

booz_xform_jax also accepts in-memory wout-like objects directly from Stage 1, bypassing file I/O.

### Input Validation

<!-- OWNER COMPLETES: Document the input validation checks that should be performed
     before running the Boozer transform. Include:
     - Required fields in wout_*.nc and how to verify they exist and are non-degenerate
     - Sanity checks on mboz/nboz (e.g., must be >= VMEC mpol/ntor, reasonable upper bounds)
     - Surface index validation (must be within the VMEC radial grid range, must exclude axis)
     - Any constraints on the equilibrium quality (e.g., force residual threshold from Stage 1)
     - Error messages or exit codes for each validation failure -->

---

## Output Specification

### Primary Output: `boozmn_*.nc` (NetCDF)

| Field | Type | Description | Used As |
|-------|------|-------------|---------|
| `bmnc_b` | 2D array (ns_b x mnboz) | $\lvert B \rvert$ in Boozer coordinates (cosine coefficients) | Downstream input |
| `rmnc_b` | 2D array | $R$ in Boozer coordinates | Geometry |
| `zmns_b` | 2D array | $Z$ in Boozer coordinates | Geometry |
| `pmns_b` | 2D array | Toroidal angle shift (sine) | Geometry |
| `gmn_b` | 2D array | Jacobian harmonics | Geometry |
| `iota_b` | 1D array | Rotational transform in Boozer coords | Downstream input |
| `pres_b` | 1D array | Pressure in Boozer coords | Downstream input |
| `beta_b` | 1D array | Beta in Boozer coords | Diagnostic |
| `phip_b` | 1D array | $d\phi/ds$ in Boozer coords | Downstream input |
| `phi_b` | 1D array | Toroidal flux in Boozer coords | Downstream input |
| `bvco_b` | 1D array | Boozer $G$ (covariant $B_\zeta$) | Downstream input |
| `buco_b` | 1D array | Boozer $I$ (covariant $B_\theta$) | Downstream input |
| `jlist` | 1D array (int) | Surface indices computed | Metadata |
| `ixm_b` | 1D array (int) | Poloidal mode numbers | Metadata |
| `ixn_b` | 1D array (int) | Toroidal mode numbers | Metadata |

### Subset Handed to Next Stage

NEO and NEO_JAX need the Boozer spectrum and radial profiles. SFINCS/sfincs_jax use the same Boozer geometry. MONKES and NEOPAX use the Boozer spectrum through direct field or file readers.

### Outputs Used as Objectives

Non-target Boozer harmonics in `bmnc_b` or symmetry-breaking measures built from them. These are **geometry diagnostics**, not transport state variables. Typical objectives include:

- Quasi-symmetry residual: sum of unwanted harmonics relative to the dominant mode.
- Mirror ratio and helical content derived from the Boozer spectrum.

### Output Validation

<!-- OWNER COMPLETES: Document the output validation checks that should be performed
     after the Boozer transform completes. Include:
     - Verify boozmn_*.nc contains all required fields listed above with correct shapes
     - Check that bmnc_b is finite and non-NaN on all computed surfaces
     - Verify iota_b matches iota from the input wout_*.nc to within a tolerance
     - Check spectral convergence: the tail of the Boozer spectrum should decay, not plateau
     - Verify bvco_b and buco_b are smooth functions of the radial coordinate
     - Cross-check: |B| reconstructed from Boozer harmonics should match |B| from VMEC on each surface
     - Define pass/fail thresholds for each check
     - Describe how validation results are logged (stdout, W&B, JSON summary) -->

---

## Governing Equations

Boozer coordinates satisfy the following dual representation of the magnetic field.

**Contravariant form:**

$$\mathbf{B} = \nabla\psi \times \nabla\theta_B + \iota(\psi)\,\nabla\zeta_B \times \nabla\psi$$

**Covariant form:**

$$\mathbf{B} = \beta\,\nabla\psi + I(\psi)\,\nabla\theta_B + G(\psi)\,\nabla\zeta_B$$

The key property of Boozer coordinates is that $I$ and $G$ are flux-surface functions (depend only on $\psi$, not on the angles). This simplifies the magnetic field strength to a pure Fourier series in $(\theta_B, \zeta_B)$ on each surface.

The Boozer angles are defined relative to the VMEC angles by:

$$\zeta_B = \zeta_0 + \nu$$

$$\theta_B = \theta_0 + \lambda + \iota\,\nu$$

The angle-shift field $\nu$ is determined from the covariant field components. The implementation uses:

$$\nu = \frac{w - I\,\lambda}{G + \iota\, I}$$

where $w$ is reconstructed from the original covariant field harmonics.

**Reference:** `stellarator_workflow.tex`, Section 4.3.

---

## Convergence & Validity

<!-- OWNER COMPLETES: Document convergence behavior and validity criteria.
     Include:
     - How to assess convergence with respect to mboz and nboz resolution
     - Typical convergence behavior: how fast do Boozer harmonics decay with mode number?
     - Known failure modes (e.g., near-axis equilibria, very high aspect ratio, near-rational surfaces)
     - Recommended resolution settings for production runs vs. quick screening
     - How to detect divergence or poor convergence at runtime
     - Comparison between booz_xform_jax and BOOZ_XFORM results for validation -->

---

## API Documentation

<!-- OWNER COMPLETES: Document the Python API for booz_xform_jax.
     Include:
     - Main entry point function(s) with full signatures and type hints
     - How to call from Python with a wout_*.nc file path
     - How to call from Python with an in-memory wout-like object (JAX arrays)
     - Return type and how to access individual output fields
     - How to write the result to boozmn_*.nc
     - Example code snippet for a minimal end-to-end call
     - Any configuration options beyond mboz/nboz (e.g., compute_surfs, verbose) -->

---

## Scripts & Workflows

<!-- OWNER COMPLETES: Document the scripts for running Stage 2.
     Include:
     - CLI invocation for booz_xform_jax (command, required args, optional args)
     - CLI invocation for BOOZ_XFORM (command, required args, optional args)
     - Example in_booz control file with comments explaining each parameter
     - How to run on a single equilibrium file
     - How to run in batch mode on multiple equilibria
     - Output directory convention: {run_dir}/stage2_boozer/
     - How results connect to Stage 3 (which files, which directory) -->

---

## W&B Tracking

**Project:** `stellaforge-stage2-boozer`

<!-- OWNER COMPLETES: Document the Weights & Biases tracking setup.
     Include:
     - What metrics to log per run (e.g., spectral decay rate, quasi-symmetry residual,
       max |bmnc_b| of unwanted harmonics, wall-clock time, mboz/nboz used)
     - What artifacts to log (boozmn_*.nc, convergence plots, input config)
     - Run naming convention
     - How to compare booz_xform_jax vs. BOOZ_XFORM results in W&B
     - Dashboard layout recommendations (panels for spectral content, convergence, etc.)
     - Integration with Stage 1 W&B project for end-to-end tracking -->

---

## Container Specification (Phase 2)

<!-- OWNER COMPLETES: Document the Docker container for Stage 2.
     Include:
     - Base image (e.g., python:3-slim for CPU, nvidia/cuda:12.x for GPU JAX)
     - Key dependencies and their pinned versions
     - Entry point command
     - Volume mount expectations: input from {run_dir}/stage1_equilibrium/, output to {run_dir}/stage2_boozer/
     - Environment variables (e.g., JAX_PLATFORM_NAME, W&B API key)
     - Health check or smoke-test command
     - Expected image size
     - GPU vs. CPU variants -->

---

## Tests (Phase 2)

<!-- OWNER COMPLETES: Document the test plan for Stage 2.
     Include:
     - Unit tests for the core Boozer transform computation
     - Regression tests: known-good boozmn_*.nc outputs for reference equilibria
     - Convergence tests: verify spectral decay with increasing mboz/nboz
     - Cross-code validation: booz_xform_jax vs. BOOZ_XFORM agreement within tolerance
     - Integration test: verify output boozmn_*.nc can be read by NEO_JAX (Stage 3)
     - Integration test: verify output boozmn_*.nc can be read by sfincs_jax (Stage 3)
     - Integration test: verify output boozmn_*.nc can be read by MONKES (Stage 3)
     - Round-trip test: |B| reconstructed from Boozer harmonics matches VMEC |B|
     - Edge case tests: single surface, all surfaces, minimal resolution
     - Performance benchmarks: wall-clock time for standard test cases -->

---

## Claude Skills

<!-- OWNER COMPLETES: Document the Claude Code skills for Stage 2.
     Include:
     - Dev skill: helps with development tasks (running booz_xform_jax, debugging,
       interpreting output, convergence analysis)
     - Operational skill: helps with production usage (running the container,
       validating outputs, troubleshooting failures)
     - What context each skill needs (file paths, config, common error patterns)
     - Skill trigger conditions (when should each skill activate?) -->
