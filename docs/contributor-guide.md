# StellaForge Contributor Guide

## Getting Started

1. Clone the repository and initialize submodules:
   ```bash
   git clone https://github.com/RKHashmani/StellaForge.git
   cd StellaForge
   git submodule update --init --recursive
   ```

2. Read the architecture document first: `docs/architecture.md`

3. Find your stage's specification: `docs/stageN-*/spec.md` (e.g., `docs/stage1-equilibrium/spec.md`)

4. Read the project coding standards: `CLAUDE.md` (at the repository root)

5. Review the reference TeX manuscripts in `stellarator_workflow/` for physics context:
   - `stellarator_workflow.tex` -- governing equations and code details
   - `stellarator_io_reference.tex` -- I/O contracts and handoff specifications

## Your Role as a Stage Owner

Each stage has one owner responsible for their stage end-to-end across two phases. Your stage's `spec.md` has pre-populated sections (extracted from the TeX manuscripts) and clearly marked "OWNER COMPLETES" sections that you need to fill in.

## Phase 1: Document & Run

Work through these steps in order. Each step should result in updates to your stage's `spec.md`.

### Step 1: Install and Run the Primary JAX Code

- Install the primary JAX code for your stage (listed in your spec's Codes section)
- Get it running locally on a reference case
- Document any installation issues, version requirements, or platform-specific notes
- Fill in the "Installation & Platform" section of your spec

### Step 2: Run a Reference Case and Capture I/O

- Run the code on a standard test case (e.g., a known stellarator configuration)
- Capture the actual input and output files
- Inspect the output files to understand their structure:
  - For NetCDF: `ncdump -h output_file.nc` (shows all variables, dimensions, attributes)
  - For HDF5: `h5dump -H output_file.h5` (shows structure without data)
  - In Python: use `xarray.open_dataset()` for NetCDF or `h5py.File()` for HDF5

### Step 3: Validate the I/O Specification

- Compare the pre-populated I/O tables in your spec against the actual code behavior
- Correct any discrepancies (wrong field names, missing fields, incorrect types/shapes)
- Add any fields that the TeX spec missed
- Note any deprecated or code-specific fields
- Fill in the "Input Validation" and "Output Validation" sections

### Step 4: Document the API

- Identify the key entry-point functions (for the primary JAX code)
- Document their full signatures (parameters with types, return values)
- Document configuration parameters and their effects
- Provide programmatic usage examples in Python/JAX
- Fill in the "API Documentation" section

### Step 5: Document Convergence & Validity

- Identify input configurations that are known to converge (cite specific stellarator configs if possible)
- Identify configurations that fail or diverge, and explain why
- Document convergence criteria and tolerances
- Note known numerical edge cases
- Fill in the "Convergence & Validity" section

### Step 6: Write Example Scripts

- Provide scripts that demonstrate how to run the code standalone (CLI and Python)
- Include sample input data or instructions for obtaining it
- Document common debugging workflows
- Fill in the "Scripts & Workflows" section

### Step 7: Set Up W&B Tracking

Set up Weights & Biases tracking for your stage:

- **Project name:** `stellaforge-stage{N}-{name}` (e.g., `stellaforge-stage1-equilibrium`)
- **Run naming:** `{code}_{config}_{timestamp}` (e.g., `vmec_jax_qa_2026-04-01T12:00`)
- **Metrics to log:** Stage-specific convergence metrics, runtime, key physics outputs (see your spec for guidance)
- Create a dashboard with the most important panels for your stage
- Fill in the "W&B Tracking" section

### Step 8: Create Claude Skills

Create two types of Claude Code skills for your stage:

**Development skill** (helps developers work on your stage):
- How to install and run the code
- How to debug common errors (convergence failures, input format issues)
- How to interpret the output fields
- Key physics context for understanding what the code does

**Operational skill** (helps operators run the stage in the pipeline):
- How to build the Docker container
- How to run the test suite
- How to validate outputs against known-good results
- How to interpret W&B metrics

Place skills in your stage's docs directory (e.g., `docs/stage1-equilibrium/skills/`).

Fill in the "Claude Skills" section of your spec.

## Phase 2: Containerize & Test

After Phase 1 is complete for your stage, move to containerization and testing.

### Step 1: Write a Dockerfile

Your stage's container build context lives in `containers/stage{N}-{name}/`. See `docs/architecture.md` for the full container architecture.

**Base image:** Each stage uses a standalone Dockerfile -- there are no shared base images.
- CPU stages: `FROM python:3-slim` (pin to a specific minor version tag, e.g., `python:3.12-slim`)
- GPU stages (Stage 4): `FROM nvidia/cuda:12.x-runtime` (install JAX[cuda] and all dependencies directly)

The `slim` variant is a minimal Debian image (~50 MB vs ~350 MB for the full `python:3` image), keeping containers small. If your stage needs system-level build tools (e.g., compilers for building C extensions from source), use the full `python:3` image or add `apt-get install` steps.

Each Dockerfile installs the full scientific stack (NumPy, SciPy, h5py, netCDF4, xarray, wandb) alongside stage-specific upstream code. Pin JAX and CUDA versions according to `versions.yaml`.

**Dockerfile structure:**
```dockerfile
FROM python:3-slim

# Install scientific stack
RUN pip install --no-cache-dir \
    numpy scipy h5py netCDF4 xarray wandb

# Install JAX (pinned version from versions.yaml)
RUN pip install --no-cache-dir jax==0.4.35 jaxlib==0.4.35

# Install upstream code (pinned to exact commit from versions.yaml)
RUN pip install --no-cache-dir git+https://github.com/uwplasma/vmec_jax.git@<COMMIT_SHA>

# Copy entry-point scripts
COPY scripts/ /app/scripts/

# Non-root user
RUN useradd -m stellaforge
USER stellaforge

WORKDIR /app
ENTRYPOINT ["python", "/app/scripts/run.py"]
```

**Version pinning (critical):**
- Record the exact commit SHA of the upstream code that works in `versions.yaml` at the repo root
- Use that SHA in your `requirements.txt` and Dockerfile -- never reference `main` or a branch name
- For pip-installable packages, pin exact versions: `jax==0.4.35`

**Entry-point script:** Your `containers/stage{N}-{name}/scripts/run.py` (developed during Phase 1) becomes the container's ENTRYPOINT. This script reads input files from a mounted volume, calls the upstream library, and writes output files.

Follow the general Docker conventions from `CLAUDE.md`:
- Run as a non-root user
- Add `.dockerignore` for your stage
- Install only production dependencies
- Keep the image minimal

Document your container in the "Container Specification" section of your spec.

### Step 2: Verify Container I/O

- Build and run the container locally
- Verify it can read input files from a mounted volume
- Verify it writes output files to the expected location on the shared volume
- Verify the output directory follows the naming convention: `{run_dir}/stage{N}_{name}/`

### Step 3: Write Unit Tests

Test mathematical invariants specific to your stage. Examples:
- Stage 1: force-balance residual decreases monotonically during convergence
- Stage 2: Boozer transform preserves iota, |B| spectrum is complete
- Stage 3 (NEO): epsilon_eff is non-negative, bounded
- Stage 3 (SFINCS): transport matrix has expected symmetry properties
- Stage 3 (MONKES): D_ij matrix satisfies Onsager symmetry
- Stage 4: growth rates are real-valued, fluxes are non-negative in steady state
- Stage 5: profiles satisfy conservation (total particle/energy content)

Place tests in `tests/stage{N}-{name}/`.

### Step 4: Write Regression Tests

- Save known-good output files from your reference case
- Write tests that compare new outputs against these baselines
- Use explicit tolerances: `np.testing.assert_allclose(actual, expected, rtol=1e-6)`
- Document what tolerance is appropriate for each field

### Step 5: Write Integration Tests

Critical: verify that your stage's output is valid input for the NEXT stage.

- Stage 1 owner: verify `wout_*.nc` can be read by booz_xform_jax (Stage 2)
- Stage 2 owner: verify `boozmn_*.nc` can be read by NEO_JAX, sfincs_jax, and MONKES (Stage 3)
- Stage 3 owner: verify D_ij HDF5 database can be read by NEOPAX (Stage 5)
- Stage 4 owner: verify flux output can be consumed by NEOPAX turbulence-coupling (Stage 5)
- Stage 5 owner: verify end-to-end output (n, T, E_r, P_fus, Q) is produced correctly

### Step 6: Define Acceptance Criteria

Document your definition of "done" in the "Tests" section:
- All unit tests pass
- All regression tests pass within tolerances
- Integration tests with adjacent stages pass
- Container builds and runs successfully
- W&B tracking is functional
- All "OWNER COMPLETES" sections in your spec are filled in

## How to Document I/O

Use the standard table format in your spec:

```markdown
| Field | Type | Shape | Units | Source/Consumer |
|-------|------|-------|-------|----------------|
| `rmnc` | float64 | (ns, mnmax) | meters | Stage 2 input |
```

**Inspecting files:**
- NetCDF: `ncdump -h file.nc` or `python -c "import xarray; print(xarray.open_dataset('file.nc'))"`
- HDF5: `h5dump -H file.h5` or `python -c "import h5py; f=h5py.File('file.h5'); f.visit(print)"`
- NumPy: `python -c "import numpy as np; d=np.load('file.npz'); print(list(d.keys()))"`

**Always verify:** shapes, types (float32 vs float64), units, and coordinate conventions match between your output and the next stage's expected input.

## How to Create Claude Skills

Claude skills are Markdown files that Claude Code loads to gain specialized knowledge. They live in `.claude/skills/` or in your stage directory.

### Dev Skill Template

```markdown
---
name: stage{N}-{name}-dev
description: Development guide for Stage {N} ({Name}) of StellaForge
---

## Running {Code Name}

[How to install, configure, and run the primary JAX code]

## Debugging Common Issues

[Common errors and how to fix them]

## Interpreting Outputs

[What the key output fields mean, how to check if results are reasonable]

## Physics Context

[Brief physics background needed to work on this stage]
```

### Operational Skill Template

```markdown
---
name: stage{N}-{name}-ops
description: Operations guide for Stage {N} ({Name}) container in StellaForge
---

## Building the Container

[Docker build commands and verification]

## Running Tests

[How to run unit, regression, and integration tests]

## Validating Outputs

[How to check that outputs are correct, W&B metrics to monitor]
```

## How to Set Up W&B

### 1. Create the Project

```python
import wandb
wandb.init(project="stellaforge-stage{N}-{name}")
```

### 2. Log Stage Metrics

Each stage should log at minimum:
- **Runtime:** wall-clock time for the computation
- **Convergence metrics:** stage-specific (e.g., force residuals for Stage 1, growth rate convergence for Stage 4)
- **Key physics outputs:** the headline numbers from your stage (see your spec for which outputs are objectives)

### 3. Run Naming Convention

```
{code}_{config}_{timestamp}
```
Examples:
- `vmec_jax_landreman_qa_2026-04-01T12:00`
- `monkes_w7x_scan_2026-04-02T09:30`
- `spectrax_gk_hsx_linear_2026-04-03T14:15`

### 4. Dashboard

Create a W&B dashboard for your stage showing:
- Convergence over time/iterations
- Key physics outputs vs configuration
- Runtime performance

## Coding Conventions

The full coding standards are in `CLAUDE.md` at the repository root. Key points for stage work:

- Each stage is an **independent module** -- no cross-stage Python imports during Phase 1
- All inter-stage communication is through **files** (NetCDF/HDF5), not Python objects
- Follow PEP 8 with 120-char line width
- Add type hints to all function signatures
- Use NumPy-style docstrings with equation references
- Use `logging` module, not `print()`
- Seed all RNGs for reproducibility
- Save configuration alongside outputs

## Communication

- Stage owners coordinate with adjacent stages on I/O contracts
- Critical coordination points:
  - Stage 1 <-> Stage 2: wout_*.nc format
  - Stage 2 <-> Stage 3: boozmn_*.nc format
  - Stage 3 (MONKES) <-> Stage 5 (NEOPAX): D_ij database HDF5 format
  - Stage 4 <-> Stage 5: turbulent flux coupling (SPECTRAX-GK -> NEOPAX)
- All coordination should be documented in the relevant spec files
