# CLAUDE.md

Project-level instructions for AI assistants working in the StellaForge repository.
This is the single source of truth for all coding standards -- team members may not have a global CLAUDE.md.

---

## Part 1: StellaForge Project

### Project Description

StellaForge is an end-to-end stellarator design pipeline connecting equilibrium, Boozer transform, neoclassical transport, turbulence, and profile evolution in containerized, orchestrated stages. The companion `stellarator_workflow/` submodule contains TeX manuscripts defining the physics equations and I/O contracts. See `docs/architecture.md` for the full pipeline design.

### Repository Structure

```
StellaForge/
├── CLAUDE.md                    # This file
├── versions.yaml                # Pinned upstream commits, JAX, CUDA versions
├── Snakefile                    # Pipeline DAG (Phase 3)
├── config.yaml                  # Default pipeline configuration
├── docs/                        # Project documentation
│   ├── architecture.md          # Master pipeline design
│   ├── contributor-guide.md     # Stage owner playbook
│   ├── workflow-integration.md  # Snakemake engineer spec
│   └── stage{N}-{name}/        # Per-stage specifications
│       └── spec.md
├── containers/                  # Dockerfiles and entry-point scripts
│   └── stage{N}-{name}/        # Per-stage container build context
│       ├── Dockerfile           # Standalone (FROM python:3-slim or nvidia/cuda:12.x)
│       ├── .dockerignore
│       ├── requirements.txt     # Upstream package pins (commit SHAs for source builds)
│       └── scripts/             # Entry-point and wrapper scripts
│           └── run.py           # Container ENTRYPOINT (also runnable locally)
├── input/                       # Reference input data (boundary coefficients, profiles)
├── tests/                       # Test suite (mirrors stage structure)
├── stellarator_workflow/        # Read-only TeX reference (git submodule)
└── runs/                        # Pipeline output (gitignored)
```

### The 5 Pipeline Stages

| Stage | Name | Primary JAX Code | Alternatives | Spec |
|-------|------|-----------------|--------------|------|
| 1 | Equilibrium | vmec_jax, DESC | VMEC++ | `docs/stage1-equilibrium/spec.md` |
| 2 | Boozer Transform | booz_xform_jax | BOOZ_XFORM | `docs/stage2-boozer/spec.md` |
| 3 | Neoclassical | NEO_JAX, sfincs_jax, MONKES | NEO, SFINCS | `docs/stage3-neoclassical/spec.md` |
| 4 | Turbulence | SPECTRAX-GK | GX, GENE | `docs/stage4-turbulence/spec.md` |
| 5 | Transport | NEOPAX | Trinity3D | `docs/stage5-transport/spec.md` |

Forward-pass chain: vmec_jax -> booz_xform_jax -> (NEO_JAX + sfincs_jax + MONKES) -> SPECTRAX-GK -> NEOPAX

### Naming Conventions

- Stage directories: `stage{N}-{name}` (e.g., `stage1-equilibrium`)
- Container images: `stellaforge/stage{N}-{name}:{version}` (on Docker Hub)
- W&B projects: `stellaforge-stage{N}-{name}`
- Output directories: `{run_dir}/stage{N}_{name}/`
- Test files: mirror source structure in `tests/`

### Inter-Stage Contracts

- All communication is file-based (NetCDF/HDF5 on shared volumes).
- Each stage reads from its predecessor's output directory.
- No wrapper protocol -- Snakemake wires stages via file dependencies.
- Stage spec docs are the authoritative source for I/O field definitions.

### Working with This Codebase

- Read `docs/architecture.md` first for the big picture.
- Each stage has its own spec in `docs/stageN-*/spec.md`.
- The `stellarator_workflow/` submodule is read-only reference material.
- When modifying a stage, consult its spec doc for I/O contracts.

### Phase-Specific Rules

- **Phase 1 (Document & Run):** Focus on documentation and running existing code. Do not restructure upstream code.
- **Phase 2 (Containerize & Test):** Container changes must pass integration tests before merge.
- **Phase 3 (Integrate):** Snakemake rules must support config-driven implementation selection.

---

## Part 2: Workflow Standards

### Planning

- Always use plan mode before changes that touch critical or complex logic (e.g., core algorithms, data pipelines, model architectures, mathematical computations).
- When something goes sideways during implementation, stop and re-plan -- don't keep pushing down a broken path.
- During planning, ask as many clarifying questions as needed to fully understand the relevant parts of the codebase, the current system, and how the proposed change fits in. Do not proceed with a plan until ambiguities are resolved.

### Review & Validation

- Act as a critical reviewer: challenge the user's reasoning, flag edge cases, and identify potential issues before implementation begins.
- After implementation, prove the changes work -- diff behavior between the working branch and `main`, run tests, and demonstrate correctness rather than just asserting it.
- Do not create a PR or propose merging until you can show evidence that the changes are correct and complete.

### Testing

- Every new feature or behavior change must include corresponding tests that verify it works as expected. Write tests alongside (or before) the implementation, not as an afterthought.
- Run the project's test suite after making changes to verify nothing is broken.

---

## Part 3: Scientific Programming Standards

### Reproducibility

- Always seed all RNG sources together so results are deterministic (e.g., `random.seed()`, `np.random.seed()`, `jax.random.PRNGKey()`, and any framework-specific seeds).
- Disable non-deterministic backend behavior for reproducible runs; only enable stochastic optimizations when explicitly trading reproducibility for speed.
- Save full configuration alongside every output so any result can be reproduced (e.g., save args/config to JSON next to output files).
- Default random seed to a deterministic value so runs are reproducible without explicit `--seed`.
- Never use truly random seeds unless explicitly requested by the user.

### Code Quality & Style

- Use the project's configured linter/formatter (e.g., ruff, black). Follow PEP 8 with 120-char line width for Python.
- Add type hints to all new/modified function signatures (parameters + return type).
- Use descriptive names; single-letter math variables (A, Sigma, z) are acceptable when they match paper notation -- add a comment referencing the equation/section.
- Keep functions under 60 lines and ~5 parameters. Extract helpers for longer functions.
- Eliminate code duplication: if the same logic appears twice, factor it into a shared function.
- Never control behavior by commenting/uncommenting code -- use flags or config parameters.
- Prefer f-strings for string formatting in Python.

### Testing

- Write unit tests for all mathematical/numerical functions using the project's test framework (e.g., pytest).
- Test mathematical invariants (e.g., non-negativity constraints, matrix properties like positive semi-definiteness, correct output dimensions).
- For numerical functions, test against known analytical solutions where available.
- Use regression tests: save known-good outputs and compare with explicit tolerances (e.g., `np.testing.assert_allclose`).
- Place tests in a top-level `tests/` directory mirroring the source structure.

### Documentation

- Add docstrings to all new/modified public functions using NumPy-style format (Parameters, Returns, Notes).
- For new docstrings, do not reformat existing docstrings in a different style unless modifying the function.
- Include mathematical context in docstrings: reference paper equations/sections where applicable.
- Every module should have a module-level docstring explaining its role in the system.
- Config/argument files should be self-documenting via comments.

### Logging & Error Handling

- Use Python's `logging` module with `logger = logging.getLogger(__name__)` for all output in production code paths.
- `print()` is acceptable only in debug blocks and CLI entry points (`if __name__ == "__main__":`).
- Use typed exceptions with descriptive messages for input validation (e.g., `raise ValueError(...)`, `raise TypeError(...)`). Never bare `assert` for user-facing checks.
- Never silently swallow exceptions. Always log or re-raise caught exceptions.
- Validate data shapes/dimensions at function boundaries when they are non-obvious.

### Numerical Best Practices

- Use library-provided linear algebra functions over manual implementations (e.g., `jax.numpy.linalg`, `numpy.linalg`, `scipy.linalg`).
- Validate mathematical properties of inputs before use (e.g., positive-definiteness of covariance matrices via Cholesky).
- Propagate device and dtype from input data -- never hardcode device or precision.
- Use higher precision for computations where numerical accuracy matters (e.g., float64 for information-theoretic or covariance calculations).
- Guard against numerical edge cases: division by zero, log(0), overflow in exp(), and similar.

### Project Organization

- All packages must have proper `__init__.py` files.
- Use proper package imports; avoid path-manipulation hacks (e.g., `sys.path.append()`).
- Scripts should be runnable from the project root, not require `cd` into a subdirectory.
- Keep distinct concerns (data processing, model architecture, training, visualization) as independent modules.

### Version Control

- Make small, focused commits with descriptive messages.
- Save argument files and config alongside results for traceability.
- Never commit large binary files (model checkpoints, datasets) or secrets. Use `.gitignore` appropriately.

### Data Management

- Save raw data separately from processed results.
- Document output formats: array shapes, value ranges, and units in docstrings or project docs.
- Use structured, self-describing formats with descriptive field names (e.g., `.npz` with named arrays, HDF5 with labeled datasets).

### Configuration Management

- All runtime behavior must be controllable via CLI arguments or config files -- never require editing source code.
- Save complete configuration alongside outputs for every run.

---

## Part 4: Software Engineering Standards

### Dependency Management

- Pin all dependencies with version bounds (e.g., `>=1.0,<2`); avoid unpinned or `*` versions.
- Commit lockfiles for reproducible installs (e.g., `pixi.lock`, `uv.lock`, `poetry.lock`).
- Keep dependency specs consistent across all environment/packaging files when adding or updating packages.
- Separate dev-only dependencies (testing, linting tools) from core dependencies using optional dependency groups.

### Security

- Never hardcode secrets, API keys, or credentials -- use environment variables or config files.
- Never use `eval()`, `exec()`, or unsafe deserialization on untrusted data.
- Use safe loading modes when available (e.g., `torch.load(..., weights_only=True)`); document the reason when unsafe loading is necessary.
- Validate and sanitize all external inputs (CLI args, file paths, JSON) before use.
- Prefer safe subprocess invocation (e.g., `subprocess.run()` with list args); never use `shell=True` with user-provided input.

### CI/CD & Automation

- Configure CI to run linting, type checking, and tests on pull requests.
- Use pre-commit hooks for formatting and linting before commit.
- Automate repetitive workflows via scripts, not manual multi-step commands.

### Docker & Containerization

- Pin base image versions to specific tags, not `latest`.
- Run containers as a non-root user for security.
- Add `.dockerignore` to exclude unnecessary files (e.g., `data/`, `output_dir/`, `.git/`, `__pycache__/`).
- Keep Dockerfiles minimal -- install only production dependencies.

**StellaForge container architecture:**
- Each stage has a fully independent Dockerfile in `containers/stage{N}-{name}/`. There are no shared base images -- each stage manages its own complete dependency stack.
- CPU stages start `FROM python:3-slim` (pin to a specific minor version tag in the actual Dockerfile). GPU stages (Stage 4: SPECTRAX-GK, and any future GPU-accelerated stages) start `FROM nvidia/cuda:12.x-runtime`.
- Prebuilt images are published to Docker Hub under the `stellaforge/` organization.
- Each stage's `containers/stage{N}-{name}/scripts/run.py` is the container entry point. These scripts bridge between the pipeline's file-based I/O and the upstream code's Python API.

**Version pinning (critical for reproducibility):**
- All upstream dependencies are pinned in `versions.yaml` at the repo root.
- For pip-installable packages: pin exact versions (e.g., `jax==0.4.35`).
- For source-built packages: pin to exact git commit SHAs, not branch names. A `git clone` + `git checkout {sha}` is reproducible; `git clone` of `main` is not.
- Each stage's `containers/stage{N}-{name}/requirements.txt` must reference the pinned versions from `versions.yaml`.
- Each stage's Dockerfile pins its own Python version independently. The Python version is not hardcoded across the project; each stage uses whatever version its upstream dependencies require.

### Performance & Memory Management

- Disable gradient computation for all inference and evaluation code paths (e.g., `jax.jit` with no grad, `torch.no_grad()`).
- Use mixed precision when memory is a concern.
- Release memory between intensive phases.
- Profile before optimizing -- use profiling tools to identify actual bottlenecks before making performance changes.
