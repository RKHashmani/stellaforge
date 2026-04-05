# MVP Pipeline: I/O Reference

```
 INDATA                   wout_*.nc                boozmn_*.nc
   |                          |                         |
   v                          v                         v
┌────────┐  NetCDF  ┌──────────────┐  NetCDF  ┌──────────────────────┐
│Stage 1 │ -------> │   Stage 2    │ -------> │      Stage 3         │
│vmec_jax│ wout_*.nc│booz_xform_jax│boozmn_*.nc│ SFINCS       monkes │
└────────┘          └──────────────┘          └───┬──────────────┬───┘
                          |                       |              |
                          |                       v              v
                          |              sfincsOutput.h5      D_ij.h5
                          |                  (HDF5)           (HDF5)
                          |                       |              |
                          |    ┌────────────┐     |              |
                          +--->│  Stage 4   │     |              |
                     wout_*.nc │ SPECTRAX-GK│     |              |
                               └─────┬──────┘     |              |
                                     |             |              |
                                flux (CSV)         |              |
                                     |             |              |
                                     v             v              v
                     ┌───────────────────────────────────────────────┐
                     │                  Stage 5                     │
                     │            Transport / Profiles              │
                     │  NEOPAX (wout + boozmn + D_ij + turb flux)   │
                     └──────────────────┬──────────────────────────┘
                                        |
                                        v
                                   profiles.h5
                              n(r), T(r), E_r(r), P_fus, Q
```

## MVP Test Data

Input configs and expected output files live in `mvp/`:

```
mvp/
├── stage1-equilibrium/vmec_jax/          input/ + expected_output/
├── stage2-boozer/booz_xform_jax/         input/ + expected_output/
├── stage3-neoclassical/sfincs_jax/       input/ + expected_output/
├── stage3-neoclassical/monkes/           input/ + expected_output/
├── stage4-turbulence/spectrax_gk/        input/ + expected_output/
└── stage5-transport/neopax/              input/ + expected_output/
```

Each `input/` directory contains config files for that code. Each `expected_output/` directory holds the reference output produced by running the code. Stage owners populate both as they validate their stage.

---

**Note:** All paths are relative to the repository root.

## Stage 1 -- Equilibrium

**Code:** vmec_jax

| Direction                     | Format                                    | Location                                                                        |
| ----------------------------- | ----------------------------------------- | ------------------------------------------------------------------------------- |
| **In**                        | Fortran-style Text                        | `mvp/stage1-equilibrium/vmec_jax/input/input.HSX_QHS_vacuum_ns201`              |
| **Out**                       | NetCDF `wout_*.nc` (similar to hdf5 file) | `mvp/stage1-equilibrium/vmec_jax/expected_output/wout_HSX_QHS_vacuum_ns201.nc`  |
| **Additional Out** (optional) | Text (terminal output)                    | `mvp/stage1-equilibrium/vmec_jax/expected_output/optional_terminal_output.vmec` |

> [!NOTE]
> `HSX_QHS_vacuum_ns201` is an example name. This can be changed. As can the entirety of the name `optional_terminal_output.vmec`.

### How to Install

From inside the `mvp/` directory

```
pixi install --environment stage-1-vmec
```

### How to Run

```
pixi run stage-1-vmec
```

---

## Stage 2 -- Boozer Transform

**Code:** booz_xform_jax

| Direction | Format               | Location                                                                          |
| --------- | -------------------- | --------------------------------------------------------------------------------- |
| **In**    | NetCDF `wout_*.nc`   | `mvp/stage1-equilibrium/vmec_jax/expected_output/wout_HSX_QHS_vacuum_ns201.nc`    |
| **Out**   | NetCDF `boozmn_*.nc` | `mvp/stage2-boozer/booz_xform_jax/expected_output/boozmn_HSX_QHS_vacuum_ns201.nc` |

> [!NOTE]
> The input comes from Stage 1 output.

### How to Install

```
pixi install --environment stage-2-booz
```

### How to Run

```
pixi run stage-2-booz
```

which is morally similar to

```python
import booz_xform_jax as bx
b=bx.Booz_xform()
b.read_wout("wout_HSX_QHS_vacuum_ns201.nc")
b.run()
b.write_boozmn("boozmn_HSX_QHS_vacuum_ns201.nc")
```

---

## Stage 3 -- Neoclassical (two parallel sub-stages)

**Code:** sfincs_jax

| Direction | Format                       | Location                                                                       |
| --------- | ---------------------------- | ------------------------------------------------------------------------------ |
| **In**    | NetCDF `wout_*.nc`           | `mvp/stage1-equilibrium/vmec_jax/expected_output/wout_HSX_QHS_vacuum_ns201.nc` |
| **In**    | Fortran-style Text `input.*` | `mvp/stage3-neoclassical/sfincs_jax/input/input.HSX_QHS_vacuum_ns201`          |
| **Out**   | HDF5 `sfincsOutput.h5`       | `mvp/stage3-neoclassical/sfincs_jax/expected_output/sfincsOutput.h5`           |

> [!NOTE]
> The input also comes from Stage 1 output. The 2nd input file has a variable `equilibriumFile` that must point to the (relative or absolute) location of the Stage 1's NetCDF output file. Additionally, the terminal output can be printed.

#### How to Install

```
pixi install --environment stage-3-sfincs
```

#### How to Run

**After** manually editing the value of `equilibriumFile` in `mvp/stage3-neoclassical/sfincs_jax/input/input.HSX_QHS_vacuum_ns201`, run

```
pixi run stage-3-sfincs
```


**code:** Monkes

| Direction | Format               | Location                                                                                         |
| --------- | -------------------- | ------------------------------------------------------------------------------------------------ |
| **In**    | NetCDF `wout_*.nc`   | `mvp/stage1-equilibrium/vmec_jax/expected_output/wout_HSX_QHS_vacuum_ns201.nc`                   |
| **In**    | NetCDF `boozmn_*.nc` | `mvp/stage2-boozer/booz_xform_jax/expected_output/boozmn_HSX_QHS_vacuum_ns201.nc`                |
| **Out**   | HDF5 `D_ij.h5`       | `mvp/stage3-neoclassical/monkes/expected_output/Monoenergetic_database_VMEC_s_coordinate_HSX.h5` |

> [!NOTE]
> The inputs come from both Stage 1 and Stage 2.

#### How to Install
```bash
git clone https://github.com/eduardolneto/monkes.git
cd monkes
pip install .
```

#### How to Run

Monkes is a little more involved. See `mvp/stage3-neoclassical/monkes/Test_Monoenergetic_database_VMEC_s_coordinate_HSX.py`

We basically call it inside a python loop to use Monkes to generate a database across different radial positions, electric fields, and collisionality, but it uses the same 2 input files for the entire loop.

---

## Stage 4 -- Turbulence

**Code:** SPECTRAX-GK

| Direction | Format             | Location                                                                           |
| --------- | ------------------ | ---------------------------------------------------------------------------------- |
| **In**    | NetCDF `wout_*.nc` | `mvp/stage1-equilibrium/vmec_jax/expected_output/wout_HSX_QHS_vacuum_ns201.nc`     |
| **In**    | TOML config        | `mvp/stage4-turbulence/spectrax_gk/input/runtime_hsx_nonlinear_vmec_geometry.toml` |
| **Out**   | Summary `JSON`     | `mvp/stage4-turbulence/spectrax_gk/expected_output/hsx_run.summary.json`           |
| **Out**   | `csv`              | `mvp/stage4-turbulence/spectrax_gk/expected_output/hsx_run.diagnostics.csv`        |

> [!NOTE]
> The input also comes from Stage 1 output. The 2nd input file has a variable `vmec_file` that must point to the (relative or absolute) location of the Stage 1's NetCDF output file.

### How to Install

```
pixi install --environment stage-4-spectrax
```

### How to Run

```
pixi run stage-4-spectrax
```

which executes something morally equivalent to

```
spectrax-gk run --config runtime_hsx_nonlinear_vmec_geometry.toml --out tools_out/hsx_run
```

---

## Stage 5 -- Transport Solver

**Code:** NEOPAX

| Direction                                 | Format               | Location                                                                                         |
| ----------------------------------------- | -------------------- | ------------------------------------------------------------------------------------------------ |
| **In**                                    | NetCDF `wout_*.nc`   | `mvp/stage1-equilibrium/vmec_jax/expected_output/wout_HSX_QHS_vacuum_ns201.nc`                   |
| **In**                                    | NetCDF `boozmn_*.nc` | `mvp/stage2-boozer/booz_xform_jax/expected_output/boozmn_HSX_QHS_vacuum_ns201.nc`                |
| **In** (Only needed if `monkes` is used.) | HDF5 `D_ij.h5`       | `mvp/stage3-neoclassical/monkes/expected_output/Monoenergetic_database_VMEC_s_coordinate_HSX.h5` |
| **Out**                                   | HDF5 `profiles_*.h5` | `mvp/stage5-transport/neopax/expected_output/NEOPAX_output.h5`                                   |
> [!NOTE]
> The inputs come from Stage 1, Stage 2, and Stage 3.

### How to Install

```
pixi install --environment stage-5-neopax
```

### How to Run

This is run inside a script. See `mvp/stage5-transport/neopax/run_NEOPAX.py` as a reference.

> [!NOTE]
> `NEOPAX`, being the final stage, has additional complexities and variabilities in the workflow.
>
> For example, if `monkes` is used, then the script using `NEOPAX` won't constantly loop since `monkes` provides a database and does the loop before reaching Stage 5.
>
> If `sfincs_jax` is used, then ideally, the script using NEOPAX runs a loop to optimize over different fluxes. While ideal, this is most computationally expensive.

---
