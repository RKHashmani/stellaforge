# MVP Pipeline: I/O Reference

```
 INDATA                   wout_*.nc                boozmn_*.nc
   |                          |                         |
   v                          v                         v
┌────────┐  NetCDF  ┌──────────────┐  NetCDF  ┌──────────────────────┐
│Stage 1 │ -------> │   Stage 2    │ -------> │      Stage 3         │
│vmec_jax│ wout_*.nc│booz_xform_jax│boozmn_*.nc│ SFINCS       MONKES │
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

Notes: `HSX_QHS_vacuum_ns201` is an example name. This can be changed. As can the entirety of the name `optional_terminal_output.vmec`.

### How to Install
```bash
git clone https://github.com/uwplasma/vmec_jax.git
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```
### How to Run

```bash
vmec_jax mvp/stage1-equilibrium/vmec_jax/input/input.HSX_QHS_vacuum_ns201
```

---

## Stage 2 -- Boozer Transform

**Code:** booz_xform_jax

| Direction | Format               | Location                                                                          |
| --------- | -------------------- | --------------------------------------------------------------------------------- |
| **In**    | NetCDF `wout_*.nc`   | `mvp/stage1-equilibrium/vmec_jax/expected_output/wout_HSX_QHS_vacuum_ns201.nc`    |
| **Out**   | NetCDF `boozmn_*.nc` | `mvp/stage2-boozer/booz_xform_jax/expected_output/boozmn_HSX_QHS_vacuum_ns201.nc` |

Notes: The input comes from Stage 1 output.

### How to Install
First, the dependencies:
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install "jax[cpu]" numpy scipy netCDF4 matplotlib
```
Then, the software itself:
```bash
git clone https://github.com/uwplasma/boozx.git
cd boozx
pip install -e .
```
### How to Run
Inside python:
```python
import booz_xform_jax as bx  
b=bx.Booz_xform()  
b.read_wout("wout_HSX_QHS_vacuum_ns201.nc")  
b.run()  
b.write_boozmn("boozmn_HSX_QHS_vacuum_ns201.nc")
```

---

## Stage 3 -- Neoclassical (three parallel sub-stages)

**Code:** sfincs_jax

| Direction | Format                       | Location                                                                       |
| --------- | ---------------------------- | ------------------------------------------------------------------------------ |
| **In**    | NetCDF `wout_*.nc`           | `mvp/stage1-equilibrium/vmec_jax/expected_output/wout_HSX_QHS_vacuum_ns201.nc` |
| **In**    | Fortran-style Text `input.*` | `mvp/stage3-neoclassical/sfincs_jax/input/input.HSX_QHS_vacuum_ns201`          |
| **Out**   | HDF5 `sfincsOutput.h5`       | `mvp/stage3-neoclassical/sfincs_jax/expected_output/sfincsOutput.h5`           |

Notes: The input also comes from Stage 1 output. The 2nd input file has a variable `equilibriumFile` that must point to the (relative or absolute) location of the Stage 1's NetCDF output file. Additionally, the terminal output can be printed.

#### How to Install
```bash
git clone https://github.com/uwplasma/sfincs_jax.git
cd sfincs_jax
pip install .
```
**Note:** the README advertises `pip install sfincs_jax` from PyPI, but this does not currently work. Use source install.
#### How to Run

```bash
sfincs_jax input.HSX_QHS_vacuum_ns201
```

---

