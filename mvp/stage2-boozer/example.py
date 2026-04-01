from pathlib import Path

# the pathlib path of this file
file_path = Path(__file__).resolve()
input_file_dir = (
    file_path.parents[1] / "stage1-equilibrium" / "vmec_jax" / "expected_output"
)
output_file_dir = file_path.parent / "booz_xform_jax" / "expected_output"
output_file_dir.mkdir(exist_ok=True)

import booz_xform_jax as bx

b = bx.Booz_xform()
b.read_wout(input_file_dir / "wout_HSX_QHS_vacuum_ns201.nc")
b.run()
b.write_boozmn(output_file_dir / "boozmn_HSX_QHS_vacuum_ns201.nc")
