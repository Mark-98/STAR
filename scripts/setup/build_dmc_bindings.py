"""Generate the retained dm_control fork's bindings with its original generator."""
import os
from pathlib import Path
import subprocess
import sys
import mujoco

ROOT = Path(__file__).resolve().parents[2]


def main():
    if mujoco.__version__ != '2.3.0':
        raise RuntimeError('The RL fork requires mujoco==2.3.0; use environments/rl.yaml.')
    base = ROOT / 'vendor/dm_control'
    headers = ['mjdata.h', 'mjmodel.h', 'mjrender.h', 'mjtnum.h', 'mjui.h', 'mjvisualize.h', 'mjxmacro.h', 'mujoco.h']
    output = base / 'dm_control/mujoco/wrapper/mjbindings'
    env = dict(os.environ, PYTHONPATH=str(base), PYTHONDONTWRITEBYTECODE='1')
    subprocess.run([sys.executable, str(base / 'dm_control/autowrap/autowrap.py'),
                    '--header_paths=' + ' '.join(str(Path(mujoco.HEADERS_DIR) / h) for h in headers),
                    '--output_dir=' + str(output)], cwd=base, env=env, check=True)
    print('Generated bindings inside vendor/dm_control for MuJoCo 2.3.0.')


if __name__ == '__main__':
    main()
