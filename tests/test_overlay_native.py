"""Native overlay lifecycle with simulated D3D, no game process."""
from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.mark.skipif(shutil.which('node') is None, reason='Node.js not installed')
def test_late_device_can_draw_paths_and_replace_its_present_hook():
    fixture=Path(__file__).with_name('native_overlay.cjs')
    result=subprocess.run(['node',str(fixture)],capture_output=True,text=True,timeout=15)
    assert result.returncode == 0, result.stdout+result.stderr
