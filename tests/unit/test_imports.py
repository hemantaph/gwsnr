import subprocess
import sys


def _run_python(code):
    return subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )


def test_import_gwsnr_is_lightweight():
    code = """
import sys
import gwsnr

assert "GWSNR" not in gwsnr.__dict__
assert "core" not in gwsnr.__dict__
for module in ("multiprocessing", "numpy", "numba", "scipy"):
    assert module not in sys.modules, f"{module} was imported by import gwsnr"
assert gwsnr.__version__
"""
    _run_python(code)


def test_core_import_is_lightweight():
    code = """
import sys
from gwsnr import core

assert core.__name__ == "gwsnr.core"
assert "GWSNR" not in core.__dict__
for module in ("numpy", "numba", "scipy"):
    assert module not in sys.modules, f"{module} was imported by import gwsnr.core"
"""
    _run_python(code)


def test_lazy_gwsnr_exports_resolve():
    code = """
from gwsnr import GWSNR
from gwsnr.core import GWSNR as CoreGWSNR

assert GWSNR is CoreGWSNR
assert GWSNR.__name__ == "GWSNR"
"""
    _run_python(code)
