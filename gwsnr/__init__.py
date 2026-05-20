"""
GWSNR: Gravitational Wave Signal-to-Noise Ratio.

``import gwsnr`` only configures lightweight threading defaults; ``GWSNR`` and
the ``core`` subpackage are loaded on first access (see ``__getattr__``).
"""

import os
import warnings

# disable OpenMP warnings
os.environ["KMP_WARNINGS"] = "off"

# Set default thread counts before numba/MKL/OpenBLAS are imported.
# NUMBA_NUM_THREADS: controls Numba prange parallelism (default: all cores).
#   Dynamically adjusted to npool via numba.set_num_threads() at call sites,
#   and set to 1 inside multiprocessing workers to avoid CPU oversubscription.
# OMP/MKL/OPENBLAS: default to 1 so BLAS libraries don't compete with
#   Numba's own thread pool or multiprocessing workers.
os.environ.setdefault("NUMBA_NUM_THREADS", str(os.cpu_count()))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


def set_multiprocessing_start_method():
    """
    Set ``multiprocessing`` start method once per process when explicitly called.

    Default choices: ``spawn`` on macOS, ``fork`` on other POSIX systems. Windows
    is left unchanged (``spawn``).

    Environment overrides (POSIX): ``GWSNR_USE_SPAWN=True`` or
    ``GWSNR_USE_FORK=True`` (if both are set, ``spawn`` is used and a warning is
    issued).
    """
    import multiprocessing as mp

    method = None

    if os.name == "posix":
        use_spawn = os.environ.get("GWSNR_USE_SPAWN", "False").lower() == "true"
        use_fork = os.environ.get("GWSNR_USE_FORK", "False").lower() == "true"

        if use_spawn and use_fork:
            warnings.warn(
                "GWSNR: Both GWSNR_USE_SPAWN=True and GWSNR_USE_FORK=True are set. "
                "Using 'spawn'."
            )
            method = "spawn"
            print(
                "GWSNR: Setting multiprocessing start method to 'spawn' per environment variable."
            )
        elif use_spawn:
            method = "spawn"
            print(
                "GWSNR: Setting multiprocessing start method to 'spawn' per environment variable."
            )
        elif use_fork:
            method = "fork"
            print(
                "GWSNR: Setting multiprocessing start method to 'fork' per environment variable."
            )
        else:
            method = "spawn" if os.uname().sysname == "Darwin" else "fork"

        current = mp.get_start_method(allow_none=True)
        if current is None:
            try:
                mp.set_start_method(method)
            except RuntimeError:
                warnings.warn(
                    f"GWSNR: Could not set multiprocessing start method to '{method}'. "
                    "This is usually because the start method was already set by another library or your script, "
                    "or because it is being set too late.\n"
                    "If you need to control the multiprocessing start method, set it at the very beginning of your script.\n"
                    "To use the 'spawn' method instead, set the environment variable GWSNR_USE_SPAWN=True before running your script.\n"
                    "Command line (single line):\n"
                    "    GWSNR_USE_SPAWN=True python yourscript.py\n"
                    "In a Python script (before importing GWSNR):\n"
                    "    import os\n"
                    "    os.environ['GWSNR_USE_SPAWN'] = 'True'\n"
                    "    import gwsnr\n"
                )
        elif current != method:
            warnings.warn(
                f"GWSNR: Multiprocessing start method is already '{current}'. "
                f"Requested '{method}' was not applied."
            )
    else:
        # For Windows, default is already 'spawn', nothing to do.
        pass


import logging
from ._version import __version__

# Package metadata
__all__ = ["GWSNR", "core", "set_multiprocessing_start_method", "__version__"]
__author__ = 'Hemantakumar Phurailatpam <hemantaphurailatpam@gmail.com>'
__license__ = "MIT"
__email__ = "hemantaphurailatpam@gmail.com"
__maintainer__ = "Hemantakumar Phurailatpam"
__status__ = "Development"
__url__ = "https://github.com/hemantaph/gwsnr"
__description__ = "A Python package for calculating gravitational wave signal-to-noise ratios"
__version_info__ = tuple(map(int, __version__.split('.')))

warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")
logging.getLogger(__name__).addHandler(logging.NullHandler())


def __getattr__(name):
    """
    Resolve ``GWSNR`` or the ``core`` subpackage on first use.

    Parameters
    ----------
    name : str
        Attribute name on the ``gwsnr`` package.

    Returns
    -------
    type or module
        ``GWSNR`` from ``gwsnr.core`` or the ``gwsnr.core`` package.

    Raises
    ------
    AttributeError
        If ``name`` is not one of the names exported via this hook.
    """
    if name == "core":
        import importlib

        _core = importlib.import_module(f"{__name__}.core")
        globals()[name] = _core
        return _core
    if name == "GWSNR":
        from .core import GWSNR as _GWSNR

        globals()[name] = _GWSNR
        return _GWSNR
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
