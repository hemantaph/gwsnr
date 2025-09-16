"""
GWSNR: Gravitational Wave Signal-to-Noise Ratio
"""
# In your package __init__.py

import os
import multiprocessing as mp
import warnings

def set_multiprocessing_start_method():
    """
    Set the multiprocessing start method based on OS and environment variables.
    By default, sets 'fork' on POSIX systems unless overridden by GWSNR_USE_SPAWN=True.
    """
    # Only set if not already set
    method = None

    # POSIX = Linux, Mac
    if os.name == 'posix':
        # User override: GWSNR_USE_SPAWN=True
        use_spawn = os.environ.get("GWSNR_USE_SPAWN", "False").lower() == "true"
        if use_spawn:
            method = "spawn"
            print("GWSNR: Setting multiprocessing start method to 'spawn' per environment variable.")
        else:
            method = "fork"
            # print(
            #     "GWSNR: Setting multiprocessing start method to 'fork'.\n"
            #     "If you need to use the 'spawn' method (in case error or warning due to other library dependencies),\n"
            #     "set the environment variable GWSNR_USE_SPAWN=True *before* running your script."
            #     "\n"
            #     "Command line (single line):\n"
            #     "    GWSNR_USE_SPAWN=True python yourscript.py\n"
            #     "In a Python script (before importing GWSNR):\n"
            #     "    import os\n"
            #     "    os.environ['GWSNR_USE_SPAWN'] = 'True'\n"
            #     "    import gwsnr\n"
            # )
        try:
            mp.set_start_method(method, force=True)
        except RuntimeError:
            # Already set (possibly by another library, or called too late)
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
    else:
        # For Windows, default is already 'spawn', nothing to do.
        pass

# Call the function on package import
set_multiprocessing_start_method()


import warnings
import logging
from . import core
from .core import GWSNR
from ._version import __version__

# Package metadata
__all__ = ['GWSNR', 'core']
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


