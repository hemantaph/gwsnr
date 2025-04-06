"""
GWSNR: Gravitational Wave Signal-to-Noise Ratio
"""

import warnings
warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")

__author__ = 'hemanta_ph <hemantaphurailatpam@gmail.com>'

__version__ = "0.3.3"

# add __file__
import os
__file__ = os.path.abspath(__file__)

# import warnings
# warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")
# import lal
# lal.swig_redirect_standard_output_error(False)

import multiprocessing as mp
def set_multiprocessing_start_method():
    if os.name == 'posix':  # posix indicates the program is run on Unix/Linux/MacOS
      try:
         mp.set_start_method('fork', force=True)
      except RuntimeError:
         # The start method can only be set once and must be set before any process starts
         pass
    else:
      # For Windows and other operating systems, use 'spawn'
      try:
         mp.set_start_method('spawn', force=True)
      except RuntimeError:
         pass
set_multiprocessing_start_method()

from .gwsnr import *
from .njit_functions import *
from .multiprocessing_routine import *
# from .pdet import *
from .utils import *
from .ripple_class import *
from .jaxjit_functions import *
from .ann_model_generator import *

from . import ann


