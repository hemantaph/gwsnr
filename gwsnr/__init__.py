"""
GWSNR: Gravitational Wave Signal-to-Noise Ratio
"""

import warnings
warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")

__author__ = 'hemanta_ph <hemantaphurailatpam@gmail.com>'

__version__ = "0.3.2"

# add __file__
import os
__file__ = os.path.abspath(__file__)

from .gwsnr import *
from .njit_functions import *
from .multiprocessing_routine import *
# from .pdet import *
from .utils import *

from . import ann


