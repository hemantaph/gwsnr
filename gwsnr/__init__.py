"""
GWSNR: Gravitational Wave Signal-to-Noise Ratio
"""

import warnings
warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")

__author__ = 'hemanta_ph <hemantaphurailatpam@gmail.com>'

__version__ = "0.2.3"

from .gwsnr import *
from .njit_functions import *
from .multiprocessing_routine import *
# from .pdet import *
from .utils import *


