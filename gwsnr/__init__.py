"""
GWSNR: Gravitational Wave Signal-to-Noise Ratio
"""

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


