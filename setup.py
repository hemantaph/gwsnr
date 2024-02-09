#!/usr/bin/env python
from setuptools import setup, find_packages
import sys
# read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# check that python version is 3.7 or above
python_version = sys.version_info
if python_version < (3, 10):
    sys.exit("Python < 3.10 is not supported, aborting setup")

#!/usr/bin/env python
from setuptools import setup, find_packages
setup(name='gwsnr',
      version='0.2.2',
      description='Fast SNR interpolator',
      author='Hemantakumar Phurailatpam',
      license="MIT",
      author_email='hemantaphurailatpam@gmail.com',
      url='https://github.com/hemantaph/gwsnr',
      packages=find_packages(),
      install_requires=[
        "setuptools>=61.1.0",
        "numpy>=1.18",
        "bilby>=1.0.2",
        "pycbc>=2.0.4",
        "scipy>=1.9.0",
        "tqdm>=4.64.0",
        "gwpy>=2.1.5",
      ]
     )
