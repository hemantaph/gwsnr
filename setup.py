#!/usr/bin/env python
from setuptools import setup, find_packages
import sys
from pathlib import Path

# read the contents of your README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# check that python version is 3.10 or above
python_version = sys.version_info
if python_version < (3, 10):
    sys.exit("Python < 3.10 is not supported, aborting setup")

# Read requirements from requirements.txt
def parse_requirements(filename):
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return [line.strip() for line in lines if line.strip() and not line.startswith('#')]

requirements = parse_requirements("requirements.txt")

setup(
    name='gwsnr',
    version='0.3.3',
    description='gwsnr: A Python Package for Efficient SNR Calculation of Gravitational Waves',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Hemantakumar Phurailatpam',
    license="MIT",
    author_email='hemantaphurailatpam@gmail.com',
    url='https://github.com/hemantaph/gwsnr',
    packages=find_packages(),
    package_data={
        'gwsnr': ['ann/data/*.json', 'ann/data/*.pkl', 'ann/data/*.h5'],
    },
    install_requires=requirements,
)

# #!/usr/bin/env python
# from setuptools import setup, find_packages
# import sys
# # read the contents of your README file
# from pathlib import Path
# this_directory = Path(__file__).parent
# long_description = (this_directory / "README.md").read_text()

# # check that python version is 3.7 or above
# python_version = sys.version_info
# if python_version < (3, 10):
#     sys.exit("Python < 3.10 is not supported, aborting setup")

# #!/usr/bin/env python
# from setuptools import setup, find_packages
# setup(name='gwsnr',
#       version='0.3.3',
#       description='gwsnr: A Python Package for Efficient SNR Calculation of Gravitational Waves',
#       author='Hemantakumar Phurailatpam',
#       license="MIT",
#       author_email='hemantaphurailatpam@gmail.com',
#       url='https://github.com/hemantaph/gwsnr',
#       packages=find_packages(),
#       package_data={
#         'gwsnr': ['ann/*.json', 'ann/*.pkl', 'ann/*.h5'],
#       },
#       install_requires=[
#         "setuptools>=65.5.0",
#         "matplotlib>=3.4.2",
#         "numpy>=1.18",
#         "bilby>=1.0.2",
#         "pycbc>=2.0.4",
#         "scipy<1.14.0",
#         "tqdm>=4.64.0",
#         "gwpy>=2.1.5",
#         "h5py>=3.11.0,<3.12.0",
#         "numba>=0.57.1,<0.58.0",
#         "numba>=0.57.1",
#         "ml-dtypes==0.3.1", # new
#         "tensorflow>=2.17.0,<2.18.0",
#         "scikit-learn==1.5.0",
#         "numexpr>=2.8.4",
#         "ripplegw",
#       ]
#      )
