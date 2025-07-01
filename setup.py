#!/usr/bin/env python
from setuptools import setup, find_packages
import sys
from pathlib import Path
import os

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

# Get version
version = {}
with open(os.path.join(this_directory, "gwsnr", "_version.py")) as fp:
    exec(fp.read(), version)

setup(
    name='gwsnr',
    version=version['__version__'],
    description='gwsnr: A Python Package for Efficient SNR Calculation of Gravitational Waves',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Hemantakumar Phurailatpam',
    license="MIT",
    author_email='hemantaphurailatpam@gmail.com',
    url='https://github.com/hemantaph/gwsnr',
    packages=find_packages(),
    include_package_data=True,
    package_data={
        'gwsnr': ['ann/ann_data/*', 'core/interpolator_pickle/*'],
        'gwsnr.ann': ['ann_data/*'],
        'gwsnr.core': ['interpolator_pickle/*'],
    },
    python_requires='>=3.10',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],
    install_requires=requirements,
)
