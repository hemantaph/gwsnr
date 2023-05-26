#!/usr/bin/env python
from setuptools import setup, find_packages
setup(name='gwsnr',
      version='0.1.0',
      description='Fast SNR interpolator',
      author='Hemantakumar, Otto',
      license="MIT",
      author_email='hemantaphurailatpam@gmail.com',
      url='https://github.com/hemantaph/gwsnr',
      packages=find_packages(),
      install_requires=[
        "setuptools>=61.1.0",
        "bilby>=1.0.2",
        "pycbc>=2.0.4",
        "scipy>=1.9.0",
        "tqdm>=4.64.0",
        "gwpy>=2.1.5",
      ]
     )
