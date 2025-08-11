# Installation

> **Note:**
> For package development and contribution refer [here](#gwsnr-for-development).

## Installation via pip

### pip

```console
pip install gwsnr
```

### pip with GPU support

```console
pip install gwsnr
pip install -U "jax[cuda12]"
```

This will also install the dependencies needed by the latest `gwsnr` version.

* `gwsnr` includes [JAX](https://jax.readthedocs.io/en/latest/) functionalities.
  For faster SNR interpolation computation using Nvidia GPU, install `JAX` with GPU support.

## gwsnr for development

To install `gwsnr` for development purposes, use [github.gwsnr](https://github.com/hemantaph/gwsnr/).
It is recommended to use a conda environment to avoid dependency errors.

### With a new conda environment

```console
git clone https://github.com/hemantaph/gwsnr.git
cd gwsnr
conda env create -f gwsnr.yml
conda activate gwsnr
pip install -e .
```

</details>

### With an existing conda environment

```console
git clone https://github.com/hemantaph/gwsnr.git
cd gwsnr
conda env update --file gwsnr.yml
pip install -e .
```

### Installation of numba with conda

```console
conda install -c conda-forge numba
```
