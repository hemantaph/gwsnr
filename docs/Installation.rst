============
Installation
============

.. note::

    ``gwsnr`` supports Python 3.10+ (but 3.11 recommended) and utilizes multi-core CPUs and NVIDIA GPUs (or Apple Silicon GPUs) when available. 
    
    For package development and contribution refer here (:ref:`development`).

.. tabs::

   .. code-tab:: bash pip (standard with Numba; CPU)

        pip install gwsnr

   .. code-tab:: bash pip (with JAX; CPU or Nvidia GPU)

      pip install gwsnr jax jaxlib
      pip install -U "jax[cuda12]" # optional, for Nvidia GPU support

   .. code-tab:: bash pip (with MLX; Apple Silicon)

      pip install gwsnr mlx

   .. code-tab:: bash pip (with ripple based JAX waveforms)

      pip install gwsnr jax jaxlib ripplegw

   .. code-tab:: bash pip (with tensorflow based ANN)

      pip install gwsnr scikit-learn tensorflow
      pip install --upgrade ml-dtypes # optional, for compatibility


This will also install the dependencies needed by the lastest ``gwsnr`` version.  

.. _development:
gwsnr for development
======================

To install ``gwsnr`` for development purposes use `github.gwsnr <https://github.com/hemantaph/gwsnr/>`_. Use conda environment to avoid dependency error. 

    
.. tabs::

     .. code-tab:: bash with new conda env

        git clone https://github.com/hemantaph/gwsnr.git
        cd gwsnr
        conda env create -f gwsnr.yml
        conda activate gwsnr
        pip install -e .
        
     .. code-tab:: bash with existing conda env
     
        git clone https://github.com/hemantaph/gwsnr.git
        cd gwsnr
        conda env update --file gwsnr.yml
        pip install -e .