============
Installation
============

.. note::
    
    For package development and contribution refer here (:ref:`development`).

.. tabs::
        
     .. code-tab:: console pip

        pip install gwsnr


This will also install the dependencies needed by the lastest ``gwsnr`` version.  

.. _development:
gwsnr for development
======================

To install ``gwsnr`` for development purposes use `github.gwsnr <https://github.com/hemantaph/gwsnr/>`_. Use conda environment to avoid dependency error. 

    
.. tabs::

     .. code-tab:: console with new conda env

        git clone https://github.com/hemantaph/gwsnr.git
        cd gwsnr
        conda env create -f gwsnr.yml
        conda activate gwsnr
        pip install -e .
        
     .. code-tab:: with existing conda env
     
        git clone https://github.com/hemantaph/gwsnr.git
        cd gwsnr
        conda env update --file gwsnr.yml
        pip install -e .
    
.. _dependencies:
Installation of numba with conda
=======================

.. code-block:: console

    conda install -c conda-forge numba