Welcome to gwsnr's documentation!
===============================

.. image:: _static/logo.svg
   :align: center
   :width: 40%
   :alt: gwsnr logo


gwsnr
-----------------

``gwsnr``: :red_first:`Gravitational` :red_first:`Wave` :red_first:`Signal`-to-:red_first:`Noise` :red_first:`Ratio` Computation Package

The ``gwsnr`` package is designed to facilitate efficient and accurate SNR computations in gravitational wave research. It implements advanced techniques for enhancing calculation speed and precision, making it a valuable tool for researchers in this field. Description available in :doc:`Summary` section.

| The code is available at `github.gwsnr <https://github.com/hemantaph/gwsnr>`_.
| For reaching out to the developer, please raise issue in `github.gwsnr.issue <https://github.com/hemantaph/gwsnr/issues>`_.

``gwsnr`` is integrated with `ler <https://ler.readthedocs.io/en/latest/>`_ package for simulating detectable gravitational wave signals.

| ``gwsnr`` main developer: `Hemanta Ph. <https://www.hemantaph.com>`_
| ``gwsnr`` developer and analyst: `Hemanta Ph. <https://www.hemantaph.com>`_, `Otto Akseli Hannuksela <https://www.oahannuksela.com/>`_.

For citation, please use the following BibTeX entry:

.. code-block:: bibtex

   @misc{gwsnr:2025,
         title={gwsnr: A python package for efficient signal-to-noise calculation of gravitational-waves}, 
         author={Hemantakumar Phurailatpam and Otto Akseli Hannuksela},
         year={2025},
         eprint={2412.09888},
         archivePrefix={arXiv},
         primaryClass={astro-ph.IM},
         url={https://arxiv.org/abs/2412.09888}, 
   }

Quick Start Guide
-----------------

Detailed Installation guide and usage examples are available in :doc:`Installation` and :doc:`Examples` sections respectively. Here is a quick start guide to use ``gwsnr`` package.

Installation (bash command)

.. code-block:: bash

   pip install gwsnr

SNR computation with `gwsnr` (Python code)

.. code-block:: python

   from gwsnr import GWSNR
   gwsnr = GWSNR()
   snrs = gwsnr.snr(mass_1=30, mass_2=30, distance=1000, psi=0.0, phase=0.0, geocent_time=1246527224.169434, ra=0.0, dec=0.0)
   print(f"SNR value: {snrs}")


Glossary
-----------------

.. glossary::

 
   Gravitational waves

      .. image:: _static/gw.gif
         :align: center
         :width: 480px
         :alt: gif

      *Animation showing the propagation of gravitational waves from inspiraling binary black holes.* 
      `Source <https://community.wolfram.com/groups/-/m/t/790989>`_ : Jeffrey Bryant, Wolfram | Alpha, LLC.

      Ripples in the fabric of space-time caused by some of the most violent and energetic processes in the Universe, such as merger of compact binaries (e.g., black holes and neutron stars) and supernovae explosions. Albert Einstein predicted the existence of gravitational waves in 1916 in his general theory of relativity, but it took a century to detect them directly. The first detection was made in 2015 by the LIGO and Virgo collaborations, which won the 2017 Nobel Prize in Physics. Gravitational waves are invisible, yet incredibly fast; they travel at the speed of light, squeezing and stretching anything in their path.


   Detection of gravitational waves

      .. image:: _static/matched_filtering.gif
         :align: center
         :width: 600px
         :alt: gif

      *Animation showing the matched filtering technique used in gravitational wave detection. Top: Signal-to-noise ratio (SNR, blue) as a function of GPS time, with the SNR threshold (orange dashed line) indicating the minimum required for confident detection. The sharp peak crossing the threshold marks a detectable GW event. Bottom: Strain versus GPS time, showing the gravitational wave signal (pink) embedded in background noise (black).*
      `Source <https://www.youtube.com/watch?v=bBBDR5jf9oU>`_ : Alex Nitz | youtube.

      Gravitational waves are detected using laser interferometers, which measure the changes in the distances of arm lengths caused by the passing waves. Four major detectors are currently operational: LIGO Livingston (in the US), LIGO Hanford (in the US), Virgo (in Italy), and KAGRA (in Japan). The amplitude of these waves is extremely small, typically on the order of 10^-21 m, and advanced techniques are required to detect and  extract the signals from the noise background. Match filtering is a common technique used to identify the presence of gravitational wave signals in the data by correlating the data with theoretical templates of expected signals.


   Signal-to-noise ratio (SNR)

      The signal-to-noise ratio (SNR) is a key metric in assessing the strength of the detected gravitational wave signal against the noise. Higher SNR values indicate a stronger signal, making it easier to distinguish from the noise and hence more reliable detection. 

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   Installation
   Codeoverview
   Summary
   innerproduct
   interpolation
   probabilityofdetection
   horizondistance
   ann
   hybrid
   
.. toctree::
   :maxdepth: 2
   :caption: API:

   autoapi/gwsnr/core/index.rst
   autoapi/gwsnr/ann/index.rst
   autoapi/gwsnr/numba/index.rst
   autoapi/gwsnr/jax/index.rst
   autoapi/gwsnr/ripple/index.rst
   autoapi/gwsnr/utils/index.rst
   
.. toctree::
   :maxdepth: 2
   :caption: Examples:

   examples/snr_generation
   examples/model_generation


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
