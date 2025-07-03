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


.. _glossary:

Glossary
========

.. glossary::

   Gravitational waves
      Ripples in the fabric of spacetime, first predicted by Albert Einstein in his theory of General Relativity in 1916. They are created by some of the most violent and energetic events in the universe, such as the collision of black holes, the merging of neutron stars, or supernova explosions. These waves travel outward from their source at the speed of light, carrying information about their origins and the nature of gravity itself.

      For a century, gravitational waves remained a theoretical prediction. It wasn't until 2015 that the LIGO and Virgo collaborations made the first direct detection, an achievement that earned the 2017 Nobel Prize in Physics and opened an entirely new way of observing the cosmos.

      .. figure:: _static/gw.gif
         :align: center
         :width: 480px
         :alt: Animation of Gravitational Waves

         Animation showing the propagation of gravitational waves from inspiraling binary black holes. As the waves travel, they stretch and squeeze spacetime in their path.
         *Source: Jeffrey Bryant, Wolfram | Alpha, LLC.*

   Detection of gravitational waves
      The effect of a passing gravitational wave is incredibly subtle. To detect these faint signals, scientists use enormous L-shaped instruments called laser interferometers. The most prominent detectors are the two LIGO observatories in the United States, the Virgo detector in Italy, and the KAGRA detector in Japan. These instruments use lasers to measure minute changes in the lengths of their kilometers-long arms—changes on the order of 1/10,000th the width of a proton.

      Because the signals are so weak, they are often buried in the detector's background noise. To find them, scientists use a technique called **matched filtering**. This involves comparing the noisy detector data against a large bank of theoretical waveform templates. When a segment of the data closely matches a template, the signal "rings up," indicating a potential detection.

      .. figure:: _static/matched_filtering.gif
         :align: center
         :width: 600px
         :alt: Animation of Matched Filtering

         Animation of the matched filtering technique. The bottom panel shows a theoretical gravitational-wave signal (pink) hidden within noisy detector data (black). The top panel shows the Signal-to-Noise Ratio (SNR) calculated from this data. A confident detection is claimed when the SNR forms a sharp peak that crosses a pre-defined threshold (orange dashed line).
         *Source: Alex Nitz | YouTube.*

   Signal-to-noise ratio (SNR)
      The central metric used to assess the strength and significance of a potential gravitational-wave signal. It quantifies how much stronger the signal is compared to the average level of the background noise.

      A higher SNR indicates a more confident detection, making it easier to distinguish a real astrophysical event from random noise fluctuations. A detection is typically confirmed only when the SNR peak surpasses a certain threshold value. The ``gwsnr`` package is designed to compute this critical quantity efficiently and accurately for a wide range of astrophysical scenarios.



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
