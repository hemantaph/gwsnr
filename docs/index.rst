Welcome to gwsnr’s documentation!
================================

.. raw:: html

   <div style="text-align: center; margin: 20px 0; padding: 20px;">
      <img src="_static/logo.svg" width="40%" alt="gwsnr logo">
   </div>

gwsnr: Efficient :red_first:`Gravitational`-:red_first:`Wave` :red_first:`Signal`-to-:red_first:`Noise` :red_first:`Ratio` Calculator
------------------------------

``gwsnr`` is a Python package for the efficient and accurate computation of the Signal-to-Noise Ratio (SNR) in gravitational-wave (GW) astronomy.

The package addresses the computational bottleneck of traditional SNR calculations by implementing advanced interpolation techniques, Just-in-Time (JIT) compilation, and parallel processing. It offers flexible backends using ``Numba`` for multi-core CPU optimization and ``JAX`` for GPU acceleration. With a simple API, ``gwsnr`` is designed for easy integration into existing analysis workflows and is used by the ``ler`` (`see ler documentation <https://ler.readthedocs.io/en/latest/>`_).

For a detailed technical overview, please see the :doc:`Summary` section.

.. raw:: html

   <div style="background-color:rgb(233, 233, 233); border-left: 5px solid rgb(247, 159, 43); padding: 10px; margin-top: 20px; margin-bottom: 20px;">
      <p style="font-size: 1em; margin: 0;">
         <em><strong>Highlight:</strong> Achieve a speed-up of more than <strong>10,000x</strong> in SNR calculations compared to traditional methods, while maintaining an accuracy greater than <strong>99.5%</strong>.</em>
      </p>
   </div>

Quick Start
-----------

Install the package using pip:

.. code-block:: bash

   pip install gwsnr

Then, compute the SNR for a binary black hole system:

.. code-block:: python

   from gwsnr import GWSNR

   # Initialize the default calculator
   gwsnr = GWSNR()

   # Compute SNR for a 30-30 Msun binary at 1000 Mpc
   snrs = gwsnr.snr(
       mass_1=30,
       mass_2=30,
       luminosity_distance=1000,
       psi=0.0,
       phase=0.0,
       geocent_time=1246527224.169434,
       ra=0.0,
       dec=0.0
   )

   print(f"Network Optimal SNR: {snrs['optimal_snr_net']:.2f}")

.. note::

   ``gwsnr`` supports Python 3.10+ and automatically utilizes multi-core CPUs and NVIDIA GPUs when available. Refer to the :doc:`Installation` section for detailed setup instructions.

About the Project
-----------------

* **Source Code:** `github.com/hemantaph/gwsnr <https://github.com/hemantaph/gwsnr>`_
* **Issue Tracker:** `Report an issue <https://github.com/hemantaph/gwsnr/issues>`_
* **Main Developer:** `Hemanta Ph. <https://www.hemantaph.com>`_
* **Contributor:** `Otto Akseli Hannuksela <https://www.oahannuksela.com/>`_
* **Citation:** If you use ``gwsnr`` in your research, please cite the `gwsnr paper <https://arxiv.org/abs/2412.09888>`_.



.. _glossary:

Glossary
========

.. glossary::

   Gravitational waves

      Ripples in the fabric of spacetime, first predicted by Albert Einstein in his theory of General Relativity in 1916. They are created by some of the most violent and energetic events in the universe, such as the collision of black holes, the merging of neutron stars, or supernova explosions. These waves travel outward from their source at the speed of light, carrying information about their origins and the nature of gravity itself.

      For a century, gravitational waves remained a theoretical prediction. It wasn't until 2015 that the LIGO and Virgo collaborations made the first direct detection, an achievement that earned the 2017 Nobel Prize in Physics and opened an entirely new way of observing the cosmos.

      .. raw:: html

         <div style="text-align:center;">
         <img src="_static/gw.gif" width="480px" alt="Animation of GW propagation">
         <div style="text-align:left; max-width:480px">
            <p style="font-size: 0.9em; font-family: Arial, sans-serif; line-height: 1.5em;">
               Animation showing the propagation of gravitational waves from inspiraling binary black holes. As the waves travel, they stretch and squeeze spacetime in their path. <em>Source: <a href="https://community.wolfram.com/groups/-/m/t/790989">Jeffrey Bryant, Wolfram | Alpha, LLC.</a>.</em>
            </p>
         </div>
         </div>

   Detection of gravitational waves

      The effect of a passing gravitational wave is incredibly subtle. To detect these faint signals, scientists use enormous L-shaped instruments called laser interferometers. The most prominent detectors are the two LIGO observatories in the United States, the Virgo detector in Italy, and the KAGRA detector in Japan. These instruments use lasers to measure minute changes in the lengths of their kilometers-long arms—changes on the order of 1/10,000th the width of a proton.

      Because the signals are so weak, they are often buried in the detector's background noise. To find them, scientists use a technique called **matched filtering**. This involves comparing the noisy detector data against a large bank of theoretical waveform templates. When a segment of the data closely matches a template, the signal "rings up," indicating a potential detection.

      .. raw:: html

         <div style="text-align:center;">
         <img src="_static/matched_filtering.gif" width="600px" alt="Animation of Matched Filtering">
         <div style="text-align:left; max-width:600px">
            <p style="font-size: 0.9em; font-family: Arial, sans-serif; line-height: 1.5em;">
               Animation of the matched filtering technique. The bottom panel shows a theoretical gravitational-wave signal (pink) hidden within noisy detector data (black). The top panel shows the Signal-to-Noise Ratio (SNR) calculated from this data. A confident detection is claimed when the SNR forms a sharp peak that crosses a pre-defined threshold (yellow dashed line).
                  <em>Source: <a href="https://www.youtube.com/watch?v=bBBDR5jf9oU">Alex Nitz | YouTube</a>.</em>
            </p>
         </div>
         </div>

   Signal-to-noise ratio (SNR)
   
      The central metric used to assess the strength and significance of a potential gravitational-wave signal. It quantifies how much stronger the signal is compared to the average level of the background noise.

      A higher SNR indicates a more confident detection, making it easier to distinguish a real astrophysical event from random noise fluctuations. A detection is typically confirmed only when the SNR peak surpasses a certain threshold value. The ``gwsnr`` package is designed to compute this critical quantity efficiently and accurately for a wide range of astrophysical scenarios.

.. raw:: html

    <iframe src="_static/gwlensing.html"
            width="100%"
            height="600"
            frameborder="0"
            allowfullscreen
            style="border:1px solid #ccc; border-radius:10px;"></iframe>




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
