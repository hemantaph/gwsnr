Welcome to :red:`gwsnr`’s documentation!
================================

.. raw:: html

   <div style="text-align: center; margin-top: 0px; margin-bottom: 10px; padding-top: 0px; padding-bottom: 10px;">
      <img src="_static/logo.png" width="220rem" alt="gwsnr logo">
   </div>

:red:`gwsnr` : Efficient :red_first:`Gravitational`-:red_first:`Wave` :red_first:`Signal`-to-:red_first:`Noise` :red_first:`Ratio` Calculator
------------------------------

``gwsnr`` is a Python package for fast and accurate gravitational-wave (GW) Signal-to-Noise Ratio (SNR) calculation. It is designed for semi-analytical Probability of Detection (Pdet) estimation in population simulations and for hierarchical Bayesian inference with selection effects. The package achieves a **speed-up of more than 5,000x** compared to traditional methods, while maintaining an **accuracy above 99.5%**.

``gwsnr`` removes computational bottlenecks using advanced interpolation, Just-in-Time (JIT) compilation, and parallel processing. It provides multiple backends optimized for different hardware: ``numba`` for multi-threaded CPU performance, and ``jax`` or ``mlx`` for GPU acceleration.

With a simple API, ``gwsnr`` integrates smoothly into existing workflows. It is also used by the ``ler`` package (`see ler documentation <https://ler.readthedocs.io/en/latest/>`_), which simulates detectable lensed and unlensed GWs. This allows researchers to include fast Pdet calculations (via SNR) with minimal overhead.

For a detailed technical overview, see the :doc:`Summary` section, and browse the topics under :blue:`CONTENTS` in the sidebar (or top-left menu) of your browser.


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

   # Compute SNR for a 30-30 Msun binary at 1000 Mpc with other random extrinsic parameters
   snrs = gwsnr.optimal_snr(mass_1=30, mass_2=30, luminosity_distance=1000, psi=0.0, phase=0.0, geocent_time=1246527224.169434, ra=0.0, dec=0.0)

   print(f"Network Optimal SNR: {snrs['snr_net']:.2f}")

.. note::

   ``gwsnr`` supports Python 3.10+ (but 3.11 recommended) and utilizes multi-core CPUs and NVIDIA GPUs (or Apple Silicon GPUs) when available. Refer to the :doc:`Installation` section for detailed setup instructions.

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
          <div style="text-align:left; max-width:480px; margin: 0 auto;">
            <p style="font-size: 0.9em; font-family: Arial, sans-serif; line-height: 1.5em;">
               Animation showing the propagation of gravitational waves from inspiraling binary black holes. As the waves travel, they stretch and squeeze spacetime in their path. <em>Source: <a href="https://community.wolfram.com/groups/-/m/t/790989">Jeffrey Bryant, Wolfram | Alpha, LLC.</a>.</em>
            </p>
         </div>
         </div>

   Detection of gravitational waves

      The effect of a passing gravitational wave is incredibly subtle. To detect these faint signals, scientists use enormous L-shaped instruments called laser interferometers. The most prominent detectors are the two LIGO observatories in the United States, the Virgo detector in Italy, and the KAGRA detector in Japan. These instruments use lasers to measure minute changes in the lengths of their kilometers-long arms—changes on the order of 1/10,000th the width of a proton.

      Since gravitational-wave signals are extremely weak, they are typically obscured by the detector's background noise. Scientists employ a data analysis technique called **matched filtering** to extract these hidden signals and compute the **Signal-to-Noise Ratio**. This process involves cross-correlating the noisy detector data with an extensive library of theoretical gravitational-wave templates. When the data containing a genuine signal aligns with a matching template, the SNR increases dramatically, creating a distinctive peak that indicates a potential detection once it exceeds a pre-determined threshold.

      .. raw:: html

         <div style="text-align:center;">
         <img src="_static/matched_filtering.gif" width="600px" alt="Animation of Matched Filtering">
         <div style="text-align:left; max-width:600px; margin: 0 auto;">
            <p style="font-size: 0.9em; font-family: Arial, sans-serif; line-height: 1.5em;">
               Animation of the matched filtering technique. The bottom panel shows a theoretical gravitational-wave signal (pink) scanning a hidden signal within noisy detector data (black). The top panel shows the Signal-to-Noise Ratio (SNR) calculated from this data. A confident detection is claimed when the SNR forms a sharp peak that crosses a pre-defined threshold (yellow dashed line).
                  <em>Source: <a href="https://www.youtube.com/watch?v=bBBDR5jf9oU">Alex Nitz | YouTube</a>.</em>
            </p>
         </div>
         </div>

      The ``gwsnr`` package efficiently computes this critical quantity for large-scale astrophysical simulations. Unlike detection pipelines such as ``PyCBC`` and ``GstLAL``, ``gwsnr`` is optimized for rapid SNR calculation across thousands or even millions of gravitational-wave simulated events in population studies.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   Installation
   Codeoverview
   Summary
   performancesummary
   innerproduct
   detectionstatistics
   psd
   interpolation
   probabilityofdetection
   horizondistance
   ann
   hybrid
   
.. toctree::
   :maxdepth: 2
   :caption: API:

   autoapi/gwsnr/core/index
   autoapi/gwsnr/numba/index
   autoapi/gwsnr/jax/index
   autoapi/gwsnr/mlx/index
   autoapi/gwsnr/ann/index
   autoapi/gwsnr/ripple/index
   autoapi/gwsnr/utils/index

   
.. toctree::
   :maxdepth: 2
   :caption: Examples:

   examples/snr_generation
   examples/pdet_generation
   examples/model_generation
   examples/threshold
   examples/horizon_distance


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
