:py:mod:`gwsnr`
===============

.. py:module:: gwsnr

.. autoapi-nested-parse::

   
   GWSNR: Gravitational Wave Signal-to-Noise Ratio.

   ``import gwsnr`` only configures lightweight threading defaults; ``GWSNR`` and
   the ``core`` subpackage are loaded on first access (see ``__getattr__``).















   ..
       !! processed by numpydoc !!


Subpackages
-----------
.. toctree::
   :titlesonly:
   :maxdepth: 3

   ann/index.rst
   core/index.rst
   jax/index.rst
   mlx/index.rst
   numba/index.rst
   ripple/index.rst
   threshold/index.rst
   utils/index.rst


Package Contents
----------------


Functions
~~~~~~~~~

.. autoapisummary::

   gwsnr.set_multiprocessing_start_method



Attributes
~~~~~~~~~~

.. autoapisummary::

   gwsnr.__version__


.. py:function:: set_multiprocessing_start_method()

   
   Set ``multiprocessing`` start method once per process when explicitly called.

   Default choices: ``spawn`` on macOS, ``fork`` on other POSIX systems. Windows
   is left unchanged (``spawn``).

   Environment overrides (POSIX): ``GWSNR_USE_SPAWN=True`` or
   ``GWSNR_USE_FORK=True`` (if both are set, ``spawn`` is used and a warning is
   issued).















   ..
       !! processed by numpydoc !!

.. py:data:: __version__
   :value: "'0.5.0'"

   

