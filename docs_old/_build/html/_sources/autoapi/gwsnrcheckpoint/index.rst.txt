:orphan:

:py:mod:`gwsnr-checkpoint`
==========================

.. py:module:: gwsnr-checkpoint


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   gwsnr-checkpoint.NumpyEncoder




Attributes
~~~~~~~~~~

.. autoapisummary::

   gwsnr-checkpoint.MTSUN_SI


.. py:data:: MTSUN_SI
   :value: 4.925491025543576e-06

   




   ------------------------------------------------
       class containing following methods
       1. to calculate fast SNR
       2. interpolation of with cubic spline
       with bilby SNR
       3. Pdet: probability of detection
   ------------------------------------------------
















   ..
       !! processed by numpydoc !!

.. py:class:: NumpyEncoder(*, skipkeys=False, ensure_ascii=True, check_circular=True, allow_nan=True, sort_keys=False, indent=None, separators=None, default=None)


   Bases: :py:obj:`json.JSONEncoder`

   
   Extensible JSON <https://json.org> encoder for Python data structures.

   Supports the following objects and types by default:

   +-------------------+---------------+
   | Python            | JSON          |
   +===================+===============+
   | dict              | object        |
   +-------------------+---------------+
   | list, tuple       | array         |
   +-------------------+---------------+
   | str               | string        |
   +-------------------+---------------+
   | int, float        | number        |
   +-------------------+---------------+
   | True              | true          |
   +-------------------+---------------+
   | False             | false         |
   +-------------------+---------------+
   | None              | null          |
   +-------------------+---------------+

   To extend this to recognize other objects, subclass and implement a
   ``.default()`` method with another method that returns a serializable
   object for ``o`` if possible, otherwise it should call the superclass
   implementation (to raise ``TypeError``).















   ..
       !! processed by numpydoc !!
   .. py:method:: default(obj)

      
      Implement this method in a subclass such that it returns
      a serializable object for ``o``, or calls the base implementation
      (to raise a ``TypeError``).

      For example, to support arbitrary iterators, you could
      implement default like this::

          def default(self, o):
              try:
                  iterable = iter(o)
              except TypeError:
                  pass
              else:
                  return list(iterable)
              # Let the base class default method raise the TypeError
              return JSONEncoder.default(self, o)















      ..
          !! processed by numpydoc !!


