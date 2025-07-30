:orphan:

:py:mod:`gwsnr.numba.test`
==========================

.. py:module:: gwsnr.numba.test


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   gwsnr.numba.test.cubic_function_4pts_numba



.. py:function:: cubic_function_4pts_numba(x_eval, x_pts, y_pts, condition_i)

   
   Catmull-Rom spline interpolation (Numba-friendly, 4 points).
   x_pts: 4 points, assumed sorted, uniform or non-uniform spacing.
   y_pts: values at those points.
   x_eval: point at which to evaluate.
   condition_i: for edge handling.
















   ..
       !! processed by numpydoc !!

