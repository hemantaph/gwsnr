"""
Core public API for GWSNR.

``import gwsnr.core`` is lightweight; ``GWSNR`` is loaded on first access.
"""

__all__ = ["GWSNR"]


def __getattr__(name):
    """
    Resolve ``GWSNR`` from ``gwsnr.core.gwsnr`` on first use.

    Parameters
    ----------
    name : str
        Attribute name on the ``gwsnr.core`` package.

    Returns
    -------
    type
        ``GWSNR`` class.

    Raises
    ------
    AttributeError
        If ``name`` is not exported by this module.
    """
    if name == "GWSNR":
        from .gwsnr import GWSNR as _GWSNR

        globals()[name] = _GWSNR
        return _GWSNR
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
