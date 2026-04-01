"""
PseudopopulationDecoder - Legacy module for backward compatibility
Classes have been moved to individual files:
- ConfigLoader -> Defineparameters.py (replaces old Defineparameters)
- SlidingWindowDecoder -> SlidingWindowDecoder.py

Import from this module for backward compatibility, or import directly from the individual files.

Example:
    from analysis_pseudopopulation import ConfigLoader, SlidingWindowDecoder
    # or
    from analysis_pseudopopulation.Defineparameters import ConfigLoader
    from analysis_pseudopopulation.SlidingWindowDecoder import SlidingWindowDecoder
"""

from .ConfigLoader import ConfigLoader
from .SlidingWindowDecoder import SlidingWindowDecoder
from .DecoderCrossTester import DecoderCrossTester  

__all__ = ['ConfigLoader', 'SlidingWindowDecoder', 'DecoderCrossTester']





