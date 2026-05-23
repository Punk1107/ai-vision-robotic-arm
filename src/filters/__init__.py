"""
filters/__init__.py — Production Filter Stack
==============================================
AI + Computer Vision + Robotics Senior Project
Complete filter pipeline covering all 7 categories:

  1. Sensor Signal Filters       (signal.py)
  2. PID / Control Filters       (control.py)
  3. Camera / Image Preprocessing (image.py)
  4. Edge Detection / Feature    (edge.py)
  5. Morphological Filters       (morphology.py)
  6. Motion / Tracking Filters   (motion.py)
  7. AI / Deep Learning Related  (ai_filters.py)

Usage::

    from src.filters import FilterPipeline
    pipeline = FilterPipeline()

    # Sensor signal
    from src.filters.signal import LowPassFilter, EMAFilter, ButterworthFilter
    from src.filters.control import DerivativeLPF, ComplementaryFilter, KalmanFilter1D
    from src.filters.image import GaussianFilter, BilateralFilter, MedianFilter
    from src.filters.edge import CannyEdgeDetector, SobelFilter
    from src.filters.morphology import MorphologicalProcessor
    from src.filters.motion import KalmanTracker2D, OpticalFlowSmoother
"""

from src.filters.signal import (
    LowPassFilter,
    EMAFilter,
    ButterworthLowPassFilter,
    NotchFilter,
    BandPassFilter,
    AntiAliasingFilter,
)
from src.filters.control import (
    DerivativeLPF,
    ComplementaryFilter,
    KalmanFilter1D,
    AdaptiveFilterLMS,
)
from src.filters.image import (
    GaussianFilter,
    MedianFilter,
    BilateralFilter,
    WienerFilter,
    HighPassFilter,
    CLAHEFilter,
)
from src.filters.edge import (
    SobelFilter,
    CannyEdgeDetector,
    LaplacianFilter,
    PrewittFilter,
)
from src.filters.morphology import (
    MorphologicalProcessor,
    Erosion,
    Dilation,
    Opening,
    Closing,
)
from src.filters.motion import (
    KalmanTracker2D,
    OpticalFlowSmoother,
)
from src.filters.pipeline import FilterPipeline

__all__ = [
    # Signal
    "LowPassFilter", "EMAFilter", "ButterworthLowPassFilter",
    "NotchFilter", "BandPassFilter", "AntiAliasingFilter",
    # Control
    "DerivativeLPF", "ComplementaryFilter", "KalmanFilter1D", "AdaptiveFilterLMS",
    # Image
    "GaussianFilter", "MedianFilter", "BilateralFilter",
    "WienerFilter", "HighPassFilter", "CLAHEFilter",
    # Edge
    "SobelFilter", "CannyEdgeDetector", "LaplacianFilter", "PrewittFilter",
    # Morphology
    "MorphologicalProcessor", "Erosion", "Dilation", "Opening", "Closing",
    # Motion
    "KalmanTracker2D", "OpticalFlowSmoother",
    # Pipeline
    "FilterPipeline",
]
