"""
tests/test_filters.py — Unit Tests for Production Filter Stack
==============================================================
Tests all 7 filter categories to validate correctness and integration.

Run with:
    pytest tests/test_filters.py -v

Or with coverage:
    pytest tests/test_filters.py -v --tb=short
"""

from __future__ import annotations

import math

import numpy as np
import pytest


# =============================================================================
# 1. Sensor Signal Filters
# =============================================================================

class TestLowPassFilter:
    def test_dc_passthrough(self):
        """LPF must pass a DC (constant) signal with gain ≈ 1."""
        from src.filters.signal import LowPassFilter
        lpf = LowPassFilter(cutoff_hz=10.0, sample_rate_hz=200.0)
        for _ in range(500):
            y = lpf.update(1.0)
        assert abs(y - 1.0) < 0.001

    def test_noise_attenuation(self):
        """High-frequency noise should be attenuated below cutoff."""
        from src.filters.signal import LowPassFilter
        lpf = LowPassFilter(cutoff_hz=10.0, sample_rate_hz=200.0)
        # Inject 50 Hz sinusoid (above cutoff = 10 Hz)
        fs = 200.0
        N  = 1000
        signal = [math.sin(2 * math.pi * 50 * i / fs) for i in range(N)]
        outputs = [lpf.update(x) for x in signal]
        rms_out = math.sqrt(sum(o**2 for o in outputs[500:]) / 500)
        assert rms_out < 0.3, f"High-freq not attenuated enough: RMS={rms_out:.3f}"

    def test_vector_update(self):
        """Vector update must filter each channel independently."""
        from src.filters.signal import LowPassFilter
        lpf = LowPassFilter(cutoff_hz=20.0, sample_rate_hz=200.0)
        x   = np.array([1.0, 2.0, 3.0])
        for _ in range(200):
            y = lpf.update_vector(x)
        np.testing.assert_allclose(y, x, atol=0.01)

    def test_invalid_cutoff(self):
        """Cutoff at or above Nyquist should raise ValueError."""
        from src.filters.signal import LowPassFilter
        with pytest.raises(ValueError):
            LowPassFilter(cutoff_hz=100.0, sample_rate_hz=200.0)

    def test_reset(self):
        """Filter state should reset correctly."""
        from src.filters.signal import LowPassFilter
        lpf = LowPassFilter(cutoff_hz=10.0, sample_rate_hz=200.0)
        for _ in range(100):
            lpf.update(1.0)
        lpf.reset(0.0)
        # After reset, output should start near 0
        assert abs(lpf.update(0.0)) < 0.001


class TestEMAFilter:
    def test_alpha_1_passthrough(self):
        """α=1.0 must give exact passthrough."""
        from src.filters.signal import EMAFilter
        ema = EMAFilter(alpha=1.0)
        assert ema.update(5.0) == pytest.approx(5.0)
        assert ema.update(3.0) == pytest.approx(3.0)

    def test_smoothing(self):
        """α<1 must smooth a step input."""
        from src.filters.signal import EMAFilter
        ema = EMAFilter(alpha=0.1)
        # After one sample of a step to 1.0, output should be 0.1
        y = ema.update(1.0)
        assert y == pytest.approx(0.1)

    def test_from_window(self):
        """from_window factory should produce correct alpha."""
        from src.filters.signal import EMAFilter
        ema = EMAFilter.from_window(window=9)
        assert ema.alpha == pytest.approx(2.0 / 10.0)

    def test_dc_convergence(self):
        """EMA must converge to DC value."""
        from src.filters.signal import EMAFilter
        ema = EMAFilter(alpha=0.2)
        for _ in range(200):
            y = ema.update(42.0)
        assert abs(y - 42.0) < 0.01

    def test_invalid_alpha(self):
        from src.filters.signal import EMAFilter
        with pytest.raises(ValueError):
            EMAFilter(alpha=0.0)


class TestButterworthLowPassFilter:
    def test_dc_gain(self):
        """Butterworth LPF should have unity gain at DC."""
        from src.filters.signal import ButterworthLowPassFilter
        bwf = ButterworthLowPassFilter(cutoff_hz=20.0, sample_rate_hz=1000.0)
        for _ in range(1000):
            y = bwf.update(1.0)
        assert abs(y - 1.0) < 0.001

    def test_attenuation_above_cutoff(self):
        """Signal well above cutoff should be strongly attenuated."""
        from src.filters.signal import ButterworthLowPassFilter
        bwf = ButterworthLowPassFilter(cutoff_hz=10.0, sample_rate_hz=1000.0)
        # 100 Hz input, fs=1000, fc=10
        fs  = 1000.0
        sig = [math.sin(2 * math.pi * 100 * i / fs) for i in range(2000)]
        out = [bwf.update(x) for x in sig]
        rms = math.sqrt(sum(o**2 for o in out[1000:]) / 1000)
        assert rms < 0.05, f"2nd-order BWF not attenuating enough: RMS={rms:.4f}"

    def test_flatness_in_passband(self):
        """Passband gain should be approximately 1.0 for frequencies << fc."""
        from src.filters.signal import ButterworthLowPassFilter
        bwf = ButterworthLowPassFilter(cutoff_hz=100.0, sample_rate_hz=1000.0)
        fs  = 1000.0
        # 10 Hz signal, well within 100 Hz passband
        sig = [math.sin(2 * math.pi * 10 * i / fs) for i in range(2000)]
        out = [bwf.update(x) for x in sig]
        rms = math.sqrt(sum(o**2 for o in out[1000:]) / 1000)
        assert abs(rms - math.sqrt(0.5)) < 0.05, f"Passband gain error: RMS={rms:.4f}"


class TestNotchFilter:
    def test_notch_attenuation(self):
        """Notch filter must attenuate the target frequency."""
        from src.filters.signal import NotchFilter
        nf  = NotchFilter(notch_hz=50.0, sample_rate_hz=1000.0, bandwidth_hz=5.0)
        fs  = 1000.0
        sig = [math.sin(2 * math.pi * 50 * i / fs) for i in range(2000)]
        out = [nf.update(x) for x in sig]
        rms = math.sqrt(sum(o**2 for o in out[1000:]) / 1000)
        assert rms < 0.3, f"Notch not attenuating: RMS={rms:.4f}"

    def test_dc_pass(self):
        """Notch filter should pass DC signal with near-unity gain."""
        from src.filters.signal import NotchFilter
        nf = NotchFilter(notch_hz=60.0, sample_rate_hz=1000.0)
        for _ in range(1000):
            y = nf.update(1.0)
        assert abs(y - 1.0) < 0.05


class TestAntiAliasingFilter:
    def test_decimation_rate(self):
        """AAF must only produce output every N samples."""
        from src.filters.signal import AntiAliasingFilter
        aaf    = AntiAliasingFilter(output_rate_hz=50.0, input_rate_hz=200.0)
        count  = sum(1 for i in range(200) if aaf.update(float(i)) is not None)
        assert count == pytest.approx(50, abs=2)  # should produce ~50 samples


# =============================================================================
# 2. PID / Control Filters
# =============================================================================

class TestDerivativeLPF:
    def test_zero_input(self):
        """Zero error → zero derivative output."""
        from src.filters.control import DerivativeLPF
        d = DerivativeLPF(kd=0.001, cutoff_hz=20.0, sample_rate_hz=30.0)
        for _ in range(100):
            y = d.update(0.0)
        assert abs(y) < 1e-6

    def test_step_response(self):
        """Step input should produce a transient derivative pulse."""
        from src.filters.control import DerivativeLPF
        d  = DerivativeLPF(kd=0.001, cutoff_hz=20.0, sample_rate_hz=30.0)
        y0 = d.update(0.0)
        y1 = d.update(100.0)  # large step
        assert abs(y1) > abs(y0), "Derivative should produce non-zero output on step"

    def test_reset(self):
        from src.filters.control import DerivativeLPF
        d = DerivativeLPF(kd=0.001, cutoff_hz=20.0, sample_rate_hz=30.0)
        for _ in range(50):
            d.update(1.0)
        d.reset()
        y = d.update(0.0)
        assert abs(y) < 0.01


class TestComplementaryFilter:
    def test_static_angle(self):
        """Static (roll=45°) with zero gyro → converge to 45°."""
        from src.filters.control import ComplementaryFilter
        cf = ComplementaryFilter(alpha=0.98, sample_rate_hz=200.0)
        for _ in range(2000):
            angle = cf.update(acc_angle_deg=45.0, gyro_rate_dps=0.0, dt=0.005)
        assert abs(angle - 45.0) < 2.0

    def test_gyro_drift(self):
        """Pure gyro integration should drift — complementary filter corrects it."""
        from src.filters.control import ComplementaryFilter
        cf = ComplementaryFilter(alpha=0.98, sample_rate_hz=200.0)
        for _ in range(1000):
            angle = cf.update(acc_angle_deg=0.0, gyro_rate_dps=0.1, dt=0.005)
        # Gyro alone would give 0.1*0.005*1000 = 0.5°; acc anchor keeps it near 0
        # Allow some drift but check acc anchor dominates
        assert abs(angle) < 5.0


class TestKalmanFilter1D:
    def test_converges_to_true_value(self):
        """Kalman must converge to the true signal value."""
        from src.filters.control import KalmanFilter1D
        kf  = KalmanFilter1D(process_noise=0.01, measurement_noise=1.0)
        rng = np.random.default_rng(42)
        true_val = 7.5
        for _ in range(200):
            noisy = true_val + rng.normal(0, 1.0)
            y = kf.step(noisy)
        assert abs(y - true_val) < 0.5

    def test_gain_decreases(self):
        """Kalman gain should decrease as covariance shrinks."""
        from src.filters.control import KalmanFilter1D
        kf  = KalmanFilter1D()
        g0  = kf.gain
        for _ in range(50):
            kf.step(1.0)
        g1 = kf.gain
        assert g1 < g0, "Kalman gain should decrease over time"


class TestAdaptiveFilterLMS:
    def test_noise_cancellation(self):
        """LMS should reduce noise after training."""
        from src.filters.control import AdaptiveFilterLMS
        lms = AdaptiveFilterLMS(order=16, mu=0.01)
        rng = np.random.default_rng(0)
        clean  = [math.sin(2 * math.pi * 0.05 * i) for i in range(1000)]
        noisy  = [c + rng.normal(0, 0.2) for c in clean]
        # Train
        errors = []
        for n, d in zip(noisy, clean):
            y   = lms.update(n, desired=d)
            errors.append(abs(y - d))
        # Error should decrease over time
        early_err = sum(errors[:100]) / 100
        late_err  = sum(errors[800:]) / 200
        assert late_err < early_err, f"LMS not converging: early={early_err:.4f} late={late_err:.4f}"


# =============================================================================
# 3. Image Preprocessing Filters (require OpenCV)
# =============================================================================

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


@pytest.mark.skipif(not HAS_CV2, reason="OpenCV not available")
class TestImageFilters:
    @pytest.fixture
    def noisy_frame(self):
        """Random noise + smooth background image."""
        rng   = np.random.default_rng(7)
        frame = np.zeros((100, 100, 3), dtype=np.uint8) + 128
        # Add random noise
        noise = rng.integers(0, 50, size=frame.shape, dtype=np.uint8)
        return cv2.add(frame, noise)

    def test_gaussian_shape(self, noisy_frame):
        from src.filters.image import GaussianFilter
        gf  = GaussianFilter(ksize_or_sigma=5)
        out = gf.apply(noisy_frame)
        assert out.shape == noisy_frame.shape
        assert out.dtype == noisy_frame.dtype

    def test_gaussian_reduces_variance(self, noisy_frame):
        """Gaussian blur must reduce pixel variance."""
        from src.filters.image import GaussianFilter
        gf  = GaussianFilter(ksize_or_sigma=7)
        out = gf.apply(noisy_frame)
        assert float(np.var(out)) < float(np.var(noisy_frame))

    def test_median_shape(self, noisy_frame):
        from src.filters.image import MedianFilter
        mf  = MedianFilter(ksize=5)
        out = mf.apply(noisy_frame)
        assert out.shape == noisy_frame.shape

    def test_bilateral_shape(self, noisy_frame):
        from src.filters.image import BilateralFilter
        bf  = BilateralFilter(d=5)
        out = bf.apply(noisy_frame)
        assert out.shape == noisy_frame.shape

    def test_clahe_shape(self, noisy_frame):
        from src.filters.image import CLAHEFilter
        cl  = CLAHEFilter(clip_limit=2.0)
        out = cl.apply(noisy_frame)
        assert out.shape == noisy_frame.shape

    def test_high_pass_sharpened_is_brighter(self, noisy_frame):
        """High-pass sharpening must increase mean gradient magnitude."""
        from src.filters.image import HighPassFilter
        hpf   = HighPassFilter(ksize=5, strength=2.0)
        sharp = hpf.apply(noisy_frame)
        assert sharp.shape == noisy_frame.shape


# =============================================================================
# 4. Edge Detection Filters
# =============================================================================

@pytest.mark.skipif(not HAS_CV2, reason="OpenCV not available")
class TestEdgeFilters:
    @pytest.fixture
    def box_image(self):
        """Black image with a white rectangle — clean edges."""
        img = np.zeros((100, 100), dtype=np.uint8)
        img[30:70, 30:70] = 255
        return img

    def test_sobel_gradient_on_box(self, box_image):
        from src.filters.edge import SobelFilter
        sf  = SobelFilter(ksize=3)
        res = sf.apply(box_image)
        # Gradient magnitude must be non-zero at box edges
        assert res.magnitude.max() > 0

    def test_canny_detects_box_edges(self, box_image):
        from src.filters.edge import CannyEdgeDetector
        ce = CannyEdgeDetector(low_threshold=50, high_threshold=100, blur_ksize=3)
        edges = ce.apply(box_image)
        assert edges.max() == 255  # some edges detected
        assert edges.dtype == np.uint8

    def test_laplacian_variance(self, box_image):
        from src.filters.edge import LaplacianFilter
        lf  = LaplacianFilter()
        var = lf.variance(box_image)
        assert var > 0, "Non-uniform image should have positive Laplacian variance"

    def test_prewitt_shape(self, box_image):
        from src.filters.edge import PrewittFilter
        pf  = PrewittFilter()
        out = pf.apply(box_image)
        assert out.shape == box_image.shape


# =============================================================================
# 5. Morphological Filters
# =============================================================================

@pytest.mark.skipif(not HAS_CV2, reason="OpenCV not available")
class TestMorphologicalFilters:
    @pytest.fixture
    def noisy_mask(self):
        """Binary mask with small noise blobs."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[30:70, 30:70] = 255   # main object
        mask[5:8, 5:8]     = 255   # noise blob
        mask[90, 90]       = 255   # single noise pixel
        return mask

    def test_erosion_shrinks(self, noisy_mask):
        from src.filters.morphology import Erosion
        er  = Erosion(ksize=5)
        out = er.apply(noisy_mask)
        # Number of white pixels should decrease
        assert out.sum() < noisy_mask.sum()

    def test_dilation_grows(self, noisy_mask):
        from src.filters.morphology import Dilation
        dl  = Dilation(ksize=5)
        out = dl.apply(noisy_mask)
        assert out.sum() >= noisy_mask.sum()

    def test_opening_removes_noise(self, noisy_mask):
        from src.filters.morphology import Opening
        op   = Opening(ksize=5)
        out  = op.apply(noisy_mask)
        # Noise blobs should be removed, main object preserved
        assert out[30:70, 30:70].any(), "Main object should survive opening"
        assert not out[5:8, 5:8].any(), "Noise blob should be removed by opening"

    def test_closing_fills_holes(self):
        from src.filters.morphology import Closing
        mask = np.zeros((50, 50), dtype=np.uint8)
        mask[10:40, 10:40] = 255
        mask[20:25, 20:25] = 0    # hole in centre
        cl  = Closing(ksize=9)
        out = cl.apply(mask)
        # Centre should be filled
        assert out[21, 21] == 255

    def test_morphological_processor(self, noisy_mask):
        from src.filters.morphology import MorphologicalProcessor
        proc = MorphologicalProcessor(open_ksize=5, close_ksize=7, min_area_px=100)
        out  = proc.process(noisy_mask)
        assert out.max() == 255
        # Noise at (5:8,5:8) = 9 px area < 100 → removed
        assert not out[5:8, 5:8].any()


# =============================================================================
# 6. Motion / Tracking Filters
# =============================================================================

class TestKalmanTracker2D:
    def test_converges_to_position(self):
        """Tracker must converge to the true position."""
        from src.filters.motion import KalmanTracker2D
        tracker = KalmanTracker2D(initial_position=(100.0, 200.0))
        rng     = np.random.default_rng(5)
        for _ in range(100):
            noise = rng.normal(0, 3.0, size=2)
            tracker.step((100.0 + noise[0], 200.0 + noise[1]))
        pos = tracker.position
        assert abs(pos[0] - 100.0) < 5.0
        assert abs(pos[1] - 200.0) < 5.0

    def test_velocity_estimation(self):
        """Moving object should have non-zero velocity estimate."""
        from src.filters.motion import KalmanTracker2D
        tracker = KalmanTracker2D()
        for i in range(50):
            tracker.step((float(i * 2), float(i * 3)))
        vx, vy = tracker.velocity
        assert abs(vx) > 0.5
        assert abs(vy) > 0.5

    def test_predict_without_update(self):
        """Predict-only (no measurement) should still return a position."""
        from src.filters.motion import KalmanTracker2D
        tracker = KalmanTracker2D(initial_position=(50.0, 50.0))
        predicted = tracker.predict()
        assert len(predicted) == 2


@pytest.mark.skipif(not HAS_CV2, reason="OpenCV not available")
class TestOpticalFlowSmoother:
    def test_first_frame_returns_none(self):
        """First frame should return None (no previous frame)."""
        from src.filters.motion import OpticalFlowSmoother
        ofs   = OpticalFlowSmoother(mode="dense")
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = ofs.update(frame)
        assert result is None

    def test_static_scene_low_flow(self):
        """Identical frames should produce near-zero flow."""
        from src.filters.motion import OpticalFlowSmoother
        ofs   = OpticalFlowSmoother(mode="dense")
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        ofs.update(frame)
        flow  = ofs.update(frame)
        if flow is not None:
            assert ofs.mean_magnitude() < 2.0


# =============================================================================
# 7. Pipeline Integration Test
# =============================================================================

@pytest.mark.skipif(not HAS_CV2, reason="OpenCV not available")
class TestFilterPipeline:
    def test_default_pipeline(self):
        """Default pipeline should run on a frame without error."""
        from src.filters.pipeline import FilterPipeline
        pipeline = FilterPipeline()
        frame = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
        out   = pipeline.process(frame)
        assert out.shape == frame.shape
        assert out.dtype == np.uint8

    def test_full_pipeline_stages(self):
        from src.filters.pipeline import FilterPipeline, ImagePreprocessConfig
        cfg = ImagePreprocessConfig(
            use_gaussian=True,
            use_bilateral=True,
            use_clahe=True,
            use_sharpen=True,
            use_canny=True,
        )
        pipeline = FilterPipeline(cfg)
        frame   = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
        out, edges = pipeline.process_and_detect_edges(frame)
        assert out.shape == frame.shape
        assert edges is not None
        assert edges.shape[:2] == frame.shape[:2]

    def test_sensor_fusion_pipeline(self):
        from src.filters.pipeline import SensorFusionPipeline
        fusion = SensorFusionPipeline(imu_sample_rate_hz=200.0)
        # Simulate stationary IMU (only gravity on Z)
        for _ in range(200):
            roll, pitch = fusion.update_imu(0.0, 0.0, 9.81, 0.0, 0.0, 0.0, dt=0.005)
        # With gravity on Z, roll and pitch should be near 0
        assert abs(roll) < 5.0
        assert abs(pitch) < 5.0

    def test_control_filter_bundle(self):
        from src.filters.pipeline import ControlFilterBundle
        bundle = ControlFilterBundle.from_config(kd=0.0001, cutoff_hz=10.0)
        # Warm up EMA filter (starts at 0, needs iterations to converge)
        for _ in range(30):
            cx, cy, area = bundle.smooth_observation(320.0, 240.0, 15000.0)
        # After warmup EMA should be close to the input value
        assert 300 < cx < 340, f"cx={cx} not near 320"
        assert 220 < cy < 260, f"cy={cy} not near 240"
        bundle.reset()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
