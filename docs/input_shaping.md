# Input Shaping System

This document is the complete reference for the three-stage input shaping
system implemented in `src/robotics/trajectory.py`, `src/robotics/shaping.py`,
and `src/robotics/ai_estimator.py`.

## Why Input Shaping?

A hobby servo arm behaves like an under-damped second-order oscillator:

```text
ẍ + 2ζωₙẋ + ωₙ²x = u(t)
```

When `u(t)` changes abruptly (as it does in a trapezoidal velocity profile),
the system rings at ωₙ after the move ends. This causes:

- Visible tip oscillation after each move
- Dropped or misplaced objects
- Mechanical stress and noise

Input shaping eliminates this ring **without slowing the move down significantly**.

---

## Stage 1 — Smooth Trajectory Profiles

### What it solves

The original `TrapezoidalPlanner` produced a velocity profile with
**instantaneous acceleration steps** at the start and end of the move —
mathematically infinite jerk. Even with smooth servos, this excites vibration.

### Planners

Configure via `config.yaml → input_shaping → planner`.

#### `SCurvePlanner` (recommended)

Uses a quintic polynomial `s(τ) = 10τ³ − 15τ⁴ + 6τ⁵` as the normalised
displacement profile. This guarantees:

- Zero velocity at start and end (C1 ✓)
- Zero acceleration at start and end (C2 ✓)
- Configurable maximum jerk via `max_jerk` (deg/s³)

All joints are time-synchronised: the slowest joint determines the total
duration T; faster joints are scaled proportionally.

```python
from src.robotics.trajectory import SCurvePlanner

planner = SCurvePlanner(
    max_vel_deg_s   = 120.0,
    max_acc_deg_s2  = 200.0,
    max_jerk_deg_s3 = 800.0,
    hz              = 50,
)
traj = planner.plan(start_ja, end_ja)
```

#### `JerkLimitedPlanner`

Applies the same quintic (or sigmoid/cubic) profile to trapezoidal timing.
Lighter CPU than full S-curve; still achieves C2 continuity.

```python
from src.robotics.trajectory import JerkLimitedPlanner

# Profile options: "quintic" | "sigmoid" | "cubic"
planner = JerkLimitedPlanner(profile="quintic", max_vel_deg_s=120, hz=50)
traj = planner.plan(start_ja, end_ja)
```

#### `CubicSplinePlanner`

Fits a natural cubic spline through any number of timed waypoints. C1
continuity (no velocity discontinuities). Best for multi-waypoint paths
like pick-lift-place.

```python
from src.robotics.trajectory import CubicSplinePlanner

planner = CubicSplinePlanner(hz=50)
traj = planner.plan([
    (0.0, home_ja),
    (1.2, pick_ja),
    (2.4, lift_ja),
    (3.6, drop_ja),
    (4.8, home_ja),
])
```

#### `TrapezoidalPlanner` (legacy)

Retained for backward compatibility. Internally upgraded to cubic ease-in/out
scaling (no raw linear ramp), so it no longer produces infinite jerk. Not
recommended for new code.

### TrajectoryPoint

All planners return `List[TrajectoryPoint]`:

```python
@dataclass
class TrajectoryPoint:
    t:      float        # time from start (s)
    angles: JointAngles  # joint angles at this time
    vel:    List[float]  # joint velocities  (deg/s),  5-element
    acc:    List[float]  # joint accelerations (deg/s²), 5-element
```

The `acc` field (new in this version) is used by the shaper and by the jerk
monitoring helper `_max_jerk()` in the test suite.

---

## Stage 2 — Classical Input Shaping

### What it solves

Even a perfect S-curve cannot eliminate ring if the trajectory itself excites
the arm's resonant frequency. Classical input shaping convolves the trajectory
with a sequence of impulses chosen to cancel the residual oscillation.

### How it works

For a move command `u(t)`, the shaped command is:

```text
u_shaped(t) = A₁·u(t) + A₂·u(t − t₂) + ...
```

The amplitudes (A) and delays (t) are set so that the system's response to
each shifted copy exactly cancels the ring from the others.

### Shapers

Configure via `config.yaml → input_shaping → shaper_kind`.

#### `ZVShaper` — Zero Vibration

```text
K  = exp(-πζ / √(1 − ζ²))
A₁ = 1/(1+K)    at t = 0
A₂ = K/(1+K)    at t = T_d/2      T_d = 2π/ωₙ√(1−ζ²)
```

Eliminates vibration exactly at the modelled ωₙ. Sensitive to parameter
uncertainty — use ZVD for real arms.

#### `ZVDShaper` — Zero Vibration Derivative (default)

```text
A₁ = 1/(1+K)²      at t = 0
A₂ = 2K/(1+K)²     at t = T_d/2
A₃ = K²/(1+K)²     at t = T_d
```

Zeroes both the vibration and its frequency derivative, giving robustness to
~10% ωₙ modelling error. Recommended for all arms.

#### `EIShaper` — Extra-Insensitive

```text
A₁ = (1 + V_tol) / 4   at t = 0
A₂ = (1 − V_tol) / 2   at t = T_d/2      V_tol default = 0.05
A₃ = (1 + V_tol) / 4   at t = T_d
```

Widest ωₙ tolerance band — useful when payload mass varies significantly.

### Using the Factory

```python
from src.robotics.shaping import build_shaper

# From config (reads config.input_shaping automatically)
shaper = build_shaper()

# Explicit parameters
shaper = build_shaper(kind="zvd", omega_n_rad_s=18.0, zeta=0.10)

# Apply to any trajectory
shaped_traj = shaper.apply(traj)
ctrl.play_trajectory(shaped_traj)
```

### Integration with RobotController

Pass the shaper at construction time; `play_trajectory()` applies it automatically:

```python
shaper = build_shaper(kind="zvd")
ctrl   = RobotController(shaper=shaper)

with ctrl:
    ctrl.play_trajectory(traj)             # shaping applied
    ctrl.play_trajectory(traj, apply_shaping=False)  # bypass for homing
```

### Tuning ωₙ and ζ

Use the measurement script:

```bash
# Simulate (no hardware needed)
python -m scripts.measure_resonance --joint base --dry

# Real hardware + write result to config.yaml
python -m scripts.measure_resonance --joint base --amplitude 20 --update-config
```

Typical hobby arm: ωₙ ≈ 12–25 rad/s, ζ ≈ 0.05–0.20.

### Parameter Reference

```yaml
input_shaping:
  shaper_kind: zvd        # zv | zvd | ei | none
  omega_n:     18.0       # rad/s  — measure with measure_resonance.py
  zeta:        0.10       # damping ratio
```

---

## Stage 3 — AI Adaptive Shaping

### What it solves

The arm's ωₙ changes with:
- Joint configuration (moment of inertia varies with extension)
- Payload mass (holding an object vs. empty gripper)
- Mechanical wear over time

Stage 3 uses an IMU mounted at the end-effector to continuously estimate ωₙ
and ζ in real-time, feeding them into the shaper automatically.

### Requirements

- **Hardware**: IMU (e.g. MPU-6050) at the arm end-effector
- **Firmware**: Arduino must include `"imu": {"ax":…, "ay":…, "az":…}` in
  its serial telemetry JSON response
- **Config**: `input_shaping.adaptive: true`

### Estimators

| Class | Algorithm | When to use |
| --- | --- | --- |
| `FrequencyEstimator` | Windowed FFT (512 pts, Hanning) | Offline calibration / initial seed |
| `RLSEstimator` | Recursive Least Squares, λ=0.98 | Fast convergence, moderate noise |
| `KalmanEstimator` | Extended Kalman Filter, 4-state | Best noise rejection; production |
| `ResonanceEstimator` | Kalman + FFT seed façade | Plug into `AdaptiveShaper` |

### AdaptiveShaper

`AdaptiveShaper` wraps any `BaseShaper` and hot-swaps its impulse parameters
when the `ResonanceEstimator` produces a new estimate:

```python
from src.robotics.shaping     import AdaptiveShaper, ZVDShaper
from src.robotics.ai_estimator import ResonanceEstimator

estimator = ResonanceEstimator(sample_rate_hz=200.0, omega_n_init=18.0)
base      = ZVDShaper(omega_n_rad_s=18.0, zeta=0.10)
shaper    = AdaptiveShaper(base_shaper=base, estimator=estimator)

ctrl = RobotController(shaper=shaper, estimator=estimator)
```

`RobotController` automatically forwards IMU JSON from the firmware telemetry
to `estimator.update(IMUSample)`. The `AdaptiveShaper` queries the estimator
before each trajectory application and updates the impulse sequence if the
estimated ωₙ has shifted.

### Enabling in config.yaml

```yaml
input_shaping:
  adaptive:        true
  estimator_kind:  kalman      # kalman | rls | fft
  imu_sample_rate: 200.0       # Hz — must match firmware IMU output rate
  shaper_kind:     zvd          # base shaper to wrap
  omega_n:         18.0        # initial seed (rad/s)
  zeta:            0.10        # initial seed
```

### Firmware Protocol

The Arduino must include an `"imu"` key in its JSON response:

```json
{
  "status": "ok",
  "angles": [0, 90, 0, 0, 0],
  "imu": {
    "ax": -0.12,
    "ay":  0.03,
    "az":  9.82
  }
}
```

Gyroscope fields `gx`, `gy`, `gz` are optional; only accelerometer data is
required for the EKF.

---

## Full Pipeline Example

```python
from src.robotics.trajectory   import SCurvePlanner
from src.robotics.shaping      import build_shaper
from src.robotics.ai_estimator import ResonanceEstimator
from src.robotics.control      import RobotController
from src.robotics.kinematics   import JointAngles

# Stage 3 — estimator (only if IMU available)
estimator = ResonanceEstimator(sample_rate_hz=200.0, omega_n_init=18.0)

# Stage 2 — adaptive ZVD shaper
shaper = build_shaper(kind="adaptive_zvd", estimator=estimator)

# Stage 1 — S-curve planner
planner = SCurvePlanner(max_vel_deg_s=120, max_acc_deg_s2=200, max_jerk_deg_s3=800)

# Wire into controller
ctrl = RobotController(shaper=shaper, estimator=estimator)
ctrl.connect()

# Plan and execute
home = JointAngles(base=0, shoulder=90, elbow=0, wrist=0, gripper=0)
pick = JointAngles(base=30, shoulder=60, elbow=45, wrist=90, gripper=0)

traj = planner.plan(home, pick)       # Stage 1: smooth trajectory
ctrl.play_trajectory(traj)             # Stage 2/3: shaped internally
```

---

## Comparison Table

| Feature | Trapezoidal | SCurve | ZVD Shaper | Adaptive ZVD |
| --- | --- | --- | --- | --- |
| Reduces servo stress | Partial | ✓ | ✓ | ✓ |
| Zero acceleration at endpoints | ✗ | ✓ | ✓ | ✓ |
| Cancels resonance vibration | ✗ | ✗ | ✓ | ✓ |
| Adapts to changing payload | ✗ | ✗ | ✗ | ✓ |
| Requires IMU hardware | ✗ | ✗ | ✗ | ✓ |
| Extra latency (settling tail) | 0 | 0 | T_d ms | T_d ms |
| Config required | None | max_jerk | omega_n, zeta | imu_sample_rate |

---

## Frequently Asked Questions

**Q: Can I use Stage 2 without Stage 1?**
Yes. Pass any `List[TrajectoryPoint]` to `shaper.apply()`. However, using
SCurvePlanner as Stage 1 reduces the amplitude of oscillation that Stage 2
needs to cancel, making the shaping more effective.

**Q: Can I disable shaping for homing moves?**
Yes: `ctrl.play_trajectory(traj, apply_shaping=False)`.

**Q: What if the estimator hasn't converged yet?**
`AdaptiveShaper` uses the seed values (`omega_n`, `zeta` from config) until the
estimator returns a valid estimate. The Kalman estimator typically needs ~1–2
seconds of IMU data after a move to update.

**Q: My arm doesn't have an IMU. Can I still use Stage 2?**
Yes. Set `adaptive: false` in config, measure ωₙ once with
`python -m scripts.measure_resonance --dry`, and write the value to `config.yaml`.
The ZVD shaper will use these fixed parameters for every move.

**Q: How do I know if the shaper is working?**
Run with `dry_run: true` and inspect the debug logs. The shaper logs the
extended trajectory length (original + settling points). On real hardware,
compare slow-motion video of the tip oscillation before and after enabling
`shaper_kind: zvd`.
