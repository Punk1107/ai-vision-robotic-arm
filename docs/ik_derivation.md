# Inverse Kinematics Derivation

This note documents the analytical IK implemented in
`src/robotics/kinematics.py`, and the trajectory math used in
`src/robotics/trajectory.py` and `src/robotics/shaping.py`.

The arm is modelled as a base yaw joint plus a planar shoulder, elbow, and
wrist. The end-effector is the gripper tip.

## Part 1 — Inverse Kinematics

### Link Definitions

| Symbol | Meaning | Default |
| --- | --- | --- |
| `L1` | Base-to-shoulder vertical offset | 0.105 m |
| `L2` | Upper arm | 0.105 m |
| `L3` | Forearm | 0.090 m |
| `L4` | Wrist-to-gripper tip | 0.060 m |

Joint variables are in degrees in the public API; the solver uses radians
internally.

| Symbol | Joint |
| --- | --- |
| `theta1` | Base yaw around Z |
| `theta2` | Shoulder pitch |
| `theta3` | Elbow pitch |
| `theta4` | Wrist pitch |

### Geometry

```text
                 z
                 |
                 |       L2          L3          L4
                 |    shoulder     elbow       wrist       tip
                 |       o----------o-----------o-----------x
                 |      /
                 |     /
                 o base

theta1 rotates the planar chain around the Z axis.
theta2, theta3, and theta4 operate in the radial-Z plane.
```

Given target point:

```text
P = [X, Y, Z]
```

The radial distance from the base axis to the gripper tip is:

```text
r_total = sqrt(X² + Y²)
```

### Step 1: Base Angle

```text
theta1 = atan2(Y, X)
```

This aligns the planar arm with the target in the XY plane.

### Step 2: Remove Wrist Contribution

The desired wrist pitch is supplied as `wrist_pitch_deg`. Convert to radians:

```text
phi = radians(wrist_pitch_deg)
```

The wrist centre is offset from the gripper tip by `L4`:

```text
r = r_total - L4 * cos(phi)
z = (Z - L1) - L4 * sin(phi)
```

Now `[r, z]` is the target for the 2-link shoulder/elbow chain.

### Step 3: Elbow Angle

Apply the law of cosines:

```text
D = (r² + z² - L2² - L3²) / (2 * L2 * L3)
```

If `abs(D) > 1`, the wrist centre is outside the analytical 2-link workspace
and the analytical solver returns `None` (numerical fallback is tried).

For elbow-up:

```text
theta3 = atan2(-sqrt(1 - D²), D)
```

For elbow-down:

```text
theta3 = atan2(+sqrt(1 - D²), D)
```

The project allows negative elbow angles in the default joint limits so the
elbow-up solution round-trips accurately.

### Step 4: Shoulder Angle

```text
theta2 = atan2(z, r) - atan2(L3 * sin(theta3), L2 + L3 * cos(theta3))
```

### Step 5: Wrist Compensation

```text
theta4 = phi - theta2 - theta3
```

Converted back to degrees and clamped to configured joint limits.

### Forward Kinematics

```text
r = L2*cos(theta2)
  + L3*cos(theta2 + theta3)
  + L4*cos(theta2 + theta3 + theta4)

z = L1
  + L2*sin(theta2)
  + L3*sin(theta2 + theta3)
  + L4*sin(theta2 + theta3 + theta4)

X = r*cos(theta1)
Y = r*sin(theta1)
Z = z
```

### Workspace

For the wrist-centre 2-link chain:

```text
max_wrist_center_reach = L2 + L3 = 0.195 m
min_wrist_center_reach = abs(L2 - L3) = 0.015 m
```

For the full gripper tip:

```text
max_tip_reach = L2 + L3 + L4 = 0.255 m
```

The runtime also clamps XYZ targets to configured workspace bounds:

```text
workspace_x = [-0.25, 0.25]
workspace_y = [ 0.10, 0.40]
workspace_z = [ 0.00, 0.30]
```

### Numerical Fallback (SLSQP)

When the analytical solve fails or returns `None`, `solve()` attempts SLSQP
numerical IK. The objective is:

```text
cost = ||forward(q) - target_xyz||²
     + 0.01 * (degrees(q_wrist) - wrist_pitch_deg)²
```

Bounded by configured joint limits with up to 300 iterations.

### Singularity Check

The solver computes a 3×4 geometric Jacobian and inspects its singular values.
The condition number is:

```text
condition = sigma_max / sigma_min
```

If the condition number exceeds the threshold (default 80), a warning is logged.

### Safety Check

`is_safe_delta(current, target)` compares all five joints and warns when any
joint moves more than `MAX_DELTA_DEG_PER_STEP` (default 45°). Large deltas are
flagged so the caller can insert intermediate waypoints or route through the
trajectory planner.

---

## Part 2 — Trajectory Generation

### Normalised Motion Profiles

All planners in `trajectory.py` use a normalised time parameter τ = t / T ∈ [0,1].

#### Cubic ease-in/out (C1)

```text
s(τ)   = 3τ² − 2τ³
s'(τ)  = 6τ  − 6τ²          ← velocity profile
s''(τ) = 6   − 12τ           ← acceleration profile
```

Zero velocity at endpoints; acceleration has a discontinuity at τ=0 and τ=1.

#### Quintic polynomial (C2)

```text
s(τ)   = 10τ³ − 15τ⁴ + 6τ⁵
s'(τ)  = 30τ² − 60τ³ + 30τ⁴
s''(τ) = 60τ  − 180τ² + 120τ³
```

Zero velocity **and** zero acceleration at both endpoints — C2 continuity.
This is the default profile used by `SCurvePlanner` and `JerkLimitedPlanner`.

#### Sigmoid (C∞)

```text
sig(τ) = 1 / (1 + exp(-k(τ - 0.5)))     k = 12 (default)
s(τ)   = (sig(τ) - sig(0)) / (sig(1) - sig(0))    ← normalised to [0, 1]
```

Infinitely differentiable; slightly longer settling time at endpoints than quintic.

### S-Curve Timing

`SCurvePlanner` determines the total move duration T by computing the
trapezoidal timing (accel + cruise + decel) and then applying a quintic
scaling over [0, T]. For a displacement d:

```text
t_acc    = v_max / a_max                   ← time to ramp to full velocity
d_acc    = 0.5 * a_max * t_acc²            ← distance covered in accel phase

if d ≤ 2 * d_acc:                          ← triangular profile (no cruise)
    T ≈ 2 * sqrt(d / a_max) + 2 * t_j

else:
    t_cruise = (d - 2 * d_acc) / v_max
    T = 4 * t_j + 2 * t_acc + t_cruise    ← t_j = jerk duration = a_max / j_max
```

All joints are time-synchronised: the joint with the largest displacement
determines T; all others are scaled to match.

---

## Part 3 — Input Shaping Math

### Second-Order System Model

The arm joint (with load) is modelled as:

```text
ẍ + 2ζωₙẋ + ωₙ²x = u(t)
```

where:
- ωₙ = natural frequency (rad/s)
- ζ  = damping ratio (0 < ζ < 1)

Input shaping convolves `u(t)` with a sequence of impulses to cancel residual
oscillation after the move ends.

### ZV Shaper (2 impulses)

Reference: Singer & Seering, ASME JDSMC, 1990.

```text
K  = exp(-πζ / sqrt(1 - ζ²))

A₁ = 1 / (1 + K)     at t = 0
A₂ = K / (1 + K)     at t = T_d / 2

T_d = 2π / ωₙ√(1 - ζ²)    ← damped period
```

Impulse sum: A₁ + A₂ = 1.

### ZVD Shaper (3 impulses)

```text
A₁ = 1     / (1 + K)²    at t = 0
A₂ = 2K    / (1 + K)²    at t = T_d / 2
A₃ = K²    / (1 + K)²    at t = T_d
```

Zeroes both V(ωₙ) and dV/dωₙ, providing first-order robustness to frequency
uncertainty. Recommended for real arms where ωₙ is only approximately known.

### EI Shaper (3 impulses)

Reference: Singhose et al., AIAA Journal of Guidance, 1996.

```text
A₁ = (1 + V_tol) / 4     at t = 0
A₂ = (1 - V_tol) / 2     at t = T_d / 2
A₃ = (1 + V_tol) / 4     at t = T_d
```

V_tol (default 0.05) is the allowed vibration fraction at ωₙ ± 10%.
Trades some on-model performance for a wider robust frequency band.

### Trajectory Convolution

`BaseShaper.apply(trajectory)` implements:

```text
u_shaped(t) = Σᵢ  Aᵢ · u(t - tᵢ)
```

For a discrete trajectory sampled at dt = 1/hz:

```text
shift_i  = round(tᵢ / dt)           ← sample shift for impulse i
output   = Σᵢ  Aᵢ · shift(input, shift_i)
```

The output is extended by `max(tᵢ)` samples to accommodate the delayed copies.
All joint channels are convolved independently.

---

## Part 4 — AI Resonance Estimation (Stage 3)

### State-Space Model

For the EKF, the state vector is:

```text
x = [pos, vel, ωₙ², 2ζωₙ]ᵀ    (4 states)
```

State transition (Euler, per IMU sample at dt):

```text
pos_new = pos + vel · dt
vel_new = vel + (-ωₙ² · pos - 2ζωₙ · vel) · dt
ωₙ²_new = ωₙ²                    ← slow-varying parameter
2ζωₙ_new = 2ζωₙ                  ← slow-varying parameter
```

Measurement model:

```text
z = -ωₙ² · pos - 2ζωₙ · vel   + noise
```

The EKF linearises the nonlinear state transition via the Jacobian F = ∂f/∂x
and applies the standard predict → update cycle at each IMU sample.

### Extracting ωₙ and ζ

From the estimated state:

```text
ωₙ = sqrt(max(0, x[2]))
ζ  = x[3] / (2 · ωₙ)        clamped to [0.001, 0.99]
```

### RLS Estimator

The Recursive Least Squares estimator solves:

```text
ẍ = θᵀ φ

θ = [-ωₙ², -2ζωₙ]ᵀ          ← parameter vector
φ = [pos, vel]ᵀ              ← regressor (integrated from IMU)
```

Update equations:

```text
K     = P · φ / (λ + φᵀ · P · φ)
θ_new = θ + K · (ẍ - φᵀ · θ)
P_new = (P - K · φᵀ · P) / λ
```

λ (forgetting factor, default 0.98) allows slow parameter drift tracking.
