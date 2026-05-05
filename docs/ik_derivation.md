# Inverse Kinematics Derivation

This note documents the analytical IK implemented in
`src/robotics/kinematics.py`.

The arm is modeled as a base yaw joint plus a planar shoulder, elbow, and wrist.
The end effector is the gripper tip.

## Link Definitions

| Symbol | Meaning | Default |
| --- | --- | --- |
| `L1` | Base-to-shoulder vertical offset | 0.105 m |
| `L2` | Upper arm | 0.105 m |
| `L3` | Forearm | 0.090 m |
| `L4` | Wrist-to-gripper tip | 0.060 m |

Joint variables are in degrees in the public API, but the solver uses radians
internally.

| Symbol | Joint |
| --- | --- |
| `theta1` | Base yaw around Z |
| `theta2` | Shoulder pitch |
| `theta3` | Elbow pitch |
| `theta4` | Wrist pitch |

## Geometry

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

The radial distance from base axis to the gripper tip is:

```text
r_total = sqrt(X^2 + Y^2)
```

## Step 1: Base Angle

```text
theta1 = atan2(Y, X)
```

This aligns the planar arm with the target in the XY plane.

## Step 2: Remove Wrist Contribution

The desired wrist pitch is supplied as `wrist_pitch_deg`. Convert it to radians:

```text
phi = radians(wrist_pitch_deg)
```

The wrist center is offset from the gripper tip by `L4`:

```text
r = r_total - L4 * cos(phi)
z = (Z - L1) - L4 * sin(phi)
```

Now `[r, z]` is the target for the 2-link shoulder/elbow chain.

## Step 3: Elbow Angle

Apply the law of cosines:

```text
D = (r^2 + z^2 - L2^2 - L3^2) / (2 * L2 * L3)
```

If `abs(D) > 1`, the wrist center is outside the analytical 2-link workspace and
the analytical solver returns `None`.

For elbow-up:

```text
theta3 = atan2(-sqrt(1 - D^2), D)
```

For elbow-down:

```text
theta3 = atan2(+sqrt(1 - D^2), D)
```

This project allows negative elbow angles in the default joint limits so the
elbow-up analytical solution can round-trip accurately.

## Step 4: Shoulder Angle

```text
theta2 = atan2(z, r) - atan2(L3 * sin(theta3), L2 + L3 * cos(theta3))
```

## Step 5: Wrist Compensation

The wrist completes the requested total pitch:

```text
theta4 = phi - theta2 - theta3
```

Converted back to degrees:

```text
base     = degrees(theta1)
shoulder = degrees(theta2)
elbow    = degrees(theta3)
wrist    = degrees(theta4)
```

Then the result is clamped to configured joint limits.

## Forward Kinematics

The forward model used by the tests and singularity analysis is:

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

## Workspace

For the wrist-center 2-link chain:

```text
max_wrist_center_reach = L2 + L3
min_wrist_center_reach = abs(L2 - L3)
```

For the full gripper tip:

```text
max_tip_reach = L2 + L3 + L4
```

With the defaults:

| Quantity | Value |
| --- | --- |
| Wrist-center max reach | 0.195 m |
| Wrist-center min reach | 0.015 m |
| Tip max reach | 0.255 m |

The runtime also clamps requested XYZ targets to configured workspace bounds
before solving:

```text
workspace_x = [-0.25, 0.25]
workspace_y = [ 0.10, 0.40]
workspace_z = [ 0.00, 0.30]
```

## Numerical Fallback

When the analytical solve fails, `solve()` attempts SLSQP numerical IK.

The objective is:

```text
cost = ||forward(q) - target_xyz||^2
     + 0.01 * (degrees(q_wrist) - wrist_pitch_deg)^2
```

The optimization is bounded by configured joint limits.

## Singularity Check

The solver computes a 3x4 geometric Jacobian and checks its singular values.
The condition number is:

```text
condition = sigma_max / sigma_min
```

If the condition number exceeds the threshold, the solution is treated as
near-singular and a warning is logged.

## Safety Check

`is_safe_delta(current, target)` compares all five joints and warns when any
joint moves more than `MAX_DELTA_DEG_PER_STEP`, currently 45 degrees.

The solver does not reject the returned solution solely because of this warning;
the caller can insert intermediate waypoints or use trajectory planning.
