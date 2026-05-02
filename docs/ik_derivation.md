## IK Solver — Derivation Notes

### 4-DOF Arm (Revolute Joints)

```
          z
          |   θ2
    L1    |  /L2
   [base]─┴─[shoulder]──[elbow θ3]──[wrist θ4]──◉ EE
          θ1 (rotation around Z)
```

### Analytical Solution (Elbow-Up)

Given target **P = [X, Y, Z]** in robot frame:

**Step 1 — Base angle:**
```
θ1 = atan2(Y, X)
r_total = sqrt(X² + Y²)
```

**Step 2 — Remove wrist contribution:**
```
r = r_total − L4·cos(θ4_desired)
z = (Z − L1) − L4·sin(θ4_desired)
```

**Step 3 — 2-R planar IK:**
```
D = (r² + z² − L2² − L3²) / (2·L2·L3)

if |D| > 1 → target unreachable

θ3 = atan2(−sqrt(1−D²), D)     ← elbow-up
θ2 = atan2(z, r) − atan2(L3·sin(θ3), L2 + L3·cos(θ3))
```

**Step 4 — Wrist:**
```
θ4 = θ4_desired − θ2 − θ3
```

### Workspace
- Max reach: `L2 + L3 = 0.195m`
- Min reach: `|L2 − L3| = 0.015m`
- Base rotation: ±90°

### Numerical Fallback
When `|D| > 1`, scipy L-BFGS-B minimises:
```
cost = ||FK(θ) − P_target||² + λ·(θ4 − θ4_desired)²
```
