# Rate-Limited Yaw Steering Design

## Purpose

`RateLimitedYawSteeringProvider` implements a nadir-pointing, Sun-constrained
yaw steering law that behaves like Orekit `YawSteering` when the ideal yaw
motion is mild, but smoothly lags when the ideal yaw motion becomes too
aggressive for configured yaw-rate and yaw-acceleration limits.

The authoritative implementation lives in Java in:

- `nstk/propagation/_java_attitude_providers/com/nstk/attitudes/RateLimitedYawSteeringProvider.java`

NSTK exposes that Java provider through the Python wrapper class:

- `nstk.propagation.attitude_providers.RateLimitedYawSteeringProvider`

## Why This Is Deterministic

This provider is intentionally not implemented as a previous-call mutable
attitude law.

That design would be incorrect in Orekit because propagators can request
attitudes:

- out of time order
- more than once for the same date
- during interpolation and step control

If the provider stored "last date / last yaw state" and advanced from there,
the returned attitude would depend on call order instead of depending only on
the requested date and model configuration.

The provider therefore recomputes the yaw state by integrating a 2-state yaw
ODE from a fixed reference epoch and initial yaw state:

- `psi`: actual yaw angle [rad]
- `omega`: actual yaw rate [rad/s]

This makes the result a deterministic function of:

- reference epoch
- initial yaw state
- requested date
- orbital geometry / Sun geometry
- control gains and yaw limits

## Controller Model

The base attitude law is Orekit `NadirPointing`.

The ideal reference attitude law is Orekit `YawSteering` wrapped around that
base law.

At each internal ODE evaluation time, the provider computes:

- `psi_ref`: ideal yaw angle relative to nadir
- `omega_ref`: ideal yaw rate
- `alpha_ref`: ideal yaw acceleration

The implementation first forms the relative angular coordinates between:

- the base nadir-pointing attitude
- the ideal Orekit `YawSteering` attitude

If that relative motion is consistent with a pure yaw offset about body `+Z`,
the implementation uses the relative `Z` angular-rate and angular-acceleration
components directly.

If it is not sufficiently pure-yaw, the implementation falls back to centered
finite differences of the extracted scalar yaw angle.

The commanded yaw acceleration uses PD tracking with feed-forward reference
acceleration:

```text
e_psi   = wrap(psi_ref - psi)
e_omega = omega_ref - omega
alpha_raw = kp * e_psi + kd * e_omega + alpha_ref
alpha_cmd = clamp(alpha_raw, -maxYawAcceleration, +maxYawAcceleration)
```

If the current yaw rate is already on a configured rate limit and the
acceleration command would drive it farther out, the command is forced to zero.

The yaw ODE is:

```text
psi_dot   = omega
omega_dot = alpha_cmd
```

The implementation uses a projected saturated yaw-rate dynamics formulation:

- `psi_dot` always uses the projected bounded yaw rate
- outward acceleration is suppressed when the current yaw rate is already on
  the configured limit
- returned yaw states clamp the reported yaw rate as a defensive guard against
  numerical overshoot

This avoids depending on an inconsistent hidden overspeed state when the
controller is operating on a saturation boundary.

## Relative Yaw Extraction

The ideal yaw is not computed with ad hoc quaternion algebra.

Instead, at each date:

1. evaluate the base nadir attitude
2. evaluate the ideal Orekit `YawSteering` attitude
3. compute the relative rotation from base to ideal
4. extract the yaw angle about the spacecraft body `+Z` axis

The relative yaw angle is extracted from the relative rotation by mapping the
body `+X` axis into the base-attitude frame and taking its azimuth in the
base-frame XY plane.

NSTK includes tests that explicitly verify:

- positive and negative yaw sign
- reconstruction from extracted yaw for non-identity base attitudes
- zero-offset reconstruction
- ideal-yaw reconstruction consistency against Orekit `YawSteering`

## Attitude Reconstruction

The returned actual attitude is reconstructed by composing:

- the base nadir attitude
- a pure yaw `AngularCoordinates` offset about body `+Z`

Orekit `AngularCoordinates` / `TimeStampedAngularCoordinates` composition is
used directly so the returned rotation, spin, and angular acceleration remain
internally consistent.

NSTK includes a regression test that checks this by comparing the provider's
returned attitude against `shiftedBy(dt)` predictions from its own returned
angular coordinates.

## Why the Provider Can Be Bound to a Separate PV Source

Orekit may call an `AttitudeProvider` with a PV provider that is local to the
current propagation point rather than a global provider intended to be valid
over long time spans.

This matters here because deterministic fixed-epoch yaw integration needs PV
information over the full interval from the reference epoch to the requested
date.

To handle that correctly in NSTK's propagator factories, the Java provider can
be rebound to a separate global PV provider using:

- `withPVProvider(...)`

The propagator factories use this to bind the attitude provider to a dedicated
"PV-only" twin propagator rather than to the same propagator object carrying
the custom attitude law. That avoids recursion and keeps the fixed-epoch
integration model valid when the provider is attached to an Orekit propagator.

## Known Limitations

- The provider can use deterministic checkpoint caching instead of integrating
  all the way from the fixed reference epoch on every query.
- The checkpoint lattice is fixed and anchored at the reference epoch, so the
  evaluation path for a given query depends only on the query time and cache
  settings, not on previous call order.
- The default cache spacing is 1 second, which is aimed at dense orbit
  sampling. For sparse queries, a larger cache step reduces checkpoint buildup.
- When the implementation falls back to finite-difference reference
  derivatives, their quality depends on the configured finite difference step.
  Too large a step smears sharp ideal-yaw features; too small a step can
  amplify numerical noise and increase runtime.

## Recommended Tuning

Good starting points for LEO use cases are typically:

- `finite_difference_step_s`: `0.01` to `0.1`
- `kp`: `0.2` to `2.0`
- `kd`: `1.0` to `6.0`

If you want the provider to stay very close to ideal Orekit `YawSteering` in
non-singular conditions:

- seed the initial yaw state from the ideal reference yaw at the reference
  epoch
- use large yaw-rate and yaw-acceleration limits
- increase `kp` / `kd`

If you want stronger smoothing through aggressive low-beta regions:

- reduce the yaw-rate limit
- reduce the yaw-acceleration limit
- keep `kd` high enough to avoid excessive overshoot
