package com.nstk.attitudes;

import java.util.NavigableMap;
import java.util.TreeMap;
import java.util.stream.Stream;

import org.hipparchus.CalculusFieldElement;
import org.hipparchus.Field;
import org.hipparchus.geometry.euclidean.threed.Rotation;
import org.hipparchus.geometry.euclidean.threed.RotationConvention;
import org.hipparchus.geometry.euclidean.threed.Vector3D;
import org.hipparchus.ode.ODEIntegrator;
import org.hipparchus.ode.ODEState;
import org.hipparchus.ode.ODEStateAndDerivative;
import org.hipparchus.ode.OrdinaryDifferentialEquation;
import org.hipparchus.ode.nonstiff.DormandPrince853Integrator;
import org.hipparchus.ode.sampling.ODEStateInterpolator;
import org.hipparchus.ode.sampling.ODEStepHandler;
import org.hipparchus.util.FastMath;
import org.hipparchus.util.MathUtils;
import org.orekit.attitudes.Attitude;
import org.orekit.attitudes.AttitudeProvider;
import org.orekit.attitudes.FieldAttitude;
import org.orekit.attitudes.FrameAlignedProvider;
import org.orekit.attitudes.GroundPointing;
import org.orekit.attitudes.NadirPointing;
import org.orekit.attitudes.YawSteering;
import org.orekit.bodies.BodyShape;
import org.orekit.frames.Frame;
import org.orekit.propagation.Propagator;
import org.orekit.propagation.events.EventDetector;
import org.orekit.propagation.events.FieldEventDetector;
import org.orekit.time.AbsoluteDate;
import org.orekit.time.FieldAbsoluteDate;
import org.orekit.utils.AngularCoordinates;
import org.orekit.utils.ExtendedPositionProvider;
import org.orekit.utils.FieldPVCoordinatesProvider;
import org.orekit.utils.PVCoordinatesProvider;
import org.orekit.utils.TimeStampedAngularCoordinates;
import org.orekit.utils.TimeStampedFieldPVCoordinates;
import org.orekit.utils.TimeStampedPVCoordinates;

/**
 * Orekit attitude provider for nadir-pointing yaw steering with rate and acceleration limits.
 *
 * <p>This provider uses a standard {@link NadirPointing} law as the base attitude and an
 * Orekit {@link YawSteering} law as the ideal reference attitude. The actual commanded yaw is
 * obtained by integrating a deterministic two-state yaw-axis ODE from a fixed reference epoch and
 * initial yaw state:
 *
 * <pre>
 * state = [psi, omega]
 * psi_dot   = omega
 * omega_dot = alpha_cmd
 * </pre>
 *
 * <p>The commanded yaw acceleration uses PD tracking with feed-forward reference acceleration:
 *
 * <pre>
 * e_psi   = wrap(psi_ref - psi)
 * e_omega = omega_ref - omega
 * alpha_raw = kp * e_psi + kd * e_omega + alpha_ref
 * alpha_cmd = clamp(alpha_raw, -maxYawAcceleration, +maxYawAcceleration)
 * </pre>
 *
 * <p>Yaw is defined as the relative rotation from the base nadir-pointing attitude into the actual
 * body attitude about the spacecraft body +Z axis. All rotations handled here use Orekit's usual
 * convention of mapping the reference frame into the body frame. The output attitude is therefore
 * reconstructed by composing:
 *
 * <ol>
 *   <li>the base nadir attitude, expressed as a reference-to-body rotation, and</li>
 *   <li>a yaw-only body-fixed offset about +Z, applied on top of that base attitude.</li>
 * </ol>
 *
 * <p>If the current yaw rate already lies on a configured limit and the commanded acceleration
 * would push the yaw rate farther outside that bound, the acceleration command is forced to zero.
 *
 * <p>The provider is intentionally <strong>not</strong> previous-call stateful. Orekit numerical
 * propagators may request attitudes out of chronological order, so correctness requires the
 * attitude to depend only on:
 *
 * <ul>
 *   <li>the requested date,</li>
 *   <li>the reference epoch and initial yaw state,</li>
 *   <li>the orbital state provider and geometry, and</li>
 *   <li>the controller and limit settings.</li>
 * </ul>
 *
 * <p>Reference yaw is extracted from the relative rotation between the base nadir attitude and the
 * ideal Orekit {@link YawSteering} attitude. Reference yaw rate and yaw acceleration are taken
 * from the relative angular coordinates only when that relative motion is consistent with a pure
 * yaw offset about body +Z. Otherwise the provider falls back to centered finite differences of
 * the extracted scalar yaw angle. The output attitude is reconstructed by composing the base
 * attitude with a yaw-only angular offset about the spacecraft body +Z axis using Orekit
 * {@link AngularCoordinates} composition.
 */
public class RateLimitedYawSteeringProvider implements AttitudeProvider {

    private static final double DEFAULT_MIN_STEP = 1.0e-8;
    private static final double DEFAULT_MAX_STEP = 600.0;
    private static final double[] DEFAULT_ABS_TOL = {1.0e-7, 1.0e-8};
    private static final double[] DEFAULT_REL_TOL = {1.0e-6, 1.0e-6};
    private static final boolean DEFAULT_CACHE_ENABLED = true;
    private static final double DEFAULT_CACHE_STEP = 1.0;
    private static final double EPS_RATE_LIMIT = 1.0e-12;
    private static final double EPS_AXIS_NORM = 1.0e-15;
    private static final double EPS_TIME = 1.0e-14;
    private static final double EPS_PURE_YAW_ROTATION = 1.0e-10;
    private static final double EPS_PURE_YAW_COMPONENT = 1.0e-10;

    private final Frame inertialFrame;
    private final BodyShape bodyShape;
    private final ExtendedPositionProvider sunProvider;
    private final Vector3D phasingAxis;
    private final double maxYawRate;
    private final double maxYawAcceleration;
    private final double kp;
    private final double kd;
    private final AbsoluteDate referenceEpoch;
    private final double psi0;
    private final double omega0;
    private final double finiteDifferenceStep;
    private final PVCoordinatesProvider trajectoryProvider;
    private final boolean cacheEnabled;
    private final double cacheStep;
    private final GroundPointing baseLaw;
    private final YawSteering idealLaw;
    private final Object checkpointLock;
    private final NavigableMap<Long, YawStateSnapshot> checkpointCache;

    /**
     * Create a deterministic rate-limited yaw-steering provider.
     *
     * @param inertialFrame pseudo-inertial frame used by the nadir and yaw-steering laws
     * @param bodyShape central-body shape used by the nadir-pointing base law
     * @param sunProvider Sun ephemeris provider used by Orekit yaw steering
     * @param phasingAxis spacecraft body-fixed phasing axis used by Orekit yaw steering
     * @param maxYawRate maximum allowed yaw-rate magnitude in radians per second
     * @param maxYawAcceleration maximum allowed yaw-acceleration magnitude in radians per second squared
     * @param kp proportional gain for yaw-angle tracking
     * @param kd derivative gain for yaw-rate tracking
     * @param referenceEpoch fixed integration reference epoch
     * @param psi0 actual yaw angle at {@code referenceEpoch} in radians
     * @param omega0 actual yaw rate at {@code referenceEpoch} in radians per second
     * @param finiteDifferenceStep centered finite-difference step in seconds for reference derivatives
     */
    public RateLimitedYawSteeringProvider(
            final Frame inertialFrame,
            final BodyShape bodyShape,
            final ExtendedPositionProvider sunProvider,
            final Vector3D phasingAxis,
            final double maxYawRate,
            final double maxYawAcceleration,
            final double kp,
            final double kd,
            final AbsoluteDate referenceEpoch,
            final double psi0,
            final double omega0,
            final double finiteDifferenceStep) {
        this(
                inertialFrame,
                bodyShape,
                sunProvider,
                phasingAxis,
                maxYawRate,
                maxYawAcceleration,
                kp,
                kd,
                referenceEpoch,
                psi0,
                omega0,
                finiteDifferenceStep,
                DEFAULT_CACHE_ENABLED,
                DEFAULT_CACHE_STEP);
    }

    /**
     * Create a deterministic rate-limited yaw-steering provider with deterministic checkpoint caching.
     *
     * @param inertialFrame pseudo-inertial frame used by the nadir and yaw-steering laws
     * @param bodyShape central-body shape used by the nadir-pointing base law
     * @param sunProvider Sun ephemeris provider used by Orekit yaw steering
     * @param phasingAxis spacecraft body-fixed phasing axis used by Orekit yaw steering
     * @param maxYawRate maximum allowed yaw-rate magnitude in radians per second
     * @param maxYawAcceleration maximum allowed yaw-acceleration magnitude in radians per second squared
     * @param kp proportional gain for yaw-angle tracking
     * @param kd derivative gain for yaw-rate tracking
     * @param referenceEpoch fixed integration reference epoch
     * @param psi0 actual yaw angle at {@code referenceEpoch} in radians
     * @param omega0 actual yaw rate at {@code referenceEpoch} in radians per second
     * @param finiteDifferenceStep centered finite-difference step in seconds for reference derivatives
     * @param cacheEnabled whether deterministic fixed-grid checkpoint caching is enabled
     * @param cacheStep checkpoint spacing in seconds when caching is enabled
     */
    public RateLimitedYawSteeringProvider(
            final Frame inertialFrame,
            final BodyShape bodyShape,
            final ExtendedPositionProvider sunProvider,
            final Vector3D phasingAxis,
            final double maxYawRate,
            final double maxYawAcceleration,
            final double kp,
            final double kd,
            final AbsoluteDate referenceEpoch,
            final double psi0,
            final double omega0,
            final double finiteDifferenceStep,
            final boolean cacheEnabled,
            final double cacheStep) {
        this(
                inertialFrame,
                bodyShape,
                sunProvider,
                phasingAxis,
                maxYawRate,
                maxYawAcceleration,
                kp,
                kd,
                referenceEpoch,
                psi0,
                omega0,
                finiteDifferenceStep,
                cacheEnabled,
                cacheStep,
                null);
    }

    private RateLimitedYawSteeringProvider(
            final Frame inertialFrame,
            final BodyShape bodyShape,
            final ExtendedPositionProvider sunProvider,
            final Vector3D phasingAxis,
            final double maxYawRate,
            final double maxYawAcceleration,
            final double kp,
            final double kd,
            final AbsoluteDate referenceEpoch,
            final double psi0,
            final double omega0,
            final double finiteDifferenceStep,
            final boolean cacheEnabled,
            final double cacheStep,
            final PVCoordinatesProvider trajectoryProvider) {

        if (inertialFrame == null) {
            throw new IllegalArgumentException("inertialFrame must not be null");
        }
        if (!inertialFrame.isPseudoInertial()) {
            throw new IllegalArgumentException("inertialFrame must be pseudo-inertial");
        }
        if (bodyShape == null) {
            throw new IllegalArgumentException("bodyShape must not be null");
        }
        if (sunProvider == null) {
            throw new IllegalArgumentException("sunProvider must not be null");
        }
        if (phasingAxis == null) {
            throw new IllegalArgumentException("phasingAxis must be non-zero");
        }
        if (phasingAxis.getNorm() <= EPS_AXIS_NORM) {
            throw new IllegalArgumentException("phasingAxis must be non-zero");
        }
        final Vector3D normalizedPhasingAxis = phasingAxis.normalize();
        if (FastMath.hypot(normalizedPhasingAxis.getX(), normalizedPhasingAxis.getY()) <= EPS_AXIS_NORM) {
            throw new IllegalArgumentException(
                    "phasingAxis must not be parallel to spacecraft +/-Z for YawSteering");
        }
        if (!Double.isFinite(maxYawRate) || maxYawRate < 0.0) {
            throw new IllegalArgumentException("maxYawRate must be finite and >= 0");
        }
        if (!Double.isFinite(maxYawAcceleration) || maxYawAcceleration < 0.0) {
            throw new IllegalArgumentException("maxYawAcceleration must be finite and >= 0");
        }
        if (!Double.isFinite(kp) || kp < 0.0) {
            throw new IllegalArgumentException("kp must be finite and >= 0");
        }
        if (!Double.isFinite(kd) || kd < 0.0) {
            throw new IllegalArgumentException("kd must be finite and >= 0");
        }
        if (referenceEpoch == null) {
            throw new IllegalArgumentException("referenceEpoch must not be null");
        }
        if (!Double.isFinite(psi0)) {
            throw new IllegalArgumentException("psi0 must be finite");
        }
        if (!Double.isFinite(omega0)) {
            throw new IllegalArgumentException("omega0 must be finite");
        }
        if (FastMath.abs(omega0) > maxYawRate + EPS_RATE_LIMIT) {
            throw new IllegalArgumentException("omega0 magnitude must not exceed maxYawRate");
        }
        if (!Double.isFinite(finiteDifferenceStep) || finiteDifferenceStep <= 0.0) {
            throw new IllegalArgumentException("finiteDifferenceStep must be finite and > 0");
        }
        if (!Double.isFinite(cacheStep) || cacheStep <= 0.0) {
            throw new IllegalArgumentException("cacheStep must be finite and > 0");
        }

        this.inertialFrame = inertialFrame;
        this.bodyShape = bodyShape;
        this.sunProvider = sunProvider;
        this.phasingAxis = normalizedPhasingAxis;
        this.maxYawRate = maxYawRate;
        this.maxYawAcceleration = maxYawAcceleration;
        this.kp = kp;
        this.kd = kd;
        this.referenceEpoch = referenceEpoch;
        this.psi0 = psi0;
        this.omega0 = omega0;
        this.finiteDifferenceStep = finiteDifferenceStep;
        this.baseLaw = new NadirPointing(inertialFrame, bodyShape);
        this.idealLaw = new YawSteering(inertialFrame, baseLaw, sunProvider, this.phasingAxis);
        this.trajectoryProvider = sanitizeTrajectoryProvider(trajectoryProvider);
        this.cacheEnabled = cacheEnabled;
        this.cacheStep = cacheStep;
        this.checkpointLock = new Object();
        this.checkpointCache = new TreeMap<>();
    }

    /**
     * Return a copy of this provider bound to a specific global PV provider.
     *
     * <p>When this attitude provider is attached to an Orekit propagator, Orekit may call it with
     * a local PV provider around the current date. Fixed-epoch deterministic yaw integration,
     * however, requires a PV provider that is valid over the full integration interval from the
     * reference epoch to the requested date. Binding the provider to a known global PV provider
     * ensures that internal reference-yaw evaluation uses that global provider rather than the
     * per-call local one.
     *
     * @param pvProvider global PV provider used internally for yaw integration
     * @return a new provider instance identical to this one but bound to {@code pvProvider}
     */
    public RateLimitedYawSteeringProvider withPVProvider(final PVCoordinatesProvider pvProvider) {
        return new RateLimitedYawSteeringProvider(
                inertialFrame,
                bodyShape,
                sunProvider,
                phasingAxis,
                maxYawRate,
                maxYawAcceleration,
                kp,
                kd,
                referenceEpoch,
                psi0,
                omega0,
                finiteDifferenceStep,
                cacheEnabled,
                cacheStep,
                pvProvider);
    }

    /**
     * Extract the relative yaw angle from a base attitude to a target attitude.
     *
     * <p>Both rotations must represent the same convention: a rotation from a common reference
     * frame into the spacecraft body frame. The returned value is the yaw angle in radians for the
     * relative body-fixed +Z rotation that maps the base attitude to the target attitude.
     *
     * @param baseReferenceToBody rotation from reference frame to body frame for the base attitude
     * @param targetReferenceToBody rotation from reference frame to body frame for the target attitude
     * @return relative yaw angle in radians in the interval {@code [-pi, +pi]}
     */
    public static double extractRelativeYaw(
            final Rotation baseReferenceToBody,
            final Rotation targetReferenceToBody) {
        final Rotation relative = targetReferenceToBody.applyTo(baseReferenceToBody.revert());
        return extractYawFromRelativeRotation(relative);
    }

    /**
     * Return the internally tracked yaw state for the requested date.
     *
     * @param pvProv orbit/PV provider used by the underlying Orekit attitude laws
     * @param date requested date
     * @param frame reference frame used for evaluating the base and ideal attitudes
     * @return tracked yaw angle, rate, and acceleration at {@code date}
     */
    public YawStateSnapshot getTrackedYawState(
            final PVCoordinatesProvider pvProv,
            final AbsoluteDate date,
            final Frame frame) {
        return integrateYawState(pvProv, date, frame == null ? inertialFrame : frame);
    }

    /**
     * Return the ideal reference yaw state derived from Orekit {@link YawSteering}.
     *
     * @param pvProv orbit/PV provider used by the underlying Orekit attitude laws
     * @param date requested date
     * @param frame reference frame used for evaluating the base and ideal attitudes
     * @return ideal yaw angle, rate, and acceleration at {@code date}
     */
    public YawStateSnapshot getReferenceYawState(
            final PVCoordinatesProvider pvProv,
            final AbsoluteDate date,
            final Frame frame) {
        return computeReferenceYawState(pvProv, date, frame == null ? inertialFrame : frame);
    }

    /**
     * Sample body angular-rate or angular-acceleration vectors for a batch of query times.
     *
     * <p>This bulk path is intended for vectorized NSTK sampling. It integrates the internal yaw
     * dynamics deterministically across the requested time set without relying on previous call
     * order or forcing the generic checkpoint cache to materialize an entire dense lattice up to
     * the farthest requested time.
     *
     * @param pvProv orbit/PV provider used by the underlying Orekit attitude laws
     * @param queryEpoch epoch relative to which {@code dtSeconds} are expressed
     * @param dtSeconds requested query times in seconds relative to {@code queryEpoch}
     * @param frame reference frame used for evaluating the base and ideal attitudes
     * @param rotationAcceleration if {@code true}, return angular-acceleration vectors;
     *     otherwise return angular-rate vectors
     * @return packed XYZ vectors of length {@code 3 * dtSeconds.length}
     */
    public double[] sampleBodyAngularVectors(
            final PVCoordinatesProvider pvProv,
            final AbsoluteDate queryEpoch,
            final double[] dtSeconds,
            final Frame frame,
            final boolean rotationAcceleration) {

        final int n = dtSeconds == null ? 0 : dtSeconds.length;
        final double[] out = new double[3 * n];
        if (n == 0) {
            return out;
        }

        final Frame evaluationFrame = frame == null ? inertialFrame : frame;
        final PVCoordinatesProvider effectivePvProvider = selectPvProvider(pvProv);
        final double epochOffset = queryEpoch.durationFrom(referenceEpoch);
        final double[] targetTimes = new double[n];

        for (int i = 0; i < n; i++) {
            if (!Double.isFinite(dtSeconds[i])) {
                throw new IllegalArgumentException("query times must be finite");
            }
            targetTimes[i] = epochOffset + dtSeconds[i];
        }

        final YawStateSnapshot[] yawStates = sampleYawStateBatch(effectivePvProvider, evaluationFrame, targetTimes);
        for (int i = 0; i < n; i++) {
            final AbsoluteDate date = queryEpoch.shiftedBy(dtSeconds[i]);
            final Attitude baseAttitude = baseLaw.getAttitude(effectivePvProvider, date, evaluationFrame);
            final TimeStampedAngularCoordinates actualOrientation =
                    buildActualOrientation(date, baseAttitude, yawStates[i]);
            final Vector3D vec =
                    rotationAcceleration
                            ? actualOrientation.getRotationAcceleration()
                            : actualOrientation.getRotationRate();
            copyVectorOrZero(vec, out, 3 * i);
        }

        return out;
    }

    @Override
    public Attitude getAttitude(
            final PVCoordinatesProvider pvProv,
            final AbsoluteDate date,
            final Frame frame) {

        final Frame evaluationFrame = frame == null ? inertialFrame : frame;
        final PVCoordinatesProvider effectivePvProvider = selectPvProvider(pvProv);
        final Attitude baseAttitude = baseLaw.getAttitude(effectivePvProvider, date, evaluationFrame);
        final YawStateSnapshot yawState = integrateYawState(effectivePvProvider, date, evaluationFrame);
        final TimeStampedAngularCoordinates actualOrientation =
                buildActualOrientation(date, baseAttitude, yawState);
        return new Attitude(evaluationFrame, actualOrientation);
    }

    @Override
    public <T extends CalculusFieldElement<T>> FieldAttitude<T> getAttitude(
            final FieldPVCoordinatesProvider<T> pvProv,
            final FieldAbsoluteDate<T> date,
            final Frame frame) {

        // The field overload intentionally reuses the deterministic regular-date implementation.
        // This keeps one authoritative yaw-integration path and avoids duplicating the controller
        // logic in field arithmetic. The field PV provider is bridged to a regular PV provider and
        // the resulting regular Attitude is then wrapped back into a FieldAttitude.
        final FieldToRegularPVProvider<T> regularProvider =
                new FieldToRegularPVProvider<>(pvProv, date.getField());
        final Attitude attitude = getAttitude(selectPvProvider(regularProvider), date.toAbsoluteDate(), frame);
        return new FieldAttitude<>(date.getField(), attitude);
    }

    @Override
    public Stream<EventDetector> getEventDetectors() {
        return idealLaw.getEventDetectors();
    }

    @Override
    public <T extends CalculusFieldElement<T>> Stream<FieldEventDetector<T>> getFieldEventDetectors(
            final Field<T> field) {
        return idealLaw.getFieldEventDetectors(field);
    }

    private YawStateSnapshot integrateYawState(
            final PVCoordinatesProvider pvProv,
            final AbsoluteDate date,
            final Frame frame) {

        final double targetTime = date.durationFrom(referenceEpoch);
        if (!canUseCheckpointCache(frame)) {
            return integrateBetween(
                    pvProv,
                    frame,
                    0.0,
                    new YawStateSnapshot(psi0, clampYawRate(omega0), Double.NaN),
                    targetTime);
        }

        final long checkpointIndex = selectCheckpointIndex(targetTime);
        final YawStateSnapshot checkpointState = getOrCreateCheckpoint(pvProv, frame, checkpointIndex);
        final double checkpointTime = checkpointIndex * cacheStep;
        return integrateBetween(pvProv, frame, checkpointTime, checkpointState, targetTime);
    }

    private PVCoordinatesProvider selectPvProvider(final PVCoordinatesProvider callTimeProvider) {
        return trajectoryProvider != null ? trajectoryProvider : callTimeProvider;
    }

    private PVCoordinatesProvider sanitizeTrajectoryProvider(final PVCoordinatesProvider provider) {
        if (provider instanceof Propagator) {
            return new NonRecursivePropagatorPVProvider((Propagator) provider, inertialFrame);
        }
        return provider;
    }

    private boolean canUseCheckpointCache(final Frame frame) {
        return cacheEnabled && trajectoryProvider != null && inertialFrame.equals(frame);
    }

    private long selectCheckpointIndex(final double targetTime) {
        if (FastMath.abs(targetTime) <= EPS_TIME) {
            return 0L;
        }
        if (targetTime > 0.0) {
            return (long) FastMath.floor(targetTime / cacheStep);
        }
        return (long) FastMath.ceil(targetTime / cacheStep);
    }

    private YawStateSnapshot getOrCreateCheckpoint(
            final PVCoordinatesProvider pvProv,
            final Frame frame,
            final long targetIndex) {

        synchronized (checkpointLock) {
            YawStateSnapshot existing = checkpointCache.get(targetIndex);
            if (existing != null) {
                return existing;
            }

            ensureReferenceCheckpoint(pvProv, frame);

            if (targetIndex > 0L) {
                long currentIndex = checkpointCache.floorKey(targetIndex);
                YawStateSnapshot currentState = checkpointCache.get(currentIndex);
                if (currentIndex < targetIndex) {
                    populateCheckpointRange(pvProv, frame, currentIndex, currentState, targetIndex);
                }
                return checkpointCache.get(targetIndex);
            }

            long currentIndex = checkpointCache.ceilingKey(targetIndex);
            if (currentIndex > 0L) {
                currentIndex = 0L;
            }
            YawStateSnapshot currentState = checkpointCache.get(currentIndex);
            if (currentIndex > targetIndex) {
                populateCheckpointRange(pvProv, frame, currentIndex, currentState, targetIndex);
            }
            return checkpointCache.get(targetIndex);
        }
    }

    private void ensureReferenceCheckpoint(final PVCoordinatesProvider pvProv, final Frame frame) {
        if (!checkpointCache.containsKey(0L)) {
            final double initialOmega = clampYawRate(omega0);
            checkpointCache.put(0L, new YawStateSnapshot(psi0, initialOmega, Double.NaN));
        }
    }

    private YawStateSnapshot integrateBetween(
            final PVCoordinatesProvider pvProv,
            final Frame frame,
            final double startTime,
            final YawStateSnapshot startState,
            final double targetTime) {

        if (FastMath.abs(targetTime - startTime) <= EPS_TIME) {
            return ensureYawAcceleration(pvProv, frame, targetTime, startState);
        }

        final YawDynamics dynamics = new YawDynamics(pvProv, frame);
        final ODEIntegrator integrator =
                new DormandPrince853Integrator(
                        DEFAULT_MIN_STEP,
                        FastMath.max(DEFAULT_MAX_STEP, finiteDifferenceStep),
                        DEFAULT_ABS_TOL,
                        DEFAULT_REL_TOL);

        final ODEState initialState = new ODEState(
                startTime,
                new double[] {startState.psi, clampYawRate(startState.omega)});
        final ODEStateAndDerivative finalState = integrator.integrate(dynamics, initialState, targetTime);

        final double psi = finalState.getPrimaryState()[0];
        final double omega = clampYawRate(finalState.getPrimaryState()[1]);
        final double alpha = finalState.getPrimaryDerivative()[1];
        return new YawStateSnapshot(psi, omega, alpha);
    }

    private YawStateSnapshot computeReferenceYawState(
            final PVCoordinatesProvider pvProv,
            final AbsoluteDate date,
            final Frame frame) {

        final Attitude baseAttitude = baseLaw.getAttitude(pvProv, date, frame);
        final Attitude idealAttitude = idealLaw.getAttitude(pvProv, date, frame);
        final AngularCoordinates relative =
                idealAttitude.getOrientation().subtractOffset(baseAttitude.getOrientation());

        final double psi = extractYawFromRelativeRotation(relative.getRotation());
        final Vector3D rotationRate = relative.getRotationRate();
        final Vector3D rotationAcceleration = relative.getRotationAcceleration();

        if (rotationRate == null || rotationAcceleration == null) {
            return computeReferenceYawStateFiniteDifference(pvProv, date, frame);
        }
        if (!isPureYawRelativeState(relative, psi, rotationRate, rotationAcceleration)) {
            return computeReferenceYawStateFiniteDifference(pvProv, date, frame);
        }

        final double omega = rotationRate.getZ();
        final double alpha = rotationAcceleration.getZ();
        if (!Double.isFinite(omega) || !Double.isFinite(alpha)) {
            return computeReferenceYawStateFiniteDifference(pvProv, date, frame);
        }

        return new YawStateSnapshot(psi, omega, alpha);
    }

    private boolean isPureYawRelativeState(
            final AngularCoordinates relative,
            final double psi,
            final Vector3D rotationRate,
            final Vector3D rotationAcceleration) {

        final Rotation expectedYaw =
                new Rotation(Vector3D.PLUS_K, psi, RotationConvention.FRAME_TRANSFORM);
        final double rotationError = rotationDistance(relative.getRotation(), expectedYaw);
        if (rotationError > EPS_PURE_YAW_ROTATION) {
            return false;
        }

        final double rateLateral = FastMath.hypot(rotationRate.getX(), rotationRate.getY());
        final double accelLateral =
                FastMath.hypot(rotationAcceleration.getX(), rotationAcceleration.getY());
        final double rateScale = FastMath.max(1.0, rotationRate.getNorm());
        final double accelScale = FastMath.max(1.0, rotationAcceleration.getNorm());
        return rateLateral <= EPS_PURE_YAW_COMPONENT * rateScale
                && accelLateral <= EPS_PURE_YAW_COMPONENT * accelScale;
    }

    private YawStateSnapshot[] sampleYawStateBatch(
            final PVCoordinatesProvider pvProv,
            final Frame frame,
            final double[] targetTimes) {

        final int n = targetTimes.length;
        final YawStateSnapshot[] out = new YawStateSnapshot[n];
        final long[] order = sortIndicesByTime(targetTimes);
        final YawStateSnapshot referenceState = ensureYawAcceleration(
                pvProv,
                frame,
                0.0,
                new YawStateSnapshot(psi0, clampYawRate(omega0), Double.NaN));
        int firstNonNegative = 0;
        while (firstNonNegative < n && targetTimes[(int) order[firstNonNegative]] < -EPS_TIME) {
            firstNonNegative++;
        }

        integrateSortedQuerySegment(
                pvProv,
                frame,
                targetTimes,
                order,
                firstNonNegative - 1,
                -1,
                -1,
                referenceState,
                out);
        integrateSortedQuerySegment(
                pvProv,
                frame,
                targetTimes,
                order,
                firstNonNegative,
                n,
                1,
                referenceState,
                out);

        return out;
    }

    private void integrateSortedQuerySegment(
            final PVCoordinatesProvider pvProv,
            final Frame frame,
            final double[] targetTimes,
            final long[] order,
            final int startIndex,
            final int stopIndexExclusive,
            final int indexStep,
            final YawStateSnapshot referenceState,
            final YawStateSnapshot[] out) {

        if (startIndex == stopIndexExclusive) {
            return;
        }

        int cursor = startIndex;
        while (cursor != stopIndexExclusive) {
            final int idx = (int) order[cursor];
            if (FastMath.abs(targetTimes[idx]) > EPS_TIME) {
                break;
            }
            out[idx] = referenceState;
            cursor += indexStep;
        }
        if (cursor == stopIndexExclusive) {
            return;
        }
        final int firstCursor = cursor;

        final int lastCursor = stopIndexExclusive - indexStep;
        final double finalTime = targetTimes[(int) order[lastCursor]];
        final YawDynamics dynamics = new YawDynamics(pvProv, frame);
        final ODEIntegrator integrator =
                new DormandPrince853Integrator(
                        DEFAULT_MIN_STEP,
                        FastMath.max(DEFAULT_MAX_STEP, finiteDifferenceStep),
                        DEFAULT_ABS_TOL,
                        DEFAULT_REL_TOL);

        integrator.addStepHandler(new ODEStepHandler() {
            private int nextCursor = firstCursor;

            @Override
            public void handleStep(final ODEStateInterpolator interpolator) {
                final double previousTime = interpolator.getPreviousState().getTime();
                final double currentTime = interpolator.getCurrentState().getTime();

                while (nextCursor != stopIndexExclusive) {
                    final int idx = (int) order[nextCursor];
                    final double queryTime = targetTimes[idx];
                    final boolean withinStep =
                            indexStep > 0
                                    ? queryTime >= previousTime - EPS_TIME && queryTime <= currentTime + EPS_TIME
                                    : queryTime <= previousTime + EPS_TIME && queryTime >= currentTime - EPS_TIME;
                    if (!withinStep) {
                        break;
                    }
                    out[idx] = snapshotFromOdeState(interpolator.getInterpolatedState(queryTime));
                    nextCursor += indexStep;
                }
            }
        });

        final ODEState initialState = new ODEState(
                0.0,
                new double[] {referenceState.psi, clampYawRate(referenceState.omega)});
        final ODEStateAndDerivative finalState = integrator.integrate(dynamics, initialState, finalTime);

        int tailCursor = firstCursor;
        while (tailCursor != stopIndexExclusive) {
            final int idx = (int) order[tailCursor];
            if (FastMath.abs(targetTimes[idx] - finalTime) > EPS_TIME) {
                break;
            }
            out[idx] = snapshotFromOdeState(finalState);
            tailCursor += indexStep;
        }
    }

    private YawStateSnapshot computeReferenceYawStateFiniteDifference(
            final PVCoordinatesProvider pvProv,
            final AbsoluteDate date,
            final Frame frame) {

        final double h = finiteDifferenceStep;

        final double psiMinus = computeIdealYawAngleAtDate(pvProv, date.shiftedBy(-h), frame);
        final double psi0Value = computeIdealYawAngleAtDate(pvProv, date, frame);
        final double psiPlus = computeIdealYawAngleAtDate(pvProv, date.shiftedBy(h), frame);

        final double psiMinusUnwrapped = psi0Value - wrapMinusPiToPi(psi0Value - psiMinus);
        final double psiPlusUnwrapped = psi0Value + wrapMinusPiToPi(psiPlus - psi0Value);

        final double omega = (psiPlusUnwrapped - psiMinusUnwrapped) / (2.0 * h);
        final double alpha = (psiPlusUnwrapped - 2.0 * psi0Value + psiMinusUnwrapped) / (h * h);

        return new YawStateSnapshot(psi0Value, omega, alpha);
    }

    private double computeIdealYawAngleAtDate(
            final PVCoordinatesProvider pvProv,
            final AbsoluteDate date,
            final Frame frame) {

        final Attitude baseAttitude = baseLaw.getAttitude(pvProv, date, frame);
        final Attitude idealAttitude = idealLaw.getAttitude(pvProv, date, frame);
        return extractYawFromRelativeRotation(
                idealAttitude.getOrientation().subtractOffset(baseAttitude.getOrientation()).getRotation());
    }

    private double computeYawAcceleration(
            final PVCoordinatesProvider pvProv,
            final AbsoluteDate date,
            final Frame frame,
            final double psi,
            final double omega) {

        final YawStateSnapshot reference = computeReferenceYawState(pvProv, date, frame);

        final double ePsi = wrapMinusPiToPi(reference.psi - psi);
        final double eOmega = reference.omega - omega;

        final double alphaRaw = kp * ePsi + kd * eOmega + reference.alpha;
        double alphaCommand = clamp(alphaRaw, -maxYawAcceleration, maxYawAcceleration);

        if ((omega >= maxYawRate - EPS_RATE_LIMIT && alphaCommand > 0.0)
                || (omega <= -maxYawRate + EPS_RATE_LIMIT && alphaCommand < 0.0)) {
            alphaCommand = 0.0;
        }

        return alphaCommand;
    }

    private double clampYawRate(final double omega) {
        return clamp(omega, -maxYawRate, maxYawRate);
    }

    private double computeYawRateDerivative(final double rawOmega, final double alphaCommand) {
        if ((rawOmega >= maxYawRate - EPS_RATE_LIMIT && alphaCommand > 0.0)
                || (rawOmega <= -maxYawRate + EPS_RATE_LIMIT && alphaCommand < 0.0)) {
            return 0.0;
        }
        return alphaCommand;
    }

    private YawStateSnapshot ensureYawAcceleration(
            final PVCoordinatesProvider pvProv,
            final Frame frame,
            final double time,
            final YawStateSnapshot state) {

        if (Double.isFinite(state.alpha)) {
            return state;
        }

        final AbsoluteDate date = referenceEpoch.shiftedBy(time);
        final double alpha = computeYawAcceleration(pvProv, date, frame, state.psi, state.omega);
        return new YawStateSnapshot(state.psi, state.omega, alpha);
    }

    private void populateCheckpointRange(
            final PVCoordinatesProvider pvProv,
            final Frame frame,
            final long startIndex,
            final YawStateSnapshot startState,
            final long targetIndex) {

        if (targetIndex == startIndex) {
            checkpointCache.put(
                    startIndex,
                    ensureYawAcceleration(pvProv, frame, startIndex * cacheStep, startState));
            return;
        }

        final double startTime = startIndex * cacheStep;
        final double targetTime = targetIndex * cacheStep;
        final long indexStep = targetIndex > startIndex ? 1L : -1L;
        final double direction = indexStep > 0L ? 1.0 : -1.0;

        final YawDynamics dynamics = new YawDynamics(pvProv, frame);
        final ODEIntegrator integrator =
                new DormandPrince853Integrator(
                        DEFAULT_MIN_STEP,
                        FastMath.max(DEFAULT_MAX_STEP, finiteDifferenceStep),
                        DEFAULT_ABS_TOL,
                        DEFAULT_REL_TOL);

        integrator.addStepHandler(new ODEStepHandler() {
            private long nextIndex = startIndex + indexStep;

            @Override
            public void handleStep(final ODEStateInterpolator interpolator) {
                final double previousTime = interpolator.getPreviousState().getTime();
                final double currentTime = interpolator.getCurrentState().getTime();

                while ((direction > 0.0 && nextIndex <= targetIndex)
                        || (direction < 0.0 && nextIndex >= targetIndex)) {
                    final double checkpointTime = nextIndex * cacheStep;
                    final boolean withinStep =
                            direction > 0.0
                                    ? checkpointTime <= currentTime + EPS_TIME
                                    : checkpointTime >= currentTime - EPS_TIME;
                    if (!withinStep) {
                        break;
                    }

                    final boolean afterPrevious =
                            direction > 0.0
                                    ? checkpointTime >= previousTime - EPS_TIME
                                    : checkpointTime <= previousTime + EPS_TIME;
                    if (!afterPrevious) {
                        break;
                    }

                    checkpointCache.put(
                            nextIndex,
                            snapshotFromOdeState(interpolator.getInterpolatedState(checkpointTime)));
                    nextIndex += indexStep;
                }
            }
        });

        final ODEState initialState = new ODEState(
                startTime,
                new double[] {startState.psi, clampYawRate(startState.omega)});
        final ODEStateAndDerivative finalState = integrator.integrate(dynamics, initialState, targetTime);
        checkpointCache.put(targetIndex, snapshotFromOdeState(finalState));
    }

    private YawStateSnapshot snapshotFromOdeState(final ODEStateAndDerivative state) {
        return new YawStateSnapshot(
                state.getPrimaryState()[0],
                clampYawRate(state.getPrimaryState()[1]),
                state.getPrimaryDerivative()[1]);
    }

    private TimeStampedAngularCoordinates buildActualOrientation(
            final AbsoluteDate date,
            final Attitude baseAttitude,
            final YawStateSnapshot yawState) {

        // baseAttitude and the returned orientation both map the same reference frame into the
        // spacecraft body frame. The yaw offset is therefore a body-fixed rotation applied on top
        // of the base attitude, with positive yaw defined about body +Z.
        final Rotation yawRotation =
                new Rotation(Vector3D.PLUS_K, yawState.psi, RotationConvention.FRAME_TRANSFORM);
        final TimeStampedAngularCoordinates yawOffset = new TimeStampedAngularCoordinates(
                date,
                yawRotation,
                new Vector3D(0.0, 0.0, yawState.omega),
                new Vector3D(0.0, 0.0, yawState.alpha));
        return yawOffset.addOffset(baseAttitude.getOrientation());
    }

    private static long[] sortIndicesByTime(final double[] times) {
        final int n = times.length;
        final long[] order = new long[n];
        for (int i = 0; i < n; i++) {
            order[i] = i;
        }
        for (int i = 1; i < n; i++) {
            final long key = order[i];
            final double keyTime = times[(int) key];
            int j = i - 1;
            while (j >= 0 && times[(int) order[j]] > keyTime) {
                order[j + 1] = order[j];
                j--;
            }
            order[j + 1] = key;
        }
        return order;
    }

    private static void copyVectorOrZero(final Vector3D vector, final double[] out, final int offset) {
        if (vector == null) {
            out[offset] = 0.0;
            out[offset + 1] = 0.0;
            out[offset + 2] = 0.0;
            return;
        }
        out[offset] = vector.getX();
        out[offset + 1] = vector.getY();
        out[offset + 2] = vector.getZ();
    }

    private static double rotationDistance(final Rotation a, final Rotation b) {
        return FastMath.abs(a.applyInverseTo(b).getAngle());
    }

    private static double extractYawFromRelativeRotation(final Rotation relativeRotation) {
        // relativeRotation maps the base-attitude frame into the target-attitude frame. For a
        // pure positive body +Z yaw offset, the body +X axis rotates toward +Y in the base frame,
        // so the azimuth of body +X in the base-frame XY plane is the signed yaw angle.
        final Vector3D bodyXInBase = relativeRotation.applyInverseTo(Vector3D.PLUS_I);
        return FastMath.atan2(bodyXInBase.getY(), bodyXInBase.getX());
    }

    private static double wrapMinusPiToPi(final double angle) {
        return MathUtils.normalizeAngle(angle, 0.0);
    }

    private static double clamp(final double value, final double lower, final double upper) {
        return FastMath.max(lower, FastMath.min(upper, value));
    }

    /** Immutable yaw-state snapshot used by diagnostic accessors and tests. */
    public static final class YawStateSnapshot {
        /** Yaw angle in radians. */
        public final double psi;

        /** Yaw rate in radians per second. */
        public final double omega;

        /** Yaw acceleration in radians per second squared. */
        public final double alpha;

        public YawStateSnapshot(final double psi, final double omega, final double alpha) {
            this.psi = psi;
            this.omega = omega;
            this.alpha = alpha;
        }

        public double[] toArray() {
            return new double[] {psi, omega, alpha};
        }
    }

    private final class YawDynamics implements OrdinaryDifferentialEquation {
        private final PVCoordinatesProvider pvProv;
        private final Frame frame;

        private YawDynamics(final PVCoordinatesProvider pvProv, final Frame frame) {
            this.pvProv = pvProv;
            this.frame = frame;
        }

        @Override
        public int getDimension() {
            return 2;
        }

        @Override
        public double[] computeDerivatives(final double t, final double[] state) {
            final AbsoluteDate date = referenceEpoch.shiftedBy(t);
            final double psi = state[0];
            final double rawOmega = state[1];
            final double omega = clampYawRate(rawOmega);
            final double alpha = computeYawAcceleration(pvProv, date, frame, psi, omega);
            final double psiDot = omega;
            final double omegaDot = computeYawRateDerivative(rawOmega, alpha);
            return new double[] {psiDot, omegaDot};
        }
    }

    private static final class FieldToRegularPVProvider<T extends CalculusFieldElement<T>>
            implements PVCoordinatesProvider {

        private final FieldPVCoordinatesProvider<T> delegate;
        private final Field<T> field;

        private FieldToRegularPVProvider(
                final FieldPVCoordinatesProvider<T> delegate,
                final Field<T> field) {
            this.delegate = delegate;
            this.field = field;
        }

        @Override
        public TimeStampedPVCoordinates getPVCoordinates(final AbsoluteDate date, final Frame frame) {
            final TimeStampedFieldPVCoordinates<T> pv =
                    delegate.getPVCoordinates(new FieldAbsoluteDate<>(field, date), frame);
            return pv.toTimeStampedPVCoordinates();
        }
    }

    private static final class NonRecursivePropagatorPVProvider implements PVCoordinatesProvider {

        private final Propagator propagator;
        private final AttitudeProvider recursionSafeAttitudeProvider;

        private NonRecursivePropagatorPVProvider(
                final Propagator propagator,
                final Frame inertialFrame) {
            this.propagator = propagator;
            this.recursionSafeAttitudeProvider = new FrameAlignedProvider(inertialFrame);
        }

        @Override
        public TimeStampedPVCoordinates getPVCoordinates(final AbsoluteDate date, final Frame frame) {
            synchronized (propagator) {
                final AttitudeProvider originalProvider = propagator.getAttitudeProvider();
                propagator.setAttitudeProvider(recursionSafeAttitudeProvider);
                try {
                    return propagator.propagate(date).getPVCoordinates(frame);
                } finally {
                    propagator.setAttitudeProvider(originalProvider);
                }
            }
        }
    }
}
