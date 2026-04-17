package com.nstk.attitudes;

import java.util.stream.Stream;

import org.hipparchus.CalculusFieldElement;
import org.hipparchus.Field;
import org.hipparchus.geometry.euclidean.threed.Rotation;
import org.hipparchus.geometry.euclidean.threed.RotationConvention;
import org.hipparchus.geometry.euclidean.threed.Vector3D;
import org.hipparchus.ode.ODEIntegrator;
import org.hipparchus.ode.ODEState;
import org.hipparchus.ode.OrdinaryDifferentialEquation;
import org.hipparchus.ode.nonstiff.DormandPrince853Integrator;
import org.hipparchus.util.FastMath;
import org.hipparchus.util.MathUtils;
import org.orekit.attitudes.Attitude;
import org.orekit.attitudes.AttitudeProvider;
import org.orekit.attitudes.FieldAttitude;
import org.orekit.attitudes.GroundPointing;
import org.orekit.attitudes.NadirPointing;
import org.orekit.attitudes.YawSteering;
import org.orekit.bodies.BodyShape;
import org.orekit.frames.Frame;
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
 * <p>Reference yaw, yaw rate, and yaw acceleration are derived from the ideal Orekit
 * {@link YawSteering} law using centered finite differences. The relative yaw is extracted from
 * the relative rotation between the base nadir attitude and ideal yaw-steered attitude. The output
 * attitude is reconstructed by composing the base attitude with a yaw-only angular offset about
 * the spacecraft body +Z axis using Orekit {@link AngularCoordinates} composition.
 */
public class RateLimitedYawSteeringProvider implements AttitudeProvider {

    private static final double DEFAULT_MIN_STEP = 1.0e-12;
    private static final double DEFAULT_MAX_STEP = 30.0;
    private static final double[] DEFAULT_ABS_TOL = {1.0e-11, 1.0e-12};
    private static final double[] DEFAULT_REL_TOL = {1.0e-9, 1.0e-9};
    private static final double EPS_RATE_LIMIT = 1.0e-12;
    private static final double EPS_AXIS_NORM = 1.0e-15;

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
    private final GroundPointing baseLaw;
    private final YawSteering idealLaw;

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
        if (phasingAxis == null || phasingAxis.getNorm() <= EPS_AXIS_NORM) {
            throw new IllegalArgumentException("phasingAxis must be non-zero");
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

        this.inertialFrame = inertialFrame;
        this.bodyShape = bodyShape;
        this.sunProvider = sunProvider;
        this.phasingAxis = phasingAxis.normalize();
        this.maxYawRate = maxYawRate;
        this.maxYawAcceleration = maxYawAcceleration;
        this.kp = kp;
        this.kd = kd;
        this.referenceEpoch = referenceEpoch;
        this.psi0 = psi0;
        this.omega0 = omega0;
        this.finiteDifferenceStep = finiteDifferenceStep;
        this.trajectoryProvider = trajectoryProvider;
        this.baseLaw = new NadirPointing(inertialFrame, bodyShape);
        this.idealLaw = new YawSteering(inertialFrame, baseLaw, sunProvider, this.phasingAxis);
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

    @Override
    public Attitude getAttitude(
            final PVCoordinatesProvider pvProv,
            final AbsoluteDate date,
            final Frame frame) {

        final Frame evaluationFrame = frame == null ? inertialFrame : frame;
        final PVCoordinatesProvider effectivePvProvider = selectPvProvider(pvProv);
        final Attitude baseAttitude = baseLaw.getAttitude(effectivePvProvider, date, evaluationFrame);
        final YawStateSnapshot yawState = integrateYawState(effectivePvProvider, date, evaluationFrame);

        final Rotation yawRotation =
                new Rotation(Vector3D.PLUS_K, yawState.psi, RotationConvention.FRAME_TRANSFORM);
        final TimeStampedAngularCoordinates yawOffset = new TimeStampedAngularCoordinates(
                date,
                yawRotation,
                new Vector3D(0.0, 0.0, yawState.omega),
                new Vector3D(0.0, 0.0, yawState.alpha));

        final TimeStampedAngularCoordinates actualOrientation =
                yawOffset.addOffset(baseAttitude.getOrientation());
        return new Attitude(evaluationFrame, actualOrientation);
    }

    @Override
    public <T extends CalculusFieldElement<T>> FieldAttitude<T> getAttitude(
            final FieldPVCoordinatesProvider<T> pvProv,
            final FieldAbsoluteDate<T> date,
            final Frame frame) {

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

        final double duration = date.durationFrom(referenceEpoch);
        final double initialOmega = clampYawRate(omega0);

        if (duration == 0.0) {
            final double alpha0 = computeYawAcceleration(pvProv, referenceEpoch, frame, psi0, initialOmega);
            return new YawStateSnapshot(psi0, initialOmega, alpha0);
        }

        final YawDynamics dynamics = new YawDynamics(pvProv, frame);
        final ODEIntegrator integrator =
                new DormandPrince853Integrator(
                        DEFAULT_MIN_STEP,
                        FastMath.max(DEFAULT_MAX_STEP, finiteDifferenceStep),
                        DEFAULT_ABS_TOL,
                        DEFAULT_REL_TOL);

        final ODEState initialState = new ODEState(0.0, new double[] {psi0, initialOmega});
        final ODEState finalState = integrator.integrate(dynamics, initialState, duration);

        final double psi = finalState.getPrimaryState()[0];
        final double omega = clampYawRate(finalState.getPrimaryState()[1]);
        final double alpha = computeYawAcceleration(pvProv, date, frame, psi, omega);
        return new YawStateSnapshot(psi, omega, alpha);
    }

    private PVCoordinatesProvider selectPvProvider(final PVCoordinatesProvider callTimeProvider) {
        return trajectoryProvider != null ? trajectoryProvider : callTimeProvider;
    }

    private YawStateSnapshot computeReferenceYawState(
            final PVCoordinatesProvider pvProv,
            final AbsoluteDate date,
            final Frame frame) {

        final double h = finiteDifferenceStep;

        final double psiMinus = computeIdealYawAngle(pvProv, date.shiftedBy(-h), frame);
        final double psi0Value = computeIdealYawAngle(pvProv, date, frame);
        final double psiPlus = computeIdealYawAngle(pvProv, date.shiftedBy(h), frame);

        final double psiMinusUnwrapped = psi0Value - wrapMinusPiToPi(psi0Value - psiMinus);
        final double psiPlusUnwrapped = psi0Value + wrapMinusPiToPi(psiPlus - psi0Value);

        final double omega = (psiPlusUnwrapped - psiMinusUnwrapped) / (2.0 * h);
        final double alpha = (psiPlusUnwrapped - 2.0 * psi0Value + psiMinusUnwrapped) / (h * h);

        return new YawStateSnapshot(psi0Value, omega, alpha);
    }

    private double computeIdealYawAngle(
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

    private static double extractYawFromRelativeRotation(final Rotation relativeRotation) {
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
            final double omega = state[1];
            final double psiDot = clampYawRate(omega);
            final double alpha = computeYawAcceleration(pvProv, date, frame, psi, omega);
            return new double[] {psiDot, alpha};
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
}
