package com.nstk.propagation;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.hipparchus.geometry.euclidean.threed.Rotation;
import org.hipparchus.geometry.euclidean.threed.RotationConvention;
import org.hipparchus.geometry.euclidean.threed.RotationOrder;
import org.hipparchus.geometry.euclidean.threed.Vector3D;
import org.orekit.attitudes.Attitude;
import org.orekit.attitudes.AttitudeProvider;
import org.orekit.attitudes.LofOffset;
import org.orekit.bodies.GeodeticPoint;
import org.orekit.bodies.OneAxisEllipsoid;
import org.orekit.frames.Frame;
import org.orekit.frames.LOFType;
import org.orekit.orbits.CartesianOrbit;
import org.orekit.orbits.EquinoctialOrbit;
import org.orekit.orbits.KeplerianOrbit;
import org.orekit.orbits.Orbit;
import org.orekit.orbits.PositionAngleType;
import org.orekit.propagation.BoundedPropagator;
import org.orekit.propagation.EphemerisGenerator;
import org.orekit.propagation.Propagator;
import org.orekit.propagation.SpacecraftState;
import org.orekit.propagation.analytical.EcksteinHechlerPropagator;
import org.orekit.propagation.analytical.KeplerianPropagator;
import org.orekit.time.AbsoluteDate;
import org.orekit.utils.Constants;
import org.orekit.utils.DataDictionary;
import org.orekit.utils.DoubleArrayDictionary;
import org.orekit.utils.PVCoordinates;

/**
 * Java-first orbit engine for the standalone orbit propagation interface.
 *
 * <p>All propagation loops, ephemeris interpolation, frame transforms, geodetic conversion,
 * and vectorized state extraction are executed in Java.
 */
public final class OrekitOrbitPropagationBridge {

    private static final double MIN_WINDOW_SECONDS = 1.0e-6;

    private final Propagator propagator;
    private final AbsoluteDate epoch;
    private final Frame nativeFrame;
    private final boolean cacheEnabled;

    private BoundedPropagator ephemeris;
    private boolean hasEphemeris;
    private double tMinSeconds;
    private double tMaxSeconds;

    public OrekitOrbitPropagationBridge(final Propagator propagator) {
        this(propagator, true);
    }

    public OrekitOrbitPropagationBridge(final Propagator propagator, final boolean cacheEnabled) {
        if (propagator == null) {
            throw new IllegalArgumentException("propagator must not be null");
        }
        this.propagator = propagator;

        final SpacecraftState state0 = propagator.getInitialState();
        this.epoch = state0.getDate();
        this.nativeFrame = state0.getFrame();
        this.cacheEnabled = cacheEnabled;

        this.ephemeris = null;
        this.hasEphemeris = false;
        this.tMinSeconds = 0.0;
        this.tMaxSeconds = 0.0;
    }

    public static OrekitOrbitPropagationBridge fromPropagator(final Propagator propagator) {
        return new OrekitOrbitPropagationBridge(propagator);
    }

    public static OrekitOrbitPropagationBridge fromPropagator(
            final Propagator propagator,
            final boolean cacheEnabled) {
        return new OrekitOrbitPropagationBridge(propagator, cacheEnabled);
    }

    public static OrekitOrbitPropagationBridge fromSpacecraftState(final SpacecraftState state) {
        return fromSpacecraftState(state, true);
    }

    public static OrekitOrbitPropagationBridge fromSpacecraftState(
            final SpacecraftState state,
            final boolean cacheEnabled) {
        if (state == null) {
            throw new IllegalArgumentException("state must not be null");
        }

        final KeplerianPropagator propagator = new KeplerianPropagator(state.getOrbit());
        propagator.resetInitialState(state);
        applyDefaultAttitudeProvider(propagator, state.getFrame());

        return new OrekitOrbitPropagationBridge(propagator, cacheEnabled);
    }

    public static OrekitOrbitPropagationBridge fromKeplerTwoBody(
            final AbsoluteDate epoch,
            final double a,
            final double e,
            final double i,
            final double raan,
            final double argp,
            final double anomaly,
            final PositionAngleType anomalyType,
            final double mass,
            final Frame inertialFrame) {
        return fromKeplerTwoBody(
                epoch,
                a,
                e,
                i,
                raan,
                argp,
                anomaly,
                anomalyType,
                mass,
                inertialFrame,
                true);
    }

    public static OrekitOrbitPropagationBridge fromKeplerTwoBody(
            final AbsoluteDate epoch,
            final double a,
            final double e,
            final double i,
            final double raan,
            final double argp,
            final double anomaly,
            final PositionAngleType anomalyType,
            final double mass,
            final Frame inertialFrame,
            final boolean cacheEnabled) {

        if (epoch == null) {
            throw new IllegalArgumentException("epoch must not be null");
        }
        if (inertialFrame == null) {
            throw new IllegalArgumentException("inertialFrame must not be null");
        }

        final PositionAngleType paType = anomalyType == null ? PositionAngleType.MEAN : anomalyType;

        final KeplerianOrbit orbit =
                new KeplerianOrbit(
                        a,
                        e,
                        i,
                        argp,
                        raan,
                        anomaly,
                        paType,
                        inertialFrame,
                        epoch,
                        Constants.EIGEN5C_EARTH_MU);

        final SpacecraftState initialState = new SpacecraftState(orbit, mass);
        final KeplerianPropagator propagator = new KeplerianPropagator(orbit);
        propagator.resetInitialState(initialState);
        applyDefaultAttitudeProvider(propagator, inertialFrame);

        return new OrekitOrbitPropagationBridge(propagator, cacheEnabled);
    }

    public static OrekitOrbitPropagationBridge fromKeplerEcksteinHechler(
            final AbsoluteDate epoch,
            final double a,
            final double e,
            final double i,
            final double raan,
            final double argp,
            final double anomaly,
            final PositionAngleType anomalyType,
            final double mass,
            final Frame inertialFrame) {
        return fromKeplerEcksteinHechler(
                epoch,
                a,
                e,
                i,
                raan,
                argp,
                anomaly,
                anomalyType,
                mass,
                inertialFrame,
                true);
    }

    public static OrekitOrbitPropagationBridge fromKeplerEcksteinHechler(
            final AbsoluteDate epoch,
            final double a,
            final double e,
            final double i,
            final double raan,
            final double argp,
            final double anomaly,
            final PositionAngleType anomalyType,
            final double mass,
            final Frame inertialFrame,
            final boolean cacheEnabled) {

        if (epoch == null) {
            throw new IllegalArgumentException("epoch must not be null");
        }
        if (inertialFrame == null) {
            throw new IllegalArgumentException("inertialFrame must not be null");
        }

        final PositionAngleType paType = anomalyType == null ? PositionAngleType.MEAN : anomalyType;

        final KeplerianOrbit orbit =
                new KeplerianOrbit(
                        a,
                        e,
                        i,
                        argp,
                        raan,
                        anomaly,
                        paType,
                        inertialFrame,
                        epoch,
                        Constants.EIGEN5C_EARTH_MU);

        final EcksteinHechlerPropagator propagator =
                new EcksteinHechlerPropagator(
                        orbit,
                        Constants.EIGEN5C_EARTH_EQUATORIAL_RADIUS,
                        Constants.EIGEN5C_EARTH_MU,
                        Constants.EIGEN5C_EARTH_C20,
                        Constants.EIGEN5C_EARTH_C30,
                        Constants.EIGEN5C_EARTH_C40,
                        Constants.EIGEN5C_EARTH_C50,
                        Constants.EIGEN5C_EARTH_C60);

        try {
            final SpacecraftState initialState = new SpacecraftState(orbit, mass);
            propagator.resetInitialState(initialState);
        } catch (Exception ignored) {
            // Keep the propagator default state when reset is unsupported.
        }

        applyDefaultAttitudeProvider(propagator, inertialFrame);

        return new OrekitOrbitPropagationBridge(propagator, cacheEnabled);
    }

    public Propagator getPropagator() {
        return propagator;
    }

    public AbsoluteDate getEpoch() {
        return epoch;
    }

    public Frame getNativeFrame() {
        return nativeFrame;
    }

    public boolean isCachingEnabled() {
        return cacheEnabled;
    }

    public AbsoluteDate getStartDate() {
        if (propagator instanceof BoundedPropagator) {
            return ((BoundedPropagator) propagator).getMinDate();
        }
        return AbsoluteDate.PAST_INFINITY;
    }

    public AbsoluteDate getStopDate() {
        if (propagator instanceof BoundedPropagator) {
            return ((BoundedPropagator) propagator).getMaxDate();
        }
        return AbsoluteDate.FUTURE_INFINITY;
    }

    public synchronized void setAttitudeProvider(final AttitudeProvider provider) {
        if (provider == null) {
            throw new IllegalArgumentException("provider must not be null");
        }
        propagator.setAttitudeProvider(provider);
        clearEphemerisCache();
    }

    public synchronized void clearCache() {
        clearEphemerisCache();
    }

    public synchronized double[] coverage() {
        if (!cacheEnabled || !hasEphemeris) {
            return new double[] {0.0, 0.0};
        }
        return new double[] {tMinSeconds, tMaxSeconds};
    }

    public synchronized void precompute(final double tMinSeconds, final double tMaxSeconds) {
        if (!cacheEnabled) {
            return;
        }
        ensureCoveredRange(tMinSeconds, tMaxSeconds);
    }

    public synchronized double[] queryPosition(final double[] dtSeconds, final Frame outputFrame) {
        return queryState(dtSeconds, outputFrame, true, false, false).p;
    }

    public synchronized double[] queryVelocity(final double[] dtSeconds, final Frame outputFrame) {
        return queryState(dtSeconds, outputFrame, false, true, false).v;
    }

    public synchronized double[] queryAcceleration(final double[] dtSeconds, final Frame outputFrame) {
        return queryState(dtSeconds, outputFrame, false, false, true).a;
    }

    public synchronized PVBatchResult queryPV(final double[] dtSeconds, final Frame outputFrame) {
        final StateBatchResult out = queryState(dtSeconds, outputFrame, true, true, false);
        return new PVBatchResult(out.p, out.v);
    }

    public synchronized PVABatchResult queryPVA(final double[] dtSeconds, final Frame outputFrame) {
        final StateBatchResult out = queryState(dtSeconds, outputFrame, true, true, true);
        return new PVABatchResult(out.p, out.v, out.a);
    }

    public synchronized GeodeticBatchResult queryGeodetic(
            final double[] dtSeconds,
            final OneAxisEllipsoid ellipsoid) {
        if (ellipsoid == null) {
            throw new IllegalArgumentException("ellipsoid must not be null");
        }

        final int n = dtSeconds.length;
        final double[] latDeg = new double[n];
        final double[] lonDeg = new double[n];
        final double[] altM = new double[n];

        if (n == 0) {
            return new GeodeticBatchResult(latDeg, lonDeg, altM);
        }

        final boolean directOrbitFastPath = isAnalyticalOrbitFastPathEligible();
        if (directOrbitFastPath) {
            validateQueryTimes(dtSeconds);
        } else {
            ensureCovered(dtSeconds);
        }
        final Frame bodyFrame = ellipsoid.getBodyFrame();

        for (int i = 0; i < n; i++) {
            final AbsoluteDate absT = epoch.shiftedBy(dtSeconds[i]);
            final Vector3D pos =
                    directOrbitFastPath
                            ? propagateOrbitDirect(absT).getPVCoordinates(bodyFrame).getPosition()
                            : getPVCoordinatesAt(absT, bodyFrame).getPosition();
            final GeodeticPoint gp = ellipsoid.transform(pos, bodyFrame, absT);

            latDeg[i] = Math.toDegrees(gp.getLatitude());
            lonDeg[i] = Math.toDegrees(gp.getLongitude());
            altM[i] = gp.getAltitude();
        }

        return new GeodeticBatchResult(latDeg, lonDeg, altM);
    }

    public synchronized double[] queryAttitudeQuaternion(final double[] dtSeconds) {
        final int n = dtSeconds.length;
        final double[] out = new double[4 * n];

        if (n == 0) {
            return out;
        }

        ensureCovered(dtSeconds);

        for (int i = 0; i < n; i++) {
            final int off = 4 * i;
            final AbsoluteDate absT = epoch.shiftedBy(dtSeconds[i]);
            final SpacecraftState state = propagateState(absT);
            final Rotation rot = state.getAttitude().getRotation();

            out[off] = rot.getQ0();
            out[off + 1] = rot.getQ1();
            out[off + 2] = rot.getQ2();
            out[off + 3] = rot.getQ3();
        }

        return out;
    }

    public synchronized double[] queryAttitudeSpin(final double[] dtSeconds) {
        return queryAttitudeVector(dtSeconds, false);
    }

    public synchronized double[] queryAttitudeRotationAcceleration(final double[] dtSeconds) {
        return queryAttitudeVector(dtSeconds, true);
    }

    public synchronized SpacecraftState[] queryStates(final double[] dtSeconds) {
        final int n = dtSeconds.length;
        final SpacecraftState[] out = new SpacecraftState[n];

        if (n == 0) {
            return out;
        }

        ensureCovered(dtSeconds);
        for (int i = 0; i < n; i++) {
            out[i] = propagateState(epoch.shiftedBy(dtSeconds[i]));
        }
        return out;
    }

    public synchronized String[] listAdditionalDataNames() {
        return getDataDictionaryKeys(propagator.getInitialState().getAdditionalDataValues());
    }

    public synchronized String[] listAdditionalStateDerivativeNames() {
        return getDoubleArrayDictionaryKeys(propagator.getInitialState().getAdditionalStatesDerivatives());
    }

    public synchronized SampleBatchResult sample(
            final double[] dtSeconds,
            final Frame cartesianFrame,
            final boolean position,
            final boolean velocity,
            final boolean acceleration,
            final Frame attitudeReferenceFrame,
            final boolean attitudeQuaternion,
            final boolean attitudeMatrix,
            final boolean attitudeEuler,
            final boolean attitudeSpin,
            final boolean attitudeAcceleration,
            final String attitudeEulerSequence,
            final boolean attitudeEulerDegrees,
            final boolean quaternionScalarLast,
            final Frame elementsFrame,
            final boolean keplerian,
            final PositionAngleType anomalyType,
            final boolean equinoctial,
            final PositionAngleType longitudeType,
            final boolean elementsAnglesDegrees,
            final boolean mass,
            final String[] additionalStates,
            final String[] additionalStateDerivatives,
            final boolean strict) {

        if (dtSeconds == null) {
            throw new IllegalArgumentException("dtSeconds must not be null");
        }

        final int n = dtSeconds.length;
        final Frame cartesianTarget = cartesianFrame == null ? nativeFrame : cartesianFrame;
        final Frame attitudeTarget = attitudeReferenceFrame == null ? nativeFrame : attitudeReferenceFrame;
        final Frame elementsTarget = elementsFrame == null ? nativeFrame : elementsFrame;

        if ((keplerian || equinoctial) && !elementsTarget.isPseudoInertial()) {
            throw new IllegalArgumentException("elementsFrame must be pseudo-inertial");
        }

        final double[] positionM = position ? new double[3 * n] : null;
        final double[] velocityMps = velocity ? new double[3 * n] : null;
        final double[] accelerationMps2 = acceleration ? new double[3 * n] : null;

        final double[] attitudeQuatRefToBody = attitudeQuaternion ? new double[4 * n] : null;
        final double[] attitudeMatrixRefToBody = attitudeMatrix ? new double[9 * n] : null;
        final double[] attitudeEulerRefToBody = attitudeEuler ? new double[3 * n] : null;
        final double[] attitudeSpinBodyRadS = attitudeSpin ? new double[3 * n] : null;
        final double[] attitudeAccelBodyRadS2 = attitudeAcceleration ? new double[3 * n] : null;

        final double[] semiMajorAxisM = keplerian ? new double[n] : null;
        final double[] eccentricity = keplerian ? new double[n] : null;
        final double[] inclination = keplerian ? new double[n] : null;
        final double[] raan = keplerian ? new double[n] : null;
        final double[] argp = keplerian ? new double[n] : null;
        final double[] anomaly = keplerian ? new double[n] : null;

        final double[] equinoctialAM = equinoctial ? new double[n] : null;
        final double[] equinoctialEx = equinoctial ? new double[n] : null;
        final double[] equinoctialEy = equinoctial ? new double[n] : null;
        final double[] equinoctialHx = equinoctial ? new double[n] : null;
        final double[] equinoctialHy = equinoctial ? new double[n] : null;
        final double[] equinoctialLongitude = equinoctial ? new double[n] : null;

        final double[] massKg = mass ? new double[n] : null;

        final AdditionalSeries[] additionalSeries = AdditionalSeries.build(additionalStates);
        final AdditionalSeries[] additionalDerivativeSeries = AdditionalSeries.build(additionalStateDerivatives);

        if (n == 0) {
            return new SampleBatchResult(
                    positionM,
                    velocityMps,
                    accelerationMps2,
                    attitudeQuatRefToBody,
                    attitudeMatrixRefToBody,
                    attitudeEulerRefToBody,
                    attitudeSpinBodyRadS,
                    attitudeAccelBodyRadS2,
                    semiMajorAxisM,
                    eccentricity,
                    inclination,
                    raan,
                    argp,
                    anomaly,
                    equinoctialAM,
                    equinoctialEx,
                    equinoctialEy,
                    equinoctialHx,
                    equinoctialHy,
                    equinoctialLongitude,
                    massKg,
                    packNames(additionalSeries),
                    packValues(additionalSeries),
                    packWidths(additionalSeries),
                    packNames(additionalDerivativeSeries),
                    packValues(additionalDerivativeSeries),
                    packWidths(additionalDerivativeSeries));
        }

        final boolean needCartesian = position || velocity || acceleration;
        final boolean needAttitude =
                attitudeQuaternion || attitudeMatrix || attitudeEuler || attitudeSpin || attitudeAcceleration;
        final boolean needElements = keplerian || equinoctial;
        final boolean needsState =
                needAttitude
                        || mass
                        || (additionalSeries != null)
                        || (additionalDerivativeSeries != null);
        final boolean directOrbitFastPath = isAnalyticalOrbitFastPathEligible() && !needsState;
        final RotationOrder rotationOrder = attitudeEuler ? parseRotationOrder(attitudeEulerSequence) : null;
        final PositionAngleType anomalyMode = anomalyType == null ? PositionAngleType.MEAN : anomalyType;
        final PositionAngleType longitudeMode = longitudeType == null ? PositionAngleType.MEAN : longitudeType;

        if (needCartesian && !needsState) {
            final StateBatchResult cartesianOnly =
                    queryState(dtSeconds, cartesianTarget, position, velocity, acceleration);
            return new SampleBatchResult(
                    cartesianOnly.p,
                    cartesianOnly.v,
                    cartesianOnly.a,
                    null,
                    null,
                    null,
                    null,
                    null,
                    null,
                    null,
                    null,
                    null,
                    null,
                    null,
                    null,
                    null,
                    null,
                    null,
                    null,
                    null,
                    null,
                    new String[0],
                    new double[0][],
                    new int[0],
                    new String[0],
                    new double[0][],
                    new int[0]);
        }

        if (directOrbitFastPath) {
            validateQueryTimes(dtSeconds);

            for (int i = 0; i < n; i++) {
                final int vectorOffset = 3 * i;
                final AbsoluteDate absT = epoch.shiftedBy(dtSeconds[i]);
                final Orbit orbit = propagateOrbitDirect(absT);

                if (needCartesian) {
                    final PVCoordinates pv = orbit.getPVCoordinates(cartesianTarget);
                    if (positionM != null) {
                        copyVector(pv.getPosition(), positionM, vectorOffset);
                    }
                    if (velocityMps != null) {
                        copyVector(pv.getVelocity(), velocityMps, vectorOffset);
                    }
                    if (accelerationMps2 != null) {
                        copyVectorOrZero(pv.getAcceleration(), accelerationMps2, vectorOffset);
                    }
                }

                if (needElements) {
                    final Orbit orbitInFrame = getOrbitInFrame(orbit, elementsTarget, absT);
                    if (keplerian) {
                        final KeplerianOrbit kep = new KeplerianOrbit(orbitInFrame);
                        semiMajorAxisM[i] = kep.getA();
                        eccentricity[i] = kep.getE();
                        inclination[i] = angleMaybeDegrees(kep.getI(), elementsAnglesDegrees);
                        raan[i] =
                                angleMaybeDegrees(
                                        kep.getRightAscensionOfAscendingNode(),
                                        elementsAnglesDegrees);
                        argp[i] = angleMaybeDegrees(kep.getPerigeeArgument(), elementsAnglesDegrees);
                        anomaly[i] =
                                angleMaybeDegrees(getKeplerianAnomaly(kep, anomalyMode), elementsAnglesDegrees);
                    }
                    if (equinoctial) {
                        final EquinoctialOrbit equi = new EquinoctialOrbit(orbitInFrame);
                        equinoctialAM[i] = equi.getA();
                        equinoctialEx[i] = equi.getEquinoctialEx();
                        equinoctialEy[i] = equi.getEquinoctialEy();
                        equinoctialHx[i] = equi.getHx();
                        equinoctialHy[i] = equi.getHy();
                        equinoctialLongitude[i] =
                                angleMaybeDegrees(
                                        getEquinoctialLongitude(equi, longitudeMode),
                                        elementsAnglesDegrees);
                    }
                }
            }

            return new SampleBatchResult(
                    positionM,
                    velocityMps,
                    accelerationMps2,
                    null,
                    null,
                    null,
                    null,
                    null,
                    semiMajorAxisM,
                    eccentricity,
                    inclination,
                    raan,
                    argp,
                    anomaly,
                    equinoctialAM,
                    equinoctialEx,
                    equinoctialEy,
                    equinoctialHx,
                    equinoctialHy,
                    equinoctialLongitude,
                    null,
                    new String[0],
                    new double[0][],
                    new int[0],
                    new String[0],
                    new double[0][],
                    new int[0]);
        }

        ensureCovered(dtSeconds);

        for (int i = 0; i < n; i++) {
            final int vectorOffset = 3 * i;
            final int quatOffset = 4 * i;
            final int matrixOffset = 9 * i;
            final AbsoluteDate absT = epoch.shiftedBy(dtSeconds[i]);
            final SpacecraftState state = propagateState(absT);

            if (needCartesian) {
                final PVCoordinates pv = state.getPVCoordinates(cartesianTarget);
                if (positionM != null) {
                    copyVector(pv.getPosition(), positionM, vectorOffset);
                }
                if (velocityMps != null) {
                    copyVector(pv.getVelocity(), velocityMps, vectorOffset);
                }
                if (accelerationMps2 != null) {
                    copyVectorOrZero(pv.getAcceleration(), accelerationMps2, vectorOffset);
                }
            }

            if (needAttitude) {
                final Attitude attitude = getAttitudeInReferenceFrame(state, attitudeTarget);
                final Rotation rot = attitude.getRotation();

                if (attitudeQuatRefToBody != null) {
                    copyQuaternion(rot, attitudeQuatRefToBody, quatOffset, quaternionScalarLast);
                }
                if (attitudeMatrixRefToBody != null) {
                    copyMatrix(rot, attitudeMatrixRefToBody, matrixOffset);
                }
                if (attitudeEulerRefToBody != null) {
                    try {
                        final double[] angles =
                                rot.getAngles(rotationOrder, RotationConvention.FRAME_TRANSFORM);
                        if (attitudeEulerDegrees) {
                            attitudeEulerRefToBody[vectorOffset] = Math.toDegrees(angles[0]);
                            attitudeEulerRefToBody[vectorOffset + 1] = Math.toDegrees(angles[1]);
                            attitudeEulerRefToBody[vectorOffset + 2] = Math.toDegrees(angles[2]);
                        } else {
                            attitudeEulerRefToBody[vectorOffset] = angles[0];
                            attitudeEulerRefToBody[vectorOffset + 1] = angles[1];
                            attitudeEulerRefToBody[vectorOffset + 2] = angles[2];
                        }
                    } catch (Exception exc) {
                        if (strict) {
                            throw new IllegalArgumentException("Failed to extract attitude Euler angles", exc);
                        }
                        fillNaN(attitudeEulerRefToBody, vectorOffset, 3);
                    }
                }
                if (attitudeSpinBodyRadS != null) {
                    copyVectorOrZero(attitude.getSpin(), attitudeSpinBodyRadS, vectorOffset);
                }
                if (attitudeAccelBodyRadS2 != null) {
                    final Vector3D accelVec = attitude.getRotationAcceleration();
                    if (accelVec == null && strict) {
                        throw new IllegalArgumentException("Attitude rotation acceleration is unavailable");
                    }
                    copyVectorOrZero(accelVec, attitudeAccelBodyRadS2, vectorOffset);
                }
            }

            if (needElements) {
                final Orbit orbitInFrame = getOrbitInFrame(state, elementsTarget);
                if (keplerian) {
                    try {
                        final KeplerianOrbit kep = new KeplerianOrbit(orbitInFrame);
                        semiMajorAxisM[i] = kep.getA();
                        eccentricity[i] = kep.getE();
                        inclination[i] = angleMaybeDegrees(kep.getI(), elementsAnglesDegrees);
                        raan[i] =
                                angleMaybeDegrees(
                                        kep.getRightAscensionOfAscendingNode(),
                                        elementsAnglesDegrees);
                        argp[i] = angleMaybeDegrees(kep.getPerigeeArgument(), elementsAnglesDegrees);
                        anomaly[i] =
                                angleMaybeDegrees(getKeplerianAnomaly(kep, anomalyMode), elementsAnglesDegrees);
                    } catch (Exception exc) {
                        if (strict) {
                            throw new IllegalArgumentException("Failed to extract Keplerian elements", exc);
                        }
                        semiMajorAxisM[i] = Double.NaN;
                        eccentricity[i] = Double.NaN;
                        inclination[i] = Double.NaN;
                        raan[i] = Double.NaN;
                        argp[i] = Double.NaN;
                        anomaly[i] = Double.NaN;
                    }
                }
                if (equinoctial) {
                    try {
                        final EquinoctialOrbit equi = new EquinoctialOrbit(orbitInFrame);
                        equinoctialAM[i] = equi.getA();
                        equinoctialEx[i] = equi.getEquinoctialEx();
                        equinoctialEy[i] = equi.getEquinoctialEy();
                        equinoctialHx[i] = equi.getHx();
                        equinoctialHy[i] = equi.getHy();
                        equinoctialLongitude[i] =
                                angleMaybeDegrees(
                                        getEquinoctialLongitude(equi, longitudeMode),
                                        elementsAnglesDegrees);
                    } catch (Exception exc) {
                        if (strict) {
                            throw new IllegalArgumentException("Failed to extract equinoctial elements", exc);
                        }
                        equinoctialAM[i] = Double.NaN;
                        equinoctialEx[i] = Double.NaN;
                        equinoctialEy[i] = Double.NaN;
                        equinoctialHx[i] = Double.NaN;
                        equinoctialHy[i] = Double.NaN;
                        equinoctialLongitude[i] = Double.NaN;
                    }
                }
            }

            if (massKg != null) {
                massKg[i] = state.getMass();
            }

            appendAdditionalSeries(additionalSeries, state, i, n, strict, false);
            appendAdditionalSeries(additionalDerivativeSeries, state, i, n, strict, true);
        }

        return new SampleBatchResult(
                positionM,
                velocityMps,
                accelerationMps2,
                attitudeQuatRefToBody,
                attitudeMatrixRefToBody,
                attitudeEulerRefToBody,
                attitudeSpinBodyRadS,
                attitudeAccelBodyRadS2,
                semiMajorAxisM,
                eccentricity,
                inclination,
                raan,
                argp,
                anomaly,
                equinoctialAM,
                equinoctialEx,
                equinoctialEy,
                equinoctialHx,
                equinoctialHy,
                equinoctialLongitude,
                massKg,
                packNames(additionalSeries),
                packValues(additionalSeries),
                packWidths(additionalSeries),
                packNames(additionalDerivativeSeries),
                packValues(additionalDerivativeSeries),
                packWidths(additionalDerivativeSeries));
    }

    private StateBatchResult queryState(
            final double[] dtSeconds,
            final Frame outputFrame,
            final boolean needPosition,
            final boolean needVelocity,
            final boolean needAcceleration) {

        final int n = dtSeconds.length;
        final double[] p = needPosition ? new double[3 * n] : null;
        final double[] v = needVelocity ? new double[3 * n] : null;
        final double[] a = needAcceleration ? new double[3 * n] : null;

        if (n == 0) {
            return new StateBatchResult(p, v, a);
        }

        final boolean directOrbitFastPath = isAnalyticalOrbitFastPathEligible();
        if (directOrbitFastPath) {
            validateQueryTimes(dtSeconds);
        } else {
            ensureCovered(dtSeconds);
        }
        final Frame target = outputFrame == null ? nativeFrame : outputFrame;

        for (int i = 0; i < n; i++) {
            final int off = 3 * i;
            final AbsoluteDate absT = epoch.shiftedBy(dtSeconds[i]);
            final PVCoordinates pv =
                    directOrbitFastPath
                            ? propagateOrbitDirect(absT).getPVCoordinates(target)
                            : getPVCoordinatesAt(absT, target);

            if (needPosition && p != null) {
                copyVector(pv.getPosition(), p, off);
            }
            if (needVelocity && v != null) {
                copyVector(pv.getVelocity(), v, off);
            }
            if (needAcceleration && a != null) {
                copyVectorOrZero(pv.getAcceleration(), a, off);
            }
        }

        return new StateBatchResult(p, v, a);
    }

    private double[] queryAttitudeVector(final double[] dtSeconds, final boolean rotationAcceleration) {
        final int n = dtSeconds.length;
        final double[] out = new double[3 * n];

        if (n == 0) {
            return out;
        }

        ensureCovered(dtSeconds);

        for (int i = 0; i < n; i++) {
            final int off = 3 * i;
            final AbsoluteDate absT = epoch.shiftedBy(dtSeconds[i]);
            final SpacecraftState state = propagateState(absT);
            final Vector3D vec =
                    rotationAcceleration
                            ? state.getAttitude().getRotationAcceleration()
                            : state.getAttitude().getSpin();
            copyVectorOrZero(vec, out, off);
        }

        return out;
    }

    private SpacecraftState propagateState(final AbsoluteDate absT) {
        return cacheEnabled && hasEphemeris ? ephemeris.propagate(absT) : propagator.propagate(absT);
    }

    private PVCoordinates getPVCoordinatesAt(final AbsoluteDate absT, final Frame targetFrame) {
        return cacheEnabled && hasEphemeris
                ? ephemeris.getPVCoordinates(absT, targetFrame)
                : propagator.getPVCoordinates(absT, targetFrame);
    }

    private boolean isAnalyticalOrbitFastPathEligible() {
        return propagator instanceof KeplerianPropagator || propagator instanceof EcksteinHechlerPropagator;
    }

    private Orbit propagateOrbitDirect(final AbsoluteDate absT) {
        if (propagator instanceof KeplerianPropagator) {
            return ((KeplerianPropagator) propagator).propagateOrbit(absT);
        }
        if (propagator instanceof EcksteinHechlerPropagator) {
            return ((EcksteinHechlerPropagator) propagator).propagateOrbit(absT);
        }
        throw new IllegalStateException("Direct orbit propagation is unavailable for this propagator type");
    }

    private void validateQueryTimes(final double[] dtSeconds) {
        for (double t : dtSeconds) {
            if (!Double.isFinite(t)) {
                throw new IllegalArgumentException("query times must be finite");
            }
        }
    }

    private Attitude getAttitudeInReferenceFrame(final SpacecraftState state, final Frame targetFrame) {
        final Attitude attitude = state.getAttitude();
        if (targetFrame == null || attitude.getReferenceFrame().equals(targetFrame)) {
            return attitude;
        }
        return attitude.withReferenceFrame(targetFrame);
    }

    private Orbit getOrbitInFrame(final SpacecraftState state, final Frame targetFrame) {
        final Orbit orbit = state.getOrbit();
        if (targetFrame == null || orbit.getFrame().equals(targetFrame)) {
            return orbit;
        }
        return new CartesianOrbit(state.getPVCoordinates(targetFrame), targetFrame, state.getDate(), orbit.getMu());
    }

    private Orbit getOrbitInFrame(final Orbit orbit, final Frame targetFrame, final AbsoluteDate date) {
        if (targetFrame == null || orbit.getFrame().equals(targetFrame)) {
            return orbit;
        }
        return new CartesianOrbit(orbit.getPVCoordinates(targetFrame), targetFrame, date, orbit.getMu());
    }

    private void ensureCovered(final double[] dtSeconds) {
        double lo = Double.POSITIVE_INFINITY;
        double hi = Double.NEGATIVE_INFINITY;

        for (double t : dtSeconds) {
            if (!Double.isFinite(t)) {
                throw new IllegalArgumentException("query times must be finite");
            }
            if (t < lo) {
                lo = t;
            }
            if (t > hi) {
                hi = t;
            }
        }

        if (dtSeconds.length == 0 || !cacheEnabled) {
            return;
        }

        ensureCoveredRange(lo, hi);
    }

    private void ensureCoveredRange(double lo, double hi) {
        if (!Double.isFinite(lo) || !Double.isFinite(hi)) {
            throw new IllegalArgumentException("coverage bounds must be finite");
        }

        if (!cacheEnabled) {
            return;
        }

        if (hi < lo) {
            final double tmp = lo;
            lo = hi;
            hi = tmp;
        }

        if ((hi - lo) < MIN_WINDOW_SECONDS) {
            hi = lo + MIN_WINDOW_SECONDS;
        }

        if (!hasEphemeris) {
            rebuildEphemeris(lo, hi);
            tMinSeconds = lo;
            tMaxSeconds = hi;
            hasEphemeris = true;
            return;
        }

        if (lo >= tMinSeconds && hi <= tMaxSeconds) {
            return;
        }

        final double newLo = Math.min(lo, tMinSeconds);
        final double newHi = Math.max(hi, tMaxSeconds);

        rebuildEphemeris(newLo, newHi);
        tMinSeconds = newLo;
        tMaxSeconds = newHi;
    }

    private void rebuildEphemeris(final double lo, final double hi) {
        final SpacecraftState previousInitial = propagator.getInitialState();

        try {
            final EphemerisGenerator generator = propagator.getEphemerisGenerator();
            final AbsoluteDate tLo = epoch.shiftedBy(lo);
            final AbsoluteDate tHi = epoch.shiftedBy(hi);

            if (tLo.compareTo(tHi) <= 0) {
                propagator.propagate(tLo);
                propagator.propagate(tHi);
            } else {
                propagator.propagate(tHi);
                propagator.propagate(tLo);
            }

            ephemeris = generator.getGeneratedEphemeris();
        } finally {
            propagator.resetInitialState(previousInitial);
        }
    }

    private void clearEphemerisCache() {
        ephemeris = null;
        hasEphemeris = false;
        tMinSeconds = 0.0;
        tMaxSeconds = 0.0;
    }

    private static void applyDefaultAttitudeProvider(final Propagator propagator, final Frame inertialFrame) {
        if (propagator == null || inertialFrame == null) {
            return;
        }
        propagator.setAttitudeProvider(new LofOffset(inertialFrame, LOFType.VVLH));
    }

    private static RotationOrder parseRotationOrder(final String sequence) {
        if (sequence == null) {
            throw new IllegalArgumentException("attitudeEulerSequence must not be null");
        }
        final String normalized = sequence.trim().toUpperCase();
        try {
            return RotationOrder.valueOf(normalized);
        } catch (IllegalArgumentException exc) {
            throw new IllegalArgumentException(
                    "Unsupported Euler sequence: '" + sequence + "'. Use one of the Hipparchus RotationOrder names.",
                    exc);
        }
    }

    private static double angleMaybeDegrees(final double angleRad, final boolean degrees) {
        return degrees ? Math.toDegrees(angleRad) : angleRad;
    }

    private static double getKeplerianAnomaly(final KeplerianOrbit orbit, final PositionAngleType type) {
        if (type == PositionAngleType.TRUE) {
            return orbit.getTrueAnomaly();
        }
        if (type == PositionAngleType.ECCENTRIC) {
            return orbit.getEccentricAnomaly();
        }
        return orbit.getMeanAnomaly();
    }

    private static double getEquinoctialLongitude(final EquinoctialOrbit orbit, final PositionAngleType type) {
        if (type == PositionAngleType.TRUE) {
            return orbit.getLv();
        }
        if (type == PositionAngleType.ECCENTRIC) {
            return orbit.getLE();
        }
        return orbit.getLM();
    }

    private static String[] getDataDictionaryKeys(final DataDictionary dictionary) {
        final List<String> keys = new ArrayList<>();
        for (DataDictionary.Entry entry : dictionary.getData()) {
            keys.add(entry.getKey());
        }
        return keys.toArray(new String[0]);
    }

    private static String[] getDoubleArrayDictionaryKeys(final DoubleArrayDictionary dictionary) {
        final List<String> keys = new ArrayList<>();
        for (DoubleArrayDictionary.Entry entry : dictionary.getData()) {
            keys.add(entry.getKey());
        }
        return keys.toArray(new String[0]);
    }

    private static void appendAdditionalSeries(
            final AdditionalSeries[] series,
            final SpacecraftState state,
            final int sampleIndex,
            final int totalSamples,
            final boolean strict,
            final boolean derivative) {

        if (series == null) {
            return;
        }

        for (AdditionalSeries item : series) {
            if (!item.active) {
                continue;
            }

            final boolean available =
                    derivative
                            ? state.hasAdditionalStateDerivative(item.name)
                            : state.hasAdditionalData(item.name);

            if (!available) {
                if (strict) {
                    throw new IllegalArgumentException(
                            "Requested additional "
                                    + (derivative ? "state derivative" : "state")
                                    + " '"
                                    + item.name
                                    + "' is unavailable");
                }
                if (item.width < 0) {
                    item.disable();
                }
                continue;
            }

            final Object rawValue =
                    derivative
                            ? state.getAdditionalStateDerivative(item.name)
                            : state.getAdditionalData(item.name);

            final double[] values;
            try {
                values = coerceAdditionalValues(rawValue);
            } catch (Exception exc) {
                if (strict) {
                    throw new IllegalArgumentException(
                            "Failed to extract additional "
                                    + (derivative ? "state derivative" : "state")
                                    + " '"
                                    + item.name
                                    + "'",
                            exc);
                }
                if (item.width < 0) {
                    item.disable();
                }
                continue;
            }

            if (item.width < 0) {
                item.initialize(values.length, totalSamples);
            } else if (item.width != values.length) {
                if (strict) {
                    throw new IllegalArgumentException(
                            "Additional "
                                    + (derivative ? "state derivative" : "state")
                                    + " '"
                                    + item.name
                                    + "' changed width between samples");
                }
                item.fillNaN(sampleIndex);
                continue;
            }

            item.put(sampleIndex, values);
        }
    }

    private static double[] coerceAdditionalValues(final Object rawValue) {
        if (rawValue == null) {
            throw new IllegalArgumentException("additional state value is null");
        }
        if (rawValue instanceof double[]) {
            return (double[]) rawValue;
        }
        if (rawValue instanceof Double[]) {
            final Double[] boxed = (Double[]) rawValue;
            final double[] out = new double[boxed.length];
            for (int i = 0; i < boxed.length; i++) {
                out[i] = boxed[i];
            }
            return out;
        }
        if (rawValue instanceof Number) {
            return new double[] {((Number) rawValue).doubleValue()};
        }
        if (rawValue instanceof List<?>) {
            final List<?> list = (List<?>) rawValue;
            final double[] out = new double[list.size()];
            for (int i = 0; i < list.size(); i++) {
                final Object item = list.get(i);
                if (!(item instanceof Number)) {
                    throw new IllegalArgumentException("additional state list must contain only numbers");
                }
                out[i] = ((Number) item).doubleValue();
            }
            return out;
        }
        throw new IllegalArgumentException(
                "Unsupported additional state value type: " + rawValue.getClass().getName());
    }

    private static String[] packNames(final AdditionalSeries[] series) {
        if (series == null || series.length == 0) {
            return new String[0];
        }

        final List<String> names = new ArrayList<>();
        for (AdditionalSeries item : series) {
            if (item.isPacked()) {
                names.add(item.name);
            }
        }
        return names.toArray(new String[0]);
    }

    private static double[][] packValues(final AdditionalSeries[] series) {
        if (series == null || series.length == 0) {
            return new double[0][];
        }

        final List<double[]> values = new ArrayList<>();
        for (AdditionalSeries item : series) {
            if (item.isPacked()) {
                values.add(item.values);
            }
        }
        return values.toArray(new double[0][]);
    }

    private static int[] packWidths(final AdditionalSeries[] series) {
        if (series == null || series.length == 0) {
            return new int[0];
        }

        final List<Integer> widths = new ArrayList<>();
        for (AdditionalSeries item : series) {
            if (item.isPacked()) {
                widths.add(item.width);
            }
        }

        final int[] out = new int[widths.size()];
        for (int i = 0; i < widths.size(); i++) {
            out[i] = widths.get(i);
        }
        return out;
    }

    private static void copyQuaternion(
            final Rotation rotation,
            final double[] out,
            final int offset,
            final boolean scalarLast) {
        if (scalarLast) {
            out[offset] = rotation.getQ1();
            out[offset + 1] = rotation.getQ2();
            out[offset + 2] = rotation.getQ3();
            out[offset + 3] = rotation.getQ0();
            return;
        }

        out[offset] = rotation.getQ0();
        out[offset + 1] = rotation.getQ1();
        out[offset + 2] = rotation.getQ2();
        out[offset + 3] = rotation.getQ3();
    }

    private static void copyMatrix(final Rotation rotation, final double[] out, final int offset) {
        final double[][] matrix = rotation.getMatrix();
        int flatIndex = offset;
        for (int row = 0; row < 3; row++) {
            for (int col = 0; col < 3; col++) {
                out[flatIndex] = matrix[row][col];
                flatIndex += 1;
            }
        }
    }

    private static void copyVector(final Vector3D vec, final double[] out, final int offset) {
        out[offset] = vec.getX();
        out[offset + 1] = vec.getY();
        out[offset + 2] = vec.getZ();
    }

    private static void copyVectorOrZero(final Vector3D vec, final double[] out, final int offset) {
        if (vec == null) {
            out[offset] = 0.0;
            out[offset + 1] = 0.0;
            out[offset + 2] = 0.0;
            return;
        }
        copyVector(vec, out, offset);
    }

    private static void fillNaN(final double[] out, final int offset, final int width) {
        for (int i = 0; i < width; i++) {
            out[offset + i] = Double.NaN;
        }
    }

    private static final class AdditionalSeries {
        private final String name;
        private boolean active;
        private int width;
        private double[] values;

        private AdditionalSeries(final String name) {
            this.name = name;
            this.active = true;
            this.width = -1;
            this.values = null;
        }

        private static AdditionalSeries[] build(final String[] names) {
            if (names == null || names.length == 0) {
                return null;
            }
            final AdditionalSeries[] out = new AdditionalSeries[names.length];
            for (int i = 0; i < names.length; i++) {
                out[i] = new AdditionalSeries(names[i]);
            }
            return out;
        }

        private void initialize(final int width, final int totalSamples) {
            this.width = width;
            this.values = new double[width * totalSamples];
            Arrays.fill(this.values, Double.NaN);
        }

        private void disable() {
            this.active = false;
            this.width = 0;
            this.values = null;
        }

        private void put(final int sampleIndex, final double[] sampleValues) {
            System.arraycopy(sampleValues, 0, this.values, sampleIndex * this.width, this.width);
        }

        private void fillNaN(final int sampleIndex) {
            if (this.values == null || this.width <= 0) {
                return;
            }
            Arrays.fill(this.values, sampleIndex * this.width, (sampleIndex + 1) * this.width, Double.NaN);
        }

        private boolean isPacked() {
            return this.width > 0 && this.values != null;
        }
    }

    private static final class StateBatchResult {
        private final double[] p;
        private final double[] v;
        private final double[] a;

        private StateBatchResult(final double[] p, final double[] v, final double[] a) {
            this.p = p;
            this.v = v;
            this.a = a;
        }
    }

    public static final class PVBatchResult {
        public final double[] p;
        public final double[] v;

        public PVBatchResult(final double[] p, final double[] v) {
            this.p = p;
            this.v = v;
        }
    }

    public static final class PVABatchResult {
        public final double[] p;
        public final double[] v;
        public final double[] a;

        public PVABatchResult(final double[] p, final double[] v, final double[] a) {
            this.p = p;
            this.v = v;
            this.a = a;
        }
    }

    public static final class GeodeticBatchResult {
        public final double[] latDeg;
        public final double[] lonDeg;
        public final double[] altM;

        public GeodeticBatchResult(final double[] latDeg, final double[] lonDeg, final double[] altM) {
            this.latDeg = latDeg;
            this.lonDeg = lonDeg;
            this.altM = altM;
        }
    }

    public static final class SampleBatchResult {
        public final double[] positionM;
        public final double[] velocityMps;
        public final double[] accelerationMps2;

        public final double[] attitudeQuatRefToBody;
        public final double[] attitudeMatrixRefToBody;
        public final double[] attitudeEulerRefToBody;
        public final double[] attitudeSpinBodyRadS;
        public final double[] attitudeAccelBodyRadS2;

        public final double[] semiMajorAxisM;
        public final double[] eccentricity;
        public final double[] inclination;
        public final double[] raan;
        public final double[] argp;
        public final double[] anomaly;

        public final double[] equinoctialAM;
        public final double[] equinoctialEx;
        public final double[] equinoctialEy;
        public final double[] equinoctialHx;
        public final double[] equinoctialHy;
        public final double[] equinoctialLongitude;

        public final double[] massKg;

        public final String[] additionalNames;
        public final double[][] additionalValues;
        public final int[] additionalWidths;

        public final String[] additionalDerivativeNames;
        public final double[][] additionalDerivativeValues;
        public final int[] additionalDerivativeWidths;

        public SampleBatchResult(
                final double[] positionM,
                final double[] velocityMps,
                final double[] accelerationMps2,
                final double[] attitudeQuatRefToBody,
                final double[] attitudeMatrixRefToBody,
                final double[] attitudeEulerRefToBody,
                final double[] attitudeSpinBodyRadS,
                final double[] attitudeAccelBodyRadS2,
                final double[] semiMajorAxisM,
                final double[] eccentricity,
                final double[] inclination,
                final double[] raan,
                final double[] argp,
                final double[] anomaly,
                final double[] equinoctialAM,
                final double[] equinoctialEx,
                final double[] equinoctialEy,
                final double[] equinoctialHx,
                final double[] equinoctialHy,
                final double[] equinoctialLongitude,
                final double[] massKg,
                final String[] additionalNames,
                final double[][] additionalValues,
                final int[] additionalWidths,
                final String[] additionalDerivativeNames,
                final double[][] additionalDerivativeValues,
                final int[] additionalDerivativeWidths) {
            this.positionM = positionM;
            this.velocityMps = velocityMps;
            this.accelerationMps2 = accelerationMps2;
            this.attitudeQuatRefToBody = attitudeQuatRefToBody;
            this.attitudeMatrixRefToBody = attitudeMatrixRefToBody;
            this.attitudeEulerRefToBody = attitudeEulerRefToBody;
            this.attitudeSpinBodyRadS = attitudeSpinBodyRadS;
            this.attitudeAccelBodyRadS2 = attitudeAccelBodyRadS2;
            this.semiMajorAxisM = semiMajorAxisM;
            this.eccentricity = eccentricity;
            this.inclination = inclination;
            this.raan = raan;
            this.argp = argp;
            this.anomaly = anomaly;
            this.equinoctialAM = equinoctialAM;
            this.equinoctialEx = equinoctialEx;
            this.equinoctialEy = equinoctialEy;
            this.equinoctialHx = equinoctialHx;
            this.equinoctialHy = equinoctialHy;
            this.equinoctialLongitude = equinoctialLongitude;
            this.massKg = massKg;
            this.additionalNames = additionalNames;
            this.additionalValues = additionalValues;
            this.additionalWidths = additionalWidths;
            this.additionalDerivativeNames = additionalDerivativeNames;
            this.additionalDerivativeValues = additionalDerivativeValues;
            this.additionalDerivativeWidths = additionalDerivativeWidths;
        }
    }
}
