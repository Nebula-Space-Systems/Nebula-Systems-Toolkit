package com.nstk.propagation;

import org.hipparchus.geometry.euclidean.threed.Rotation;
import org.hipparchus.geometry.euclidean.threed.Vector3D;
import org.orekit.attitudes.AttitudeProvider;
import org.orekit.attitudes.LofOffset;
import org.orekit.bodies.GeodeticPoint;
import org.orekit.bodies.OneAxisEllipsoid;
import org.orekit.frames.Frame;
import org.orekit.frames.LOFType;
import org.orekit.orbits.KeplerianOrbit;
import org.orekit.orbits.PositionAngleType;
import org.orekit.propagation.BoundedPropagator;
import org.orekit.propagation.EphemerisGenerator;
import org.orekit.propagation.Propagator;
import org.orekit.propagation.SpacecraftState;
import org.orekit.propagation.analytical.EcksteinHechlerPropagator;
import org.orekit.propagation.analytical.KeplerianPropagator;
import org.orekit.time.AbsoluteDate;
import org.orekit.utils.Constants;
import org.orekit.utils.PVCoordinates;

/**
 * Java-first orbit engine for the separate orbit propagation interface.
 *
 * <p>All propagation loops, ephemeris interpolation, frame transforms, geodetic conversion,
 * and attitude extraction are executed in Java.
 */
public final class OrekitOrbitPropagationBridge {

    private static final double MIN_WINDOW_SECONDS = 1.0e-6;

    private final Propagator propagator;
    private final AbsoluteDate epoch;
    private final Frame nativeFrame;

    private BoundedPropagator ephemeris;
    private boolean hasEphemeris;
    private double tMinSeconds;
    private double tMaxSeconds;

    public OrekitOrbitPropagationBridge(final Propagator propagator) {
        if (propagator == null) {
            throw new IllegalArgumentException("propagator must not be null");
        }
        this.propagator = propagator;

        final SpacecraftState state0 = propagator.getInitialState();
        this.epoch = state0.getDate();
        this.nativeFrame = state0.getFrame();

        this.ephemeris = null;
        this.hasEphemeris = false;
        this.tMinSeconds = 0.0;
        this.tMaxSeconds = 0.0;
    }

    public static OrekitOrbitPropagationBridge fromPropagator(final Propagator propagator) {
        return new OrekitOrbitPropagationBridge(propagator);
    }

    public static OrekitOrbitPropagationBridge fromSpacecraftState(final SpacecraftState state) {
        if (state == null) {
            throw new IllegalArgumentException("state must not be null");
        }

        final KeplerianPropagator propagator = new KeplerianPropagator(state.getOrbit());
        propagator.resetInitialState(state);
        applyDefaultAttitudeProvider(propagator, state.getFrame());

        return new OrekitOrbitPropagationBridge(propagator);
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

        return new OrekitOrbitPropagationBridge(propagator);
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

        return new OrekitOrbitPropagationBridge(propagator);
    }

    public Propagator getPropagator() {
        return propagator;
    }

    public Frame getNativeFrame() {
        return nativeFrame;
    }

    public synchronized void setAttitudeProvider(final AttitudeProvider provider) {
        if (provider == null) {
            throw new IllegalArgumentException("provider must not be null");
        }
        propagator.setAttitudeProvider(provider);
        clearEphemerisCache();
    }

    public synchronized double[] coverage() {
        if (!hasEphemeris) {
            return new double[] {0.0, 0.0};
        }
        return new double[] {tMinSeconds, tMaxSeconds};
    }

    public synchronized void precompute(final double tMinSeconds, final double tMaxSeconds) {
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

        ensureCovered(dtSeconds);
        final Frame bodyFrame = ellipsoid.getBodyFrame();

        for (int i = 0; i < n; i++) {
            final AbsoluteDate absT = epoch.shiftedBy(dtSeconds[i]);
            final SpacecraftState state = ephemeris.propagate(absT);
            final Vector3D pos = state.getPVCoordinates(bodyFrame).getPosition();
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
            final SpacecraftState state = ephemeris.propagate(absT);
            final Rotation rot = state.getAttitude().getRotation();

            out[off] = rot.getQ0();
            out[off + 1] = rot.getQ1();
            out[off + 2] = rot.getQ2();
            out[off + 3] = rot.getQ3();
        }

        return out;
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

        ensureCovered(dtSeconds);
        final Frame target = outputFrame == null ? nativeFrame : outputFrame;

        for (int i = 0; i < n; i++) {
            final int off = 3 * i;
            final AbsoluteDate absT = epoch.shiftedBy(dtSeconds[i]);
            final SpacecraftState state = ephemeris.propagate(absT);
            final PVCoordinates pv = state.getPVCoordinates(target);

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

        ensureCoveredRange(lo, hi);
    }

    private void ensureCoveredRange(double lo, double hi) {
        if (!Double.isFinite(lo) || !Double.isFinite(hi)) {
            throw new IllegalArgumentException("coverage bounds must be finite");
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
        propagator.setAttitudeProvider(new LofOffset(inertialFrame, LOFType.LVLH_CCSDS));
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
}




