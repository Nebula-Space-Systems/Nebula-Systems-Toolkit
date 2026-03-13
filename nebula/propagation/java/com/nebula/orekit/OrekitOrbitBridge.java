package com.nebula.orekit;

import java.util.Arrays;

import org.hipparchus.geometry.euclidean.threed.Vector3D;
import org.orekit.frames.Frame;
import org.orekit.frames.Transform;
import org.orekit.propagation.BoundedPropagator;
import org.orekit.propagation.EphemerisGenerator;
import org.orekit.propagation.Propagator;
import org.orekit.propagation.SpacecraftState;
import org.orekit.time.AbsoluteDate;
import org.orekit.utils.PVCoordinates;

/**
 * Java-side precision orbit engine for high-throughput state queries.
 *
 * <p>Design notes:
 *
 * <ul>
 *   <li>Caches propagation-frame samples; optional output-frame cache for speed.
 *   <li>Transforms to output frame at query time unless cached-frame interpolation is enabled.
 *   <li>All per-sample loops stay in Java to minimize Python/JVM crossings.
 * </ul>
 */
public final class OrekitOrbitBridge {

    private static final int CACHE_MARGIN = 64;

    private final Propagator propagator;
    private final AbsoluteDate epoch;
    private final Frame nativeFrame;
    private final Frame cachedFrame;
    private final boolean cacheFrameSamples;
    private final double dt;
    private final boolean useQuintic;

    private SpacecraftState firstState;
    private SpacecraftState lastState;

    private int kMin;
    private int kMax;
    private int nSamples;

    private double[] rSamples;
    private double[] vSamples;
    private double[] aSamples;
    private double[] rFrameSamples;
    private double[] vFrameSamples;
    private double[] aFrameSamples;

    public OrekitOrbitBridge(
            final Propagator propagator,
            final double dt,
            final boolean useQuintic) {
        this(propagator, null, dt, useQuintic, false);
    }

    public OrekitOrbitBridge(
            final Propagator propagator,
            final Frame cachedFrame,
            final double dt,
            final boolean useQuintic,
            final boolean cacheFrameSamples) {
        if (propagator == null) {
            throw new IllegalArgumentException("propagator must not be null");
        }
        if (!Double.isFinite(dt) || dt <= 0.0) {
            throw new IllegalArgumentException("dt must be finite and > 0");
        }
        this.propagator = propagator;
        this.cachedFrame = cachedFrame;
        this.cacheFrameSamples = cacheFrameSamples && cachedFrame != null;
        this.dt = dt;
        this.useQuintic = useQuintic;

        final SpacecraftState state0 = propagator.getInitialState();
        this.firstState = state0;
        this.lastState = state0;
        this.epoch = state0.getDate();
        this.nativeFrame = state0.getFrame();

        this.kMin = 0;
        this.kMax = 0;
        this.nSamples = 1;

        final int cap = 1 + CACHE_MARGIN;
        this.rSamples = new double[3 * cap];
        this.vSamples = new double[3 * cap];
        this.aSamples = useQuintic ? new double[3 * cap] : null;
        this.rFrameSamples = this.cacheFrameSamples ? new double[3 * cap] : null;
        this.vFrameSamples = this.cacheFrameSamples ? new double[3 * cap] : null;
        this.aFrameSamples = (this.cacheFrameSamples && useQuintic) ? new double[3 * cap] : null;

        final PVCoordinates pv0 = state0.getPVCoordinates(nativeFrame);
        copyVector(pv0.getPosition(), this.rSamples, 0);
        copyVector(pv0.getVelocity(), this.vSamples, 0);
        if (useQuintic) {
            copyAcceleration(pv0, this.aSamples, 0);
        }

        if (this.cacheFrameSamples) {
            final PVCoordinates pvf = state0.getPVCoordinates(this.cachedFrame);
            copyVector(pvf.getPosition(), this.rFrameSamples, 0);
            copyVector(pvf.getVelocity(), this.vFrameSamples, 0);
            if (useQuintic) {
                copyAcceleration(pvf, this.aFrameSamples, 0);
            }
        }
    }

    public double[] coverage() {
        return new double[] {(double) kMin * dt, (double) kMax * dt};
    }

    public void precompute(final double tMinSeconds, final double tMaxSeconds) {
        ensureCovered(new double[] {tMinSeconds, tMaxSeconds});
    }

    public double[] queryPosition(final double[] dtSeconds, final Frame outputFrame) {
        return queryState(dtSeconds, outputFrame, true, false, false).r;
    }

    public double[] queryVelocity(final double[] dtSeconds, final Frame outputFrame) {
        return queryState(dtSeconds, outputFrame, false, true, false).v;
    }

    public double[] queryAcceleration(final double[] dtSeconds, final Frame outputFrame) {
        return queryState(dtSeconds, outputFrame, false, false, true).a;
    }

    public PVBatchResult queryPV(final double[] dtSeconds, final Frame outputFrame) {
        final StateBatchResult out = queryState(dtSeconds, outputFrame, true, true, false);
        return new PVBatchResult(out.r, out.v);
    }

    public PVABatchResult queryPVA(final double[] dtSeconds, final Frame outputFrame) {
        final StateBatchResult out = queryState(dtSeconds, outputFrame, true, true, true);
        return new PVABatchResult(out.r, out.v, out.a);
    }

    private StateBatchResult queryState(
            final double[] dtSeconds,
            final Frame outputFrame,
            final boolean needPosition,
            final boolean needVelocity,
            final boolean needAcceleration) {
        final int n = dtSeconds.length;
        final double[] rOut = needPosition ? new double[3 * n] : null;
        final double[] vOut = needVelocity ? new double[3 * n] : null;
        final double[] aOut = needAcceleration ? new double[3 * n] : null;

        if (n == 0) {
            return new StateBatchResult(rOut, vOut, aOut);
        }

        ensureCovered(dtSeconds);

        final Frame target = outputFrame == null ? nativeFrame : outputFrame;
        final boolean same = sameFrame(target, nativeFrame);
        final boolean useCachedFrame = cacheFrameSamples && sameFrame(target, cachedFrame);

        final double[] rSrc = useCachedFrame ? rFrameSamples : rSamples;
        final double[] vSrc = useCachedFrame ? vFrameSamples : vSamples;
        final double[] aSrc = useCachedFrame ? aFrameSamples : aSamples;

        final double[] rTmp = new double[3];
        final double[] vTmp = new double[3];
        final double[] aTmp = needAcceleration ? new double[3] : null;
        final boolean singleSample = nSamples < 2;
        final double singleSampleTime = (double) kMin * dt;

        for (int i = 0; i < n; i++) {
            final int off = 3 * i;
            final double tq = dtSeconds[i];

            if (singleSample) {
                if (Math.abs(tq - singleSampleTime) > 1e-9) {
                    throw new IllegalStateException("Cache has insufficient samples for interpolation");
                }
                copyAt(rSrc, 0, rTmp, 0);
                copyAt(vSrc, 0, vTmp, 0);
                if (needAcceleration && aTmp != null) {
                    if (useQuintic && aSrc != null) {
                        copyAt(aSrc, 0, aTmp, 0);
                    } else {
                        aTmp[0] = 0.0;
                        aTmp[1] = 0.0;
                        aTmp[2] = 0.0;
                    }
                }
            } else if (needAcceleration) {
                interpolateOnePVA(
                        tq,
                        kMin,
                        dt,
                        nSamples,
                        rSrc,
                        vSrc,
                        aSrc,
                        useQuintic,
                        rTmp,
                        vTmp,
                        aTmp,
                        0);
            } else if (needVelocity) {
                interpolateOne(
                        tq,
                        kMin,
                        dt,
                        nSamples,
                        rSrc,
                        vSrc,
                        aSrc,
                        useQuintic,
                        rTmp,
                        vTmp,
                        0);
            } else {
                interpolateOneR(
                        tq,
                        kMin,
                        dt,
                        nSamples,
                        rSrc,
                        vSrc,
                        aSrc,
                        useQuintic,
                        rTmp,
                        0);
            }

            if (same || useCachedFrame) {
                if (needPosition && rOut != null) {
                    copyAt(rTmp, 0, rOut, off);
                }
                if (needVelocity && vOut != null) {
                    copyAt(vTmp, 0, vOut, off);
                }
                if (needAcceleration && aOut != null && aTmp != null) {
                    copyAt(aTmp, 0, aOut, off);
                }
                continue;
            }

            final AbsoluteDate absT = epoch.shiftedBy(tq);
            final Transform tr = nativeFrame.getTransformTo(target, absT);

            if (needPosition && !needVelocity && !needAcceleration) {
                final Vector3D pOut = tr.transformPosition(new Vector3D(rTmp[0], rTmp[1], rTmp[2]));
                copyVector(pOut, rOut, off);
                continue;
            }

            final PVCoordinates inPv;
            if (needAcceleration && aTmp != null) {
                inPv =
                        new PVCoordinates(
                                new Vector3D(rTmp[0], rTmp[1], rTmp[2]),
                                new Vector3D(vTmp[0], vTmp[1], vTmp[2]),
                                new Vector3D(aTmp[0], aTmp[1], aTmp[2]));
            } else {
                inPv =
                        new PVCoordinates(
                                new Vector3D(rTmp[0], rTmp[1], rTmp[2]),
                                new Vector3D(vTmp[0], vTmp[1], vTmp[2]));
            }
            final PVCoordinates pvOut = tr.transformPVCoordinates(inPv);
            if (needPosition && rOut != null) {
                copyVector(pvOut.getPosition(), rOut, off);
            }
            if (needVelocity && vOut != null) {
                copyVector(pvOut.getVelocity(), vOut, off);
            }
            if (needAcceleration && aOut != null) {
                copyVectorOrZero(pvOut.getAcceleration(), aOut, off);
            }
        }

        return new StateBatchResult(rOut, vOut, aOut);
    }

    private void ensureCovered(final double[] dtSeconds) {
        double lo = Double.POSITIVE_INFINITY;
        double hi = Double.NEGATIVE_INFINITY;
        for (double t : dtSeconds) {
            if (!Double.isFinite(t)) {
                throw new IllegalArgumentException("Non-finite query times are not supported");
            }
            if (t < lo) {
                lo = t;
            }
            if (t > hi) {
                hi = t;
            }
        }

        final int kNeedLo = (int) Math.floor(lo / dt);
        final int kNeedHi = (int) Math.ceil(hi / dt);

        if (kNeedLo < kMin) {
            extendBackwardTo(kNeedLo);
        }
        if (kNeedHi > kMax) {
            extendForwardTo(kNeedHi);
        }
    }

    private EphemerisGenerator newEphemerisGenerator() {
        try {
            // Available on numerical propagators; keep optional for generic Propagator support.
            propagator.getClass().getMethod("clearEphemerisGenerators").invoke(propagator);
        } catch (Exception ignored) {
            // best effort
        }
        return propagator.getEphemerisGenerator();
    }

    private void extendForwardTo(final int kTarget) {
        if (kTarget <= kMax) {
            return;
        }

        final SpacecraftState prevInit = propagator.getInitialState();
        try {
            propagator.resetInitialState(lastState);

            final EphemerisGenerator gen = newEphemerisGenerator();
            final AbsoluteDate targetDate = epoch.shiftedBy((double) kTarget * dt);
            final SpacecraftState newLastState = propagator.propagate(targetDate);
            final BoundedPropagator ephem = gen.getGeneratedEphemeris();

            final int nNew = kTarget - kMax;
            final double[] rNew = new double[3 * nNew];
            final double[] vNew = new double[3 * nNew];
            final double[] aNew = useQuintic ? new double[3 * nNew] : null;
            final double[] rFrameNew = cacheFrameSamples ? new double[3 * nNew] : null;
            final double[] vFrameNew = cacheFrameSamples ? new double[3 * nNew] : null;
            final double[] aFrameNew = (cacheFrameSamples && useQuintic) ? new double[3 * nNew] : null;

            for (int j = 0; j < nNew; j++) {
                final int k = kMax + 1 + j;
                final int off = 3 * j;
                final SpacecraftState st = ephem.propagate(epoch.shiftedBy((double) k * dt));
                final PVCoordinates pv = st.getPVCoordinates(nativeFrame);
                copyVector(pv.getPosition(), rNew, off);
                copyVector(pv.getVelocity(), vNew, off);
                if (useQuintic) {
                    copyAcceleration(pv, aNew, off);
                }
                if (cacheFrameSamples) {
                    final PVCoordinates pvf = st.getPVCoordinates(cachedFrame);
                    copyVector(pvf.getPosition(), rFrameNew, off);
                    copyVector(pvf.getVelocity(), vFrameNew, off);
                    if (useQuintic) {
                        copyAcceleration(pvf, aFrameNew, off);
                    }
                }
            }

            appendSamples(rNew, vNew, aNew, rFrameNew, vFrameNew, aFrameNew, nNew);
            kMax = kTarget;
            lastState = newLastState;
        } finally {
            propagator.resetInitialState(prevInit);
        }
    }

    private void extendBackwardTo(final int kTarget) {
        if (kTarget >= kMin) {
            return;
        }

        final SpacecraftState prevInit = propagator.getInitialState();
        try {
            propagator.resetInitialState(firstState);

            final EphemerisGenerator gen = newEphemerisGenerator();
            final AbsoluteDate targetDate = epoch.shiftedBy((double) kTarget * dt);
            final SpacecraftState newFirstState = propagator.propagate(targetDate);
            final BoundedPropagator ephem = gen.getGeneratedEphemeris();

            final int nNew = kMin - kTarget;
            final double[] rNew = new double[3 * nNew];
            final double[] vNew = new double[3 * nNew];
            final double[] aNew = useQuintic ? new double[3 * nNew] : null;
            final double[] rFrameNew = cacheFrameSamples ? new double[3 * nNew] : null;
            final double[] vFrameNew = cacheFrameSamples ? new double[3 * nNew] : null;
            final double[] aFrameNew = (cacheFrameSamples && useQuintic) ? new double[3 * nNew] : null;

            for (int j = 0; j < nNew; j++) {
                final int k = kTarget + j;
                final int off = 3 * j;
                final SpacecraftState st = ephem.propagate(epoch.shiftedBy((double) k * dt));
                final PVCoordinates pv = st.getPVCoordinates(nativeFrame);
                copyVector(pv.getPosition(), rNew, off);
                copyVector(pv.getVelocity(), vNew, off);
                if (useQuintic) {
                    copyAcceleration(pv, aNew, off);
                }
                if (cacheFrameSamples) {
                    final PVCoordinates pvf = st.getPVCoordinates(cachedFrame);
                    copyVector(pvf.getPosition(), rFrameNew, off);
                    copyVector(pvf.getVelocity(), vFrameNew, off);
                    if (useQuintic) {
                        copyAcceleration(pvf, aFrameNew, off);
                    }
                }
            }

            prependSamples(rNew, vNew, aNew, rFrameNew, vFrameNew, aFrameNew, nNew);
            kMin = kTarget;
            firstState = newFirstState;
        } finally {
            propagator.resetInitialState(prevInit);
        }
    }

    private void appendSamples(
            final double[] rNew,
            final double[] vNew,
            final double[] aNew,
            final double[] rFrameNew,
            final double[] vFrameNew,
            final double[] aFrameNew,
            final int nNew) {
        final int oldN = nSamples;
        final int newN = oldN + nNew;
        ensureCapacity(newN);

        System.arraycopy(rNew, 0, rSamples, 3 * oldN, 3 * nNew);
        System.arraycopy(vNew, 0, vSamples, 3 * oldN, 3 * nNew);
        if (useQuintic && aNew != null) {
            System.arraycopy(aNew, 0, aSamples, 3 * oldN, 3 * nNew);
        }
        if (cacheFrameSamples && rFrameNew != null && vFrameNew != null) {
            System.arraycopy(rFrameNew, 0, rFrameSamples, 3 * oldN, 3 * nNew);
            System.arraycopy(vFrameNew, 0, vFrameSamples, 3 * oldN, 3 * nNew);
            if (useQuintic && aFrameNew != null) {
                System.arraycopy(aFrameNew, 0, aFrameSamples, 3 * oldN, 3 * nNew);
            }
        }
        nSamples = newN;
    }

    private void prependSamples(
            final double[] rNew,
            final double[] vNew,
            final double[] aNew,
            final double[] rFrameNew,
            final double[] vFrameNew,
            final double[] aFrameNew,
            final int nNew) {
        final int oldN = nSamples;
        final int newN = oldN + nNew;
        final int oldCap = rSamples.length / 3;
        final int newCap = Math.max(newN + CACHE_MARGIN, oldCap);

        final double[] rAll = new double[3 * newCap];
        final double[] vAll = new double[3 * newCap];
        System.arraycopy(rNew, 0, rAll, 0, 3 * nNew);
        System.arraycopy(vNew, 0, vAll, 0, 3 * nNew);
        System.arraycopy(rSamples, 0, rAll, 3 * nNew, 3 * oldN);
        System.arraycopy(vSamples, 0, vAll, 3 * nNew, 3 * oldN);
        rSamples = rAll;
        vSamples = vAll;

        if (useQuintic) {
            final double[] aAll = new double[3 * newCap];
            if (aNew != null) {
                System.arraycopy(aNew, 0, aAll, 0, 3 * nNew);
            }
            System.arraycopy(aSamples, 0, aAll, 3 * nNew, 3 * oldN);
            aSamples = aAll;
        }
        if (cacheFrameSamples) {
            final double[] rAllF = new double[3 * newCap];
            final double[] vAllF = new double[3 * newCap];
            if (rFrameNew != null) {
                System.arraycopy(rFrameNew, 0, rAllF, 0, 3 * nNew);
            }
            if (vFrameNew != null) {
                System.arraycopy(vFrameNew, 0, vAllF, 0, 3 * nNew);
            }
            System.arraycopy(rFrameSamples, 0, rAllF, 3 * nNew, 3 * oldN);
            System.arraycopy(vFrameSamples, 0, vAllF, 3 * nNew, 3 * oldN);
            rFrameSamples = rAllF;
            vFrameSamples = vAllF;

            if (useQuintic) {
                final double[] aAllF = new double[3 * newCap];
                if (aFrameNew != null) {
                    System.arraycopy(aFrameNew, 0, aAllF, 0, 3 * nNew);
                }
                System.arraycopy(aFrameSamples, 0, aAllF, 3 * nNew, 3 * oldN);
                aFrameSamples = aAllF;
            }
        }

        nSamples = newN;
    }

    private void ensureCapacity(final int requiredSamples) {
        final int oldCap = rSamples.length / 3;
        if (requiredSamples <= oldCap) {
            return;
        }
        final int growth = Math.max(CACHE_MARGIN, oldCap / 2);
        final int newCap = Math.max(requiredSamples + CACHE_MARGIN, oldCap + growth);
        rSamples = Arrays.copyOf(rSamples, 3 * newCap);
        vSamples = Arrays.copyOf(vSamples, 3 * newCap);
        if (useQuintic) {
            aSamples = Arrays.copyOf(aSamples, 3 * newCap);
        }
        if (cacheFrameSamples) {
            rFrameSamples = Arrays.copyOf(rFrameSamples, 3 * newCap);
            vFrameSamples = Arrays.copyOf(vFrameSamples, 3 * newCap);
            if (useQuintic) {
                aFrameSamples = Arrays.copyOf(aFrameSamples, 3 * newCap);
            }
        }
    }

    private static void interpolateOneR(
            final double tq,
            final int kMin,
            final double h,
            final int nSamples,
            final double[] rFlat,
            final double[] vFlat,
            final double[] aFlat,
            final boolean useQuintic,
            final double[] rOut,
            final int outOffset) {
        long k = (long) Math.floor(tq / h);
        final long kMaxCell = (long) kMin + nSamples - 2L;
        if (k < kMin) {
            k = kMin;
        } else if (k > kMaxCell) {
            k = kMaxCell;
        }
        final int i0 = (int) (k - kMin);
        final int i1 = i0 + 1;
        final int j0 = 3 * i0;
        final int j1 = 3 * i1;

        final double tk = k * h;
        final double u = (tq - tk) / h;
        final double u2 = u * u;
        final double u3 = u2 * u;

        if (!useQuintic) {
            final double h00 = 2.0 * u3 - 3.0 * u2 + 1.0;
            final double h10 = u3 - 2.0 * u2 + u;
            final double h01 = -2.0 * u3 + 3.0 * u2;
            final double h11 = u3 - u2;

            for (int c = 0; c < 3; c++) {
                final double r0 = rFlat[j0 + c];
                final double r1 = rFlat[j1 + c];
                final double v0 = vFlat[j0 + c];
                final double v1 = vFlat[j1 + c];
                rOut[outOffset + c] = h00 * r0 + h10 * (h * v0) + h01 * r1 + h11 * (h * v1);
            }
            return;
        }

        final double u4 = u2 * u2;
        final double u5 = u4 * u;
        final double h2 = h * h;

        final double h00 = 1.0 - 10.0 * u3 + 15.0 * u4 - 6.0 * u5;
        final double h10 = u - 6.0 * u3 + 8.0 * u4 - 3.0 * u5;
        final double h20 = 0.5 * u2 - 1.5 * u3 + 1.5 * u4 - 0.5 * u5;
        final double h01 = 10.0 * u3 - 15.0 * u4 + 6.0 * u5;
        final double h11 = -4.0 * u3 + 7.0 * u4 - 3.0 * u5;
        final double h21 = 0.5 * u3 - u4 + 0.5 * u5;

        for (int c = 0; c < 3; c++) {
            final double r0 = rFlat[j0 + c];
            final double r1 = rFlat[j1 + c];
            final double v0 = vFlat[j0 + c];
            final double v1 = vFlat[j1 + c];
            final double a0 = aFlat[j0 + c];
            final double a1 = aFlat[j1 + c];

            rOut[outOffset + c] =
                    h00 * r0
                            + h10 * (h * v0)
                            + h20 * (h2 * a0)
                            + h01 * r1
                            + h11 * (h * v1)
                            + h21 * (h2 * a1);
        }
    }

    private static void interpolateOne(
            final double tq,
            final int kMin,
            final double h,
            final int nSamples,
            final double[] rFlat,
            final double[] vFlat,
            final double[] aFlat,
            final boolean useQuintic,
            final double[] rOut,
            final double[] vOut,
            final int outOffset) {
        long k = (long) Math.floor(tq / h);
        final long kMaxCell = (long) kMin + nSamples - 2L;
        if (k < kMin) {
            k = kMin;
        } else if (k > kMaxCell) {
            k = kMaxCell;
        }
        final int i0 = (int) (k - kMin);
        final int i1 = i0 + 1;
        final int j0 = 3 * i0;
        final int j1 = 3 * i1;

        final double tk = k * h;
        final double u = (tq - tk) / h;
        final double u2 = u * u;
        final double u3 = u2 * u;

        if (!useQuintic) {
            final double h00 = 2.0 * u3 - 3.0 * u2 + 1.0;
            final double h10 = u3 - 2.0 * u2 + u;
            final double h01 = -2.0 * u3 + 3.0 * u2;
            final double h11 = u3 - u2;

            final double dh00 = 6.0 * u2 - 6.0 * u;
            final double dh10 = 3.0 * u2 - 4.0 * u + 1.0;
            final double dh01 = -6.0 * u2 + 6.0 * u;
            final double dh11 = 3.0 * u2 - 2.0 * u;

            for (int c = 0; c < 3; c++) {
                final double r0 = rFlat[j0 + c];
                final double r1 = rFlat[j1 + c];
                final double v0 = vFlat[j0 + c];
                final double v1 = vFlat[j1 + c];

                rOut[outOffset + c] = h00 * r0 + h10 * (h * v0) + h01 * r1 + h11 * (h * v1);
                vOut[outOffset + c] =
                        (dh00 * r0) / h + dh10 * v0 + (dh01 * r1) / h + dh11 * v1;
            }
            return;
        }

        final double u4 = u2 * u2;
        final double u5 = u4 * u;
        final double h2 = h * h;

        final double h00 = 1.0 - 10.0 * u3 + 15.0 * u4 - 6.0 * u5;
        final double h10 = u - 6.0 * u3 + 8.0 * u4 - 3.0 * u5;
        final double h20 = 0.5 * u2 - 1.5 * u3 + 1.5 * u4 - 0.5 * u5;
        final double h01 = 10.0 * u3 - 15.0 * u4 + 6.0 * u5;
        final double h11 = -4.0 * u3 + 7.0 * u4 - 3.0 * u5;
        final double h21 = 0.5 * u3 - u4 + 0.5 * u5;

        final double dh00 = -30.0 * u2 + 60.0 * u3 - 30.0 * u4;
        final double dh10 = 1.0 - 18.0 * u2 + 32.0 * u3 - 15.0 * u4;
        final double dh20 = u - 4.5 * u2 + 6.0 * u3 - 2.5 * u4;
        final double dh01 = 30.0 * u2 - 60.0 * u3 + 30.0 * u4;
        final double dh11 = -12.0 * u2 + 28.0 * u3 - 15.0 * u4;
        final double dh21 = 1.5 * u2 - 4.0 * u3 + 2.5 * u4;

        for (int c = 0; c < 3; c++) {
            final double r0 = rFlat[j0 + c];
            final double r1 = rFlat[j1 + c];
            final double v0 = vFlat[j0 + c];
            final double v1 = vFlat[j1 + c];
            final double a0 = aFlat[j0 + c];
            final double a1 = aFlat[j1 + c];

            rOut[outOffset + c] =
                    h00 * r0
                            + h10 * (h * v0)
                            + h20 * (h2 * a0)
                            + h01 * r1
                            + h11 * (h * v1)
                            + h21 * (h2 * a1);

            vOut[outOffset + c] =
                    (dh00 * r0) / h
                            + dh10 * v0
                            + dh20 * (h * a0)
                            + (dh01 * r1) / h
                            + dh11 * v1
                            + dh21 * (h * a1);
        }
    }

    private static void interpolateOnePVA(
            final double tq,
            final int kMin,
            final double h,
            final int nSamples,
            final double[] rFlat,
            final double[] vFlat,
            final double[] aFlat,
            final boolean useQuintic,
            final double[] rOut,
            final double[] vOut,
            final double[] aOut,
            final int outOffset) {
        interpolateOne(
                tq,
                kMin,
                h,
                nSamples,
                rFlat,
                vFlat,
                aFlat,
                useQuintic,
                rOut,
                vOut,
                outOffset);

        long k = (long) Math.floor(tq / h);
        final long kMaxCell = (long) kMin + nSamples - 2L;
        if (k < kMin) {
            k = kMin;
        } else if (k > kMaxCell) {
            k = kMaxCell;
        }
        final int i0 = (int) (k - kMin);
        final int i1 = i0 + 1;
        final int j0 = 3 * i0;
        final int j1 = 3 * i1;

        final double tk = k * h;
        final double u = (tq - tk) / h;
        final double u2 = u * u;
        final double u3 = u2 * u;

        if (!useQuintic) {
            final double d2h00 = 12.0 * u - 6.0;
            final double d2h10 = 6.0 * u - 4.0;
            final double d2h01 = -12.0 * u + 6.0;
            final double d2h11 = 6.0 * u - 2.0;

            for (int c = 0; c < 3; c++) {
                final double r0 = rFlat[j0 + c];
                final double r1 = rFlat[j1 + c];
                final double v0 = vFlat[j0 + c];
                final double v1 = vFlat[j1 + c];
                aOut[outOffset + c] =
                        (d2h00 * r0) / (h * h)
                                + (d2h10 * v0) / h
                                + (d2h01 * r1) / (h * h)
                                + (d2h11 * v1) / h;
            }
            return;
        }

        final double d2h00 = -60.0 * u + 180.0 * u2 - 120.0 * u3;
        final double d2h10 = -36.0 * u + 96.0 * u2 - 60.0 * u3;
        final double d2h20 = 1.0 - 9.0 * u + 18.0 * u2 - 10.0 * u3;
        final double d2h01 = 60.0 * u - 180.0 * u2 + 120.0 * u3;
        final double d2h11 = -24.0 * u + 84.0 * u2 - 60.0 * u3;
        final double d2h21 = 3.0 * u - 12.0 * u2 + 10.0 * u3;

        for (int c = 0; c < 3; c++) {
            final double r0 = rFlat[j0 + c];
            final double r1 = rFlat[j1 + c];
            final double v0 = vFlat[j0 + c];
            final double v1 = vFlat[j1 + c];
            final double a0 = aFlat[j0 + c];
            final double a1 = aFlat[j1 + c];

            aOut[outOffset + c] =
                    (d2h00 * r0) / (h * h)
                            + (d2h10 * v0) / h
                            + d2h20 * a0
                            + (d2h01 * r1) / (h * h)
                            + (d2h11 * v1) / h
                            + d2h21 * a1;
        }
    }

    private static boolean sameFrame(final Frame a, final Frame b) {
        if (a == b) {
            return true;
        }
        if (a == null || b == null) {
            return false;
        }
        try {
            return a.getName().equalsIgnoreCase(b.getName());
        } catch (Exception ex) {
            return false;
        }
    }

    private static void copyAt(
            final double[] src,
            final int srcOffset,
            final double[] dst,
            final int dstOffset) {
        dst[dstOffset] = src[srcOffset];
        dst[dstOffset + 1] = src[srcOffset + 1];
        dst[dstOffset + 2] = src[srcOffset + 2];
    }

    private static void copyVector(final Vector3D src, final double[] dst, final int offset) {
        dst[offset] = src.getX();
        dst[offset + 1] = src.getY();
        dst[offset + 2] = src.getZ();
    }

    private static void copyVectorOrZero(
            final Vector3D src,
            final double[] dst,
            final int offset) {
        if (src == null) {
            dst[offset] = 0.0;
            dst[offset + 1] = 0.0;
            dst[offset + 2] = 0.0;
            return;
        }
        copyVector(src, dst, offset);
    }

    private static void copyAcceleration(
            final PVCoordinates pv, final double[] dst, final int offset) {
        final Vector3D acc = pv.getAcceleration();
        if (acc == null) {
            dst[offset] = 0.0;
            dst[offset + 1] = 0.0;
            dst[offset + 2] = 0.0;
            return;
        }
        copyVector(acc, dst, offset);
    }

    private static final class StateBatchResult {
        final double[] r;
        final double[] v;
        final double[] a;

        StateBatchResult(final double[] r, final double[] v, final double[] a) {
            this.r = r;
            this.v = v;
            this.a = a;
        }
    }

    public static final class PVBatchResult {
        public final double[] r;
        public final double[] v;

        public PVBatchResult(final double[] r, final double[] v) {
            this.r = r;
            this.v = v;
        }
    }

    public static final class PVABatchResult {
        public final double[] r;
        public final double[] v;
        public final double[] a;

        public PVABatchResult(final double[] r, final double[] v, final double[] a) {
            this.r = r;
            this.v = v;
            this.a = a;
        }
    }
}

