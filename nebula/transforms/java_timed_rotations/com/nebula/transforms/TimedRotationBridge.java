package com.nebula.transforms;

import org.hipparchus.geometry.euclidean.threed.Vector3D;
import org.orekit.frames.Frame;
import org.orekit.frames.Transform;
import org.orekit.time.AbsoluteDate;
import org.orekit.utils.PVCoordinates;

/** Java-first timed frame transformation bridge for cartesian state vectors. */
public final class TimedRotationBridge {

    private static final Vector3D ZERO = Vector3D.ZERO;

    private TimedRotationBridge() {
        // Static utility class.
    }

    public static PVAResult transformAtOffsets(
            final Frame fromFrame,
            final Frame toFrame,
            final AbsoluteDate epoch,
            final double[] dtSeconds,
            final double[] positions,
            final double[] velocities,
            final double[] accelerations) {

        if (epoch == null) {
            throw new IllegalArgumentException("epoch must not be null");
        }
        if (dtSeconds == null) {
            throw new IllegalArgumentException("dtSeconds must not be null");
        }

        final int n = dtSeconds.length;
        validateInputs(fromFrame, toFrame, n, positions, velocities, accelerations);
        return transformFromOffsets(fromFrame, toFrame, epoch, dtSeconds, positions, velocities, accelerations);
    }

    public static PVAResult transformAtDates(
            final Frame fromFrame,
            final Frame toFrame,
            final AbsoluteDate[] dates,
            final double[] positions,
            final double[] velocities,
            final double[] accelerations) {

        if (dates == null) {
            throw new IllegalArgumentException("dates must not be null");
        }

        final int n = dates.length;
        validateInputs(fromFrame, toFrame, n, positions, velocities, accelerations);
        return transformFromDates(fromFrame, toFrame, dates, positions, velocities, accelerations);
    }

    private static void validateInputs(
            final Frame fromFrame,
            final Frame toFrame,
            final int n,
            final double[] positions,
            final double[] velocities,
            final double[] accelerations) {

        if (fromFrame == null) {
            throw new IllegalArgumentException("fromFrame must not be null");
        }
        if (toFrame == null) {
            throw new IllegalArgumentException("toFrame must not be null");
        }
        if (positions == null) {
            throw new IllegalArgumentException("positions must not be null");
        }
        if (positions.length != 3 * n) {
            throw new IllegalArgumentException("positions must have length 3 * n");
        }
        if (velocities != null && velocities.length != 3 * n) {
            throw new IllegalArgumentException("velocities must have length 3 * n");
        }
        if (accelerations != null && accelerations.length != 3 * n) {
            throw new IllegalArgumentException("accelerations must have length 3 * n");
        }
    }

    private static PVAResult transformFromOffsets(
            final Frame fromFrame,
            final Frame toFrame,
            final AbsoluteDate epoch,
            final double[] dtSeconds,
            final double[] positions,
            final double[] velocities,
            final double[] accelerations) {

        final int n = dtSeconds.length;
        final double[] outP = new double[3 * n];
        final double[] outV = velocities != null ? new double[3 * n] : null;
        final double[] outA = accelerations != null ? new double[3 * n] : null;

        if (n == 0) {
            return new PVAResult(outP, outV, outA);
        }

        double previousDt = Double.NaN;
        Transform transform = null;

        for (int i = 0; i < n; i++) {
            final double dt = dtSeconds[i];
            if (!Double.isFinite(dt)) {
                throw new IllegalArgumentException("All dtSeconds values must be finite");
            }

            if (i == 0 || dt != previousDt) {
                final AbsoluteDate date = epoch.shiftedBy(dt);
                transform = fromFrame.getTransformTo(toFrame, date);
                previousDt = dt;
            }

            final int off = 3 * i;
            applyOne(transform, positions, velocities, accelerations, off, outP, outV, outA);
        }

        return new PVAResult(outP, outV, outA);
    }

    private static PVAResult transformFromDates(
            final Frame fromFrame,
            final Frame toFrame,
            final AbsoluteDate[] dates,
            final double[] positions,
            final double[] velocities,
            final double[] accelerations) {

        final int n = dates.length;
        final double[] outP = new double[3 * n];
        final double[] outV = velocities != null ? new double[3 * n] : null;
        final double[] outA = accelerations != null ? new double[3 * n] : null;

        if (n == 0) {
            return new PVAResult(outP, outV, outA);
        }

        AbsoluteDate previousDate = null;
        Transform transform = null;

        for (int i = 0; i < n; i++) {
            final AbsoluteDate date = dates[i];
            if (date == null) {
                throw new IllegalArgumentException("dates must not contain null entries");
            }

            if (previousDate == null || date.compareTo(previousDate) != 0) {
                transform = fromFrame.getTransformTo(toFrame, date);
                previousDate = date;
            }

            final int off = 3 * i;
            applyOne(transform, positions, velocities, accelerations, off, outP, outV, outA);
        }

        return new PVAResult(outP, outV, outA);
    }

    private static void applyOne(
            final Transform transform,
            final double[] positions,
            final double[] velocities,
            final double[] accelerations,
            final int off,
            final double[] outP,
            final double[] outV,
            final double[] outA) {

        final Vector3D pIn = new Vector3D(positions[off], positions[off + 1], positions[off + 2]);

        if (velocities == null && accelerations == null) {
            final Vector3D pOut = transform.transformPosition(pIn);
            copyVector(pOut, outP, off);
            return;
        }

        final Vector3D vIn =
                velocities != null
                        ? new Vector3D(velocities[off], velocities[off + 1], velocities[off + 2])
                        : ZERO;

        final PVCoordinates pvOut;
        if (accelerations != null) {
            final Vector3D aIn = new Vector3D(accelerations[off], accelerations[off + 1], accelerations[off + 2]);
            pvOut = transform.transformPVCoordinates(new PVCoordinates(pIn, vIn, aIn));
        } else {
            pvOut = transform.transformPVCoordinates(new PVCoordinates(pIn, vIn));
        }

        copyVector(pvOut.getPosition(), outP, off);
        if (outV != null) {
            copyVector(pvOut.getVelocity(), outV, off);
        }
        if (outA != null) {
            final Vector3D a = pvOut.getAcceleration();
            if (a == null) {
                outA[off] = 0.0;
                outA[off + 1] = 0.0;
                outA[off + 2] = 0.0;
            } else {
                copyVector(a, outA, off);
            }
        }
    }

    private static void copyVector(final Vector3D vec, final double[] out, final int offset) {
        out[offset] = vec.getX();
        out[offset + 1] = vec.getY();
        out[offset + 2] = vec.getZ();
    }

    public static final class PVAResult {
        public final double[] p;
        public final double[] v;
        public final double[] a;

        public PVAResult(final double[] p, final double[] v, final double[] a) {
            this.p = p;
            this.v = v;
            this.a = a;
        }
    }
}
