package io.teknek.deliverance.math;

import net.jafama.FastMath;

public class VectorMathUtils {

    public static float[] outerProduct(float[] xs, float[] ys) {
        int n = xs.length;
        int m = ys.length;
        float[] result = new float[n * m];
        int idx = 0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                result[idx++] = xs[i] * ys[j];
            }
        }
        return result;
    }

    public static float[][] precomputeFreqsCis(int dim, int end, double theta, double scaling_factor) {
        float[] freqs = new float[dim / 2];
        float step = 0.0f;
        for (int i = 0; i < freqs.length; i++, step += 2.0f) {
            freqs[i] = (float) ((1.0 / FastMath.pow(theta, step / dim)) / scaling_factor);
        }
        float[] t = new float[end];
        for (int i = 0; i < end; i++) {
            t[i] = i;
        }
        float[] freqs_cis = outerProduct(t, freqs);

        float[][] r = new float[freqs_cis.length][];
        for (int i = 0; i < freqs_cis.length; i++) {
            r[i] = new float[]{(float) FastMath.cos(freqs_cis[i]), (float) FastMath.sin(freqs_cis[i])};
        }
        return r;
    }

    public static float cosineSimilarity(float[] a, float[] b) {
        float dotProduct = 0.0f;
        float aMagnitude = 0.0f;
        float bMagnitude = 0.0f;
        for (int i = 0; i < a.length; i++) {
            dotProduct += a[i] * b[i];
            aMagnitude += a[i] * a[i];
            bMagnitude += b[i] * b[i];
        }
        return (float) (dotProduct / (FastMath.sqrt(aMagnitude) * FastMath.sqrt(bMagnitude)));
    }

    public static void l2normalize(float[] x) {
        float sum = 0.0f;
        for (int i = 0; i < x.length; i++)
            sum += x[i] * x[i];

        double magnitude = FastMath.sqrt(sum);
        for (int i = 0; i < x.length; i++)
            x[i] /= magnitude;
    }

    public static void l1normalize(float[] x) {
        float sum = 0.0f;
        for (int i = 0; i < x.length; i++) {
            sum += FastMath.abs(x[i]);
        }
        for (int i = 0; i < x.length; i++) {
            x[i] /= sum;
        }
    }
}
