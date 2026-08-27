package io.teknek.deliverance.tensor;

import com.google.common.base.Preconditions;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;
import net.jafama.FastMath;

import java.util.*;

public class VectorTensorMathUtils {
    private static final VectorSpecies<Float> FLOAT_SPECIES = FloatVector.SPECIES_PREFERRED;

    public static void softMax(AbstractTensor x, int offset, int length) {
        Preconditions.checkArgument(x.shape().first() == 1);
        long size = offset + length;

        // find max value (for numerical stability)
        float max_val = x.get(0, offset);
        for (int i = offset + 1; i < size; i++) {
            if (x.get(0, i) > max_val) {
                max_val = x.get(0, i);
            }
        }
        // exp and sum
        float sum = 0.0f;
        for (int i = offset; i < size; i++) {
            x.set((float) FastMath.exp(x.get(0, i) - max_val), 0, i);
            sum += x.get(0, i);
        }
        normalizeSoftmaxRow(x, offset, length, sum);
    }

    public static void scaledSoftMax(AbstractTensor x, int offset, int length, float scale, Float softcap) {
        Preconditions.checkArgument(x.shape().first() == 1);
        long size = offset + length;

        float maxVal = transformForAttentionSoftmax(x.get(0, offset), scale, softcap);
        for (int i = offset + 1; i < size; i++) {
            float v = transformForAttentionSoftmax(x.get(0, i), scale, softcap);
            if (v > maxVal) {
                maxVal = v;
            }
        }
        float sum = 0.0f;
        for (int i = offset; i < size; i++) {
            float v = transformForAttentionSoftmax(x.get(0, i), scale, softcap);
            x.set((float) FastMath.exp(v - maxVal), 0, i);
            sum += x.get(0, i);
        }
        normalizeSoftmaxRow(x, offset, length, sum);
    }

    private static void normalizeSoftmaxRow(AbstractTensor x, int offset, int length, float sum) {
        float invSum = 1.0f / sum;
        if (x instanceof FloatBufferTensor floatTensor) {
            int upper = offset + FLOAT_SPECIES.loopBound(length);
            FloatVector inv = FloatVector.broadcast(FLOAT_SPECIES, invSum);
            for (int i = offset; i < upper; i += FLOAT_SPECIES.length()) {
                floatTensor.intoTensor(floatTensor.getVector(FLOAT_SPECIES, 0, i).mul(inv), 0, i);
            }
            for (int i = upper; i < offset + length; i++) {
                x.set(x.get(0, i) * invSum, 0, i);
            }
            return;
        }
        for (int i = offset; i < offset + length; i++) {
            x.set(x.get(0, i) * invSum, 0, i);
        }
    }

    private static float transformForAttentionSoftmax(float value, float scale, Float softcap) {
        float scaled = value * scale;
        if (softcap == null) {
            return scaled;
        }
        return (float) FastMath.tanh(scaled / softcap) * softcap;
    }

    public static void l2normalize(AbstractTensor x) {
        float sum = 0.0f;
        for (int i = 0; i < x.shape().last(); i++) {
            float v = x.get(0, i);
            sum += v * v;
        }
        double magnitude = FastMath.sqrt(sum);
        for (int i = 0; i < x.shape().last(); i++)
            x.set((float) (x.get(0, i) / magnitude), 0, i);
    }

    public static void logSumExpTensor(AbstractTensor result, AbstractTensor input) {
        float logsumexp = (float) logSumExp(input);
        for (int i = 0; i < input.size(); i++) {
            float v = input.get(0, i);
            result.set(v - logsumexp, 0, i);
        }
    }

    //https://nhigham.com/2021/01/05/what-is-the-log-sum-exp-function/
    // TODO: This is not the numerically stable log-sum-exp form. It should subtract max(x) before exponentiating and
    // add max(x) back after the log; otherwise large positive logits can overflow and large negative logits can
    // underflow. Use a provider-backed max/exp path when this is updated for hot inference code.
    public static double logSumExp(AbstractTensor x){
        float sum = 0.0f;
        for (int i = 0; i < x.size(); i++) {
            sum += (float) FastMath.exp(x.get(0, i));
        }
        return (float) FastMath.log(sum);
    }

    public static int percentile(SortedMap<Float, List<Integer>> valueBuckets, float perc, long size) {
        int element = (int) ((size * perc) - 1);
        Iterator<Map.Entry<Float, List<Integer>>> iter = valueBuckets.entrySet().iterator();
        int ct = 0;
        while (iter.hasNext()) {
            Map.Entry<Float, List<Integer>> entry = iter.next();
            ct += entry.getValue().size();
            //This condition returns a slightly higher percentile then requested as we are not doing
            //and exact count inside the bucket
            if (ct >= element) {
                System.out.println("arrived at element "+ ct);
                //System.out.println(entry.getValue());
                return entry.getValue().get(0); //could be a random one here
            }
        }
        return -1;
    }

    public static SortedMap<Float, List<Integer>> valueBuckets(AbstractTensor x) {
        SortedMap<Float, List<Integer>> buckets = new TreeMap<>();
        for (int i = 0; i < x.size(); i++) {
            float v = x.get(0, i);
            if (buckets.containsKey(v)) {
                buckets.get(v).add(i);
            } else {
                ArrayList<Integer> al = new ArrayList<>();
                al.add(i);
                buckets.put(v, al);
            }
        }
        return buckets;
    }

    public static void normalize(AbstractTensor t){
        double sum = 0.0;
        for (int i = 0; i < t.shape().last(); i++) {
            sum += t.get(0, i);
        }
        for (int i = 0; i < t.shape().last(); i++) {
            t.set((float) (t.get(0, i) / sum), 0, i);
        }
    }
    /*
        public static double[] normalize(double[] input) {
        double sum = 0;
        for (double p : input) sum += p;

        double[] normalized = new double[input.length];
        for (int i = 0; i < input.length; i++) {
            normalized[i] = input[i] / sum;
        }
        return normalized;
    }
     */
}
