package io.teknek.deliverance.tensor;

import com.google.common.base.Preconditions;
import net.jafama.FastMath;

import java.util.*;

public class VectorTensorMathUtils {
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
        // normalize
        for (int i = 0; i < size; i++) {
            x.set(x.get(0, i) / sum, 0, i);
        }
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
