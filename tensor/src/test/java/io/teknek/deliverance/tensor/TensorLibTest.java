package io.teknek.deliverance.tensor;

import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import io.teknek.deliverance.tensorlib.ReadOnlyTensorMap;
import io.teknek.deliverance.tensorlib.Reduce;
import io.teknek.deliverance.tensorlib.TensorLib;
import io.teknek.deliverance.tensorlib.TensorMap;
import io.teknek.dysfx.Maybe;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.List;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class TensorLibTest {

    static class MaxTensor implements TensorMap<Float> {

        @Override
        public Float map(AbstractTensor t1, long offset, long length) {
            float max = Float.NEGATIVE_INFINITY; //correct
            for (long i = offset; i < offset + length; i++) {
                float it = t1.get(0, (int)i);
                if (it > max) {
                    max = it;
                }
            }
            return max;
        }
    }

    static class MaxReadOnlyTensor implements ReadOnlyTensorMap<Float> {

        @Override
        public Float map(ReadableTensor t1, long offset, long length) {
            float max = Float.NEGATIVE_INFINITY; //correct
            for (long i = offset; i < offset + length; i++) {
                float it = t1.get(0, (int)i);
                if (it > max) {
                    max = it;
                }
            }
            return max;
        }
    }

    public AbstractTensor simpleTensor(){
        int rows =  1;
        int cols = 8;
        AbstractTensor original = new FloatBufferTensor(rows, cols);
        original.set(1.0f, 0, 0);
        original.set(2.0f, 0, 1);
        original.set(3.0f, 0, 2);
        original.set(-3.0f, 0, 3);
        original.set(3.0f, 0, 4);
        return original;
    }

    @Test
    void testTensorlibMapper(){
        AbstractTensor original = simpleTensor();
        TensorLib tensorLib =  new TensorLib(new WrappedForkJoinPool(WrappedForkJoinPool.autoSizeByCores()));
        List<Float> results = tensorLib.unary(original).mapper(new MaxTensor())
                .execute(0,8,4);
        assertEquals(Arrays.asList(2.0f, 3.0f, 3.0f, 0.0f), results);
        List<Long> results2 = tensorLib.unary(original).readOnlyMapper((t1, offset, length) -> {
            return t1.size();
        }).execute(0,8,4);
    }

    @Test
    void testTensorLibWithReduce(){
        Reduce<Float, Float> myReduce = t -> {
            Optional<Float> x = t.stream().max(Float::compareTo);
            return x.map(Maybe::possibly).orElseGet(Maybe::nothing);
        };
        AbstractTensor original = simpleTensor();
        TensorLib tensorLib =  new TensorLib(new WrappedForkJoinPool(WrappedForkJoinPool.autoSizeByCores()));
        Maybe<Float> result = tensorLib.unary(original).readOnlyMapper(new MaxReadOnlyTensor())
                .prepare(0,8,4).reduce(myReduce);
        assertEquals(Maybe.possibly(3.0f), result);
    }

    @Test
    void testMutableTensorLibWithReduce(){
        Reduce<Float, Float> myReduce = t -> {
            Optional<Float> x = t.stream().max(Float::compareTo);
            return x.map(Maybe::possibly).orElseGet(Maybe::nothing);
        };
        AbstractTensor original = simpleTensor();
        TensorLib tensorLib =  new TensorLib(new WrappedForkJoinPool(WrappedForkJoinPool.autoSizeByCores()));
        Maybe<Float> result = tensorLib.unary(original).mapper(new MaxTensor())
                .prepare(0,8,4).reduce(myReduce);
        assertEquals(Maybe.definitely(3.0f), result);
    }

}
