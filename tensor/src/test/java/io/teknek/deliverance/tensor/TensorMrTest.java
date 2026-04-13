package io.teknek.deliverance.tensor;


public class TensorMrTest {
/*
    @Test
    public void testMr(){
        int rows =  1;
        int cols = 10;
        AbstractTensor original = new FloatBufferTensor(rows, cols);
        original.set(1.0f, 0, 0);
        original.set(2.0f, 0, 1);
        original.set(3.0f, 0, 2);
        original.set(-3.0f, 0, 3);
        original.set(3.0f, 0, 4);
        TensorMr t = new TensorMr(new WrappedForkJoinPool(WrappedForkJoinPool.autoSizeByCores()));
        float value = t.exec((t1, start, end) -> {
            float max = Float.NEGATIVE_INFINITY;
            for (long i = start; i < end; i++) {
                float it = t1.get(0, (int)i);
                if (it > max) {
                    max = it;
                }
            }
            return max;
        }, t2 -> t2.stream().max(Float::compareTo).get(), 0, 10, 2, original);
        assertEquals(3.0f, value);
    }
*/
}
