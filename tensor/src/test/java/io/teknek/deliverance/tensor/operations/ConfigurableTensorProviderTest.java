package io.teknek.deliverance.tensor.operations;


import com.codahale.metrics.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.tensor.AbstractTensor;


import io.teknek.deliverance.tensor.TensorCacheIface;
import io.teknek.deliverance.tensor.TensorShape;
import org.junit.jupiter.api.Test;import org.mockito.Mock;import org.mockito.Mockito;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;

public class ConfigurableTensorProviderTest {

    @Test
    void defaultTest(){
        try(WrappedForkJoinPool pool = new WrappedForkJoinPool(WrappedForkJoinPool.autoSizeByCores());) {
            ConfigurableTensorProvider p = new ConfigurableTensorProvider(Mockito.mock(TensorCacheIface.class), pool);
            assertNotNull(p.get());
        }
    }

    @Test
    void customizableTest(){
        // go put mockito on the classpath and clean this up
        ConfigurableTensorProvider p = new ConfigurableTensorProvider(new TensorOperations() {
            @Override
            public String name() {
                return "edsbestimpl";
            }

            @Override
            public int parallelSplitSize() {
                return 0;
            }

            @Override
            public void batchDotProduct(AbstractTensor result, AbstractTensor a, AbstractTensor b, int aColumnOffset, int bColumnOffset, int columnLimit, int rRowOffset, int bRowOffset, int rowChunkSize) {

            }

            @Override
            public void accumulate(AbstractTensor a, AbstractTensor b, int offset, int length) {

            }

            @Override
            public void maccumulate(AbstractTensor a, AbstractTensor b, int offset, int length) {

            }

            @Override
            public void saxpy(float alpha, AbstractTensor x, AbstractTensor y, int xoffset, int yoffset, int limit) {

            }

            @Override
            public void scale(float factor, AbstractTensor x, int offset, int length) {

            }

            @Override
            public AbstractTensor quantize(AbstractTensor t, DType qtype, int offset, int length) {
                return null;
            }
        });
        assertEquals("edsbestimpl", p.get().name());
    }
}
