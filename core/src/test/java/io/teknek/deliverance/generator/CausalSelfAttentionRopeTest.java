package io.teknek.deliverance.generator;

import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.TensorDisplayUtil;
import io.teknek.deliverance.tensor.TensorTestSupport;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class CausalSelfAttentionRopeTest {

    @Test
    void rotateRopeHeadsRotatesEachHeadIndependently() {
        try (AbstractTensor tensor = TensorTestSupport.tensorOf(1, 8,
                1, 2, 3, 4,
                5, 6, 7, 8)) {
            float[][] rope = {
                    { 0.0f, 1.0f },
                    { 0.0f, 1.0f }
            };

            CausalSelfAttention.rotateRopeHeads(tensor, 2, 4, 2, 0, rope);

            assertEquals("""
                    [0][0]= -3.0000 [0][1]= -4.0000 [0][2]=  1.0000 [0][3]=  2.0000 [0][4]= -7.0000 [0][5]= -8.0000 [0][6]=  5.0000 [0][7]=  6.0000
                    """.trim(), TensorDisplayUtil.pretty2dDisplayAll(tensor).trim());
        }
    }
}
