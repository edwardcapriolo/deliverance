import io.teknek.reckoning.tensor.AbstractTensor;
import io.teknek.reckoning.tensor.TensorShape;
import io.teknek.sketches.ReckoningKernel;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

public class ReckoningKernelTest {

    @Test
    void basicKernelTests(){
        ReckoningKernel dk = new ReckoningKernel();
        AbstractTensor t = dk.allocateTokenBitmask( 90);
        //TODO later when we throw a real tokenizer/model here we can make a better assert
        Assertions.assertEquals( TensorShape.of(1,3), t.shape());
        Assertions.assertEquals(2, t.shape().dims());
        Assertions.assertEquals(-1f, t.get(0, 2));
        t.close();
    }
}
