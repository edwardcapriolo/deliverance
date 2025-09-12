import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.VectorSpecies;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

public class BasicTests {
    @Test
    void doIt(){
        int[] a = {1, 2, 3, 4,5,6,7,8};
        int[] b = {1,2,3,4,5,6,7,8};
        int[] c = {2,4,6,8,10,12,14,16};
        Assertions.assertArrayEquals(c, Basic.addTwoVectorArrays(a,b));
    }

}
