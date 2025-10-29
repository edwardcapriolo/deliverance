import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.VectorSpecies;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;

public class BasicTests {
    @Test
    void add128Test(){
        Assumptions.assumeTrue(IntVector.SPECIES_PREFERRED == IntVector.SPECIES_128);
        int[] a = {1, 2, 3, 4};
        int[] b = {1, 2, 3, 4};
        int[] c = {2, 4, 6, 8};
        Assertions.assertArrayEquals(c, Basic.addTwoVectorArrays(a,b));
    }
    @Test
    void add256Test(){
        Assumptions.assumeTrue(IntVector.SPECIES_PREFERRED == IntVector.SPECIES_256);
        int[] a = {1, 2, 3, 4, 5, 6, 7, 8};
        int[] b = {1, 2, 3, 4, 5, 6, 7, 8};
        int[] c = {2, 4, 6, 8, 10, 12, 14, 16};
        Assertions.assertArrayEquals(c, Basic.addTwoVectorArrays(a,b));
    }

}
