import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.VectorSpecies;

public class Basic {
    public static int[] addTwoScalarArrays(int[] arr1, int[] arr2) {
        int[] result = new int[arr1.length];
        for(int i = 0; i< arr1.length; i++) {
            result[i] = arr1[i] + arr2[i];
        }
        return result;
    }

    public static int[] addTwoVectorArrays(int[] arr1, int[] arr2) {
        VectorSpecies<Integer> SPECIES = IntVector.SPECIES_PREFERRED;
        System.out.println(SPECIES);
        IntVector v1 = IntVector.fromArray(SPECIES, arr1, 0);
        System.out.println(v1);
        IntVector v2 = IntVector.fromArray(SPECIES, arr2, 0);
        System.out.println(v2);
        IntVector result = v1.add(v2);
        return result.toArray();
    }
}
