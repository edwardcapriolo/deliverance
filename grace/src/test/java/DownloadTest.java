import io.teknek.deliverance.grace.AutoTokenizer;
import io.teknek.deliverance.grace.PreTrainedTokenizer;
import io.teknek.deliverance.grace.TokenIds;
import io.teknek.deliverance.safetensors.fetch.ModelFetcher;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class DownloadTest {

    @Test
    void downloadTest(){
        PreTrainedTokenizer l = AutoTokenizer.fromPretrained(new AutoTokenizer
                .OwnerNameOrPath(new AutoTokenizer.OwnerName("Qwen", "Qwen2.5-7B-Instruct")));
        assertEquals(151643, l.getVocabSize());
    }

    @Test
    void seeHowItDoes(){
        int[] x = new int [] { 8586, 374, 279, 8102, 2082, 430, 20628, 279, 29803, 1473, 74694, 10248, 198, 1757, 6533, 32953, 74, 48045, 7201, 401, 898, 3834, 23342, 341, 262, 2033, 3158, 545, 633, 898, 538, 21918, 2289, 23342, 341, 262, 879, 2033, 10801, 401, 262, 586, 21918, 9078, 10801, 8, 341, 286, 420, 35194, 284, 10801, 280, 262, 557, 262, 571, 2226, 198, 262, 586, 2033, 3158, 368, 341, 286, 471, 4242, 21010, 353, 10801, 353, 10801, 280, 262, 457, 534, 14196, 19884, 2028, 2082, 19170, 264, 1595, 12581, 63, 3834, 449, 264, 3254, 1749, 1595, 4903, 368, 7964, 902, 4780, 264, 1595, 4429, 63, 907, 13, 1102, 1101, 19170, 264, 1595, 26264, 63, 538, 430, 2289, 279, 1595, 12581, 63, 3834, 323, 5280, 279, 1595, 4903, 55358, 1749, 13, 578, 1595, 26264, 63, 538, 706, 264, 879, 1595, 27813, 63, 2115, 323, 264, 4797, 430, 5097, 264, 1595, 27813, 63, 5852, 13, 578, 1595, 4903, 55358, 1749, 4780, 279, 3158, 315, 279, 12960, 1701, 279, 15150, 1595, 49345, 81, 61, 17, 29687, 128009};
        String modelName = "Llama-3.2-3B-Instruct-JQ4";
        String modelOwner = "tjake";
        PreTrainedTokenizer l = AutoTokenizer.fromPretrained(new AutoTokenizer
                .OwnerNameOrPath(new AutoTokenizer.OwnerName(modelOwner, modelName)));
        Assertions.assertEquals("""
Here is the Java code that meets the specifications:

```java
package io.teknek.shape;

public interface Shape {
    double area();
}

public class Circle extends Shape {
    private double radius;

    public Circle(double radius) {
        this.radius = radius;
    }

    @Override
    public double area() {
        return Math.PI * radius * radius;
    }
}
```

This code defines a `Shape` interface with a single method `area()`, which returns a `double` value. It also defines a `Circle` class that extends the `Shape` interface and implements the `area()` method. The `Circle` class has a private `radius` field and a constructor that takes a `radius` parameter. The `area()` method returns the area of the circle using the formula `πr^2`.""", l.decode(new TokenIds(x), true, false, false, false));

    }
}
