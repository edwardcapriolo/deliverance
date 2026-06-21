package io.teknek.deliverance.integration;

import io.teknek.deliverance.generator.GeneratorParameters;
import io.teknek.deliverance.generator.Response;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.DoNothingGenerateEvent;
import io.teknek.deliverance.safetensors.prompt.PromptContext;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.UUID;

public class Llama32ThreeBPromptIT {

    @Test
    public void calc() {
        AbstractModel m = Llama32ThreeBSuite.getOrCreate();
        PromptContext ctx = m.promptSupport().get().builder()
                .addSystemMessage("You are an assistant that produces concise, production-grade software.")
                .addSystemMessage("Output java code.")
                .addSystemMessage("Refrain from editorializing your reply.")
                .addSystemMessage("Generate java code into the package 'io.teknek.shape' .")
                .addSystemMessage("Do not import java.awt")
                .addUserMessage("Generate a java interface named Shape with a method named area that returns a double.")
                .addUserMessage("Generate a java class named Circle that extends the Shape interface.")
                .build();

        Response k = m.generate(UUID.randomUUID(), ctx, new GeneratorParameters()
                .withNtokens(512)
                .withIncludeStopStrInOutput(false)
                .withStopWords(List.of("<|eot_id|>"))
                .withTemperature(0.2f).withSeed(99998), new DoNothingGenerateEvent());
/*
        assertEquals("""
```java
package io.teknek.shape;

public class Circle extends Shape {
    private double radius;

    public Circle(double radius) {
        this.radius = radius;
    }

    @Override
    public double area() {
        return Math.PI * Math.pow(radius, 2);
    }
}
```

Note: The `Shape` interface is assumed to be defined elsewhere in the codebase. If not, it can be defined as follows:

```java
package io.teknek.shape;

public interface Shape {
    double area();
}
```
""".trim(), k.responseText);

 */
        //assertTrue(k.responseText.contains("public interface Shape"));
    }

    @Test
    public void nanocodeRootPromptHiNoTools() {
        AbstractModel m = Llama32ThreeBSuite.getOrCreate();
        PromptContext ctx = m.promptSupport().get().builder()
                .addSystemMessage("You are a concise coding assistant. cwd: /tmp. Use tools when needed. Prefer small, direct changes.")
                .addUserMessage("hi")
                .build();

        Response response = m.generate(UUID.randomUUID(), ctx, new GeneratorParameters()
                .withNtokens(1024)
                .withMaxTokens(200)
                .withTemperature(0.0f)
                .withSeed(99999), new DoNothingGenerateEvent());
        System.out.println(response);
    }
}
