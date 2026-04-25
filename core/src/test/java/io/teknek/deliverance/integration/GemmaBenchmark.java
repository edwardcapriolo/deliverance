package io.teknek.deliverance.integration;

/*
import io.teknek.deliverance.generator.GeneratorParameters;
import io.teknek.deliverance.generator.Response;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.DoNothingGenerateEvent;
import io.teknek.deliverance.safetensors.prompt.PromptSupport;
import org.openjdk.jmh.annotations.*;

import java.io.IOException;
import java.util.UUID;

@State(Scope.Benchmark)
@Fork(value = 1)
@Warmup(iterations = 2)
public class GemmaBenchmark {

    private AbstractModel model;
    @Setup
    public void setup(){
        model = Gemma2Suite.getOrCreate();
    }

    @Benchmark
    public Response generate(){
        String prompt = "Pick a random number between 1 and 9.";
        PromptSupport.Builder g = model.promptSupport().get().builder()
                .addUserMessage(prompt);
        var uuid = UUID.randomUUID();

        Response k = model.generate(uuid, g.build(), new GeneratorParameters()
                        .withTemperature(0.0f)
                        .withSeed(41)
                        .withMaxTokens(300)
                        .withLogProbs(true)
                        .withTopLogProbs(10)
                        .withXtcThreshold(0.1f)
                        .withXtcProbability(0.5f)
                , new DoNothingGenerateEvent());
        return k;
    }

    public static void main (String [] args) throws IOException {
        org.openjdk.jmh.Main.main(args);
    }
}*/
