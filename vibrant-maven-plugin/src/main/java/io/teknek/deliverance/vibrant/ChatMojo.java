package io.teknek.deliverance.vibrant;

import io.teknek.deliverance.DType;
import io.teknek.deliverance.generator.GeneratorParameters;
import io.teknek.deliverance.generator.Response;
import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.AutoModelForCausaLm;
import io.teknek.deliverance.safetensors.fetch.ModelFetcher;
import io.teknek.deliverance.safetensors.prompt.PromptSupport;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;
import io.teknek.deliverance.tensor.operations.NativeSimdTensorOperations;
import org.apache.maven.plugin.AbstractMojo;
import org.apache.maven.plugin.MojoExecutionException;
import org.apache.maven.plugin.MojoFailureException;
import org.apache.maven.plugins.annotations.LifecyclePhase;
import org.apache.maven.plugins.annotations.Mojo;
import org.apache.maven.plugins.annotations.Parameter;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.List;
import java.util.UUID;

@Mojo(name = "chat", defaultPhase = LifecyclePhase.COMPILE)
public class ChatMojo extends AbstractMojo {
    @Parameter(name="modelConfig")
    private ModelConfig modelConfig;

    public void execute() throws MojoExecutionException, MojoFailureException {
        if (modelConfig == null) {
            modelConfig = new ModelConfig();
        }
        ModelFetcher fetch = new ModelFetcher(modelConfig.getOwner(), modelConfig.getModelName());
        DType working = DType.valueOf(modelConfig.getWorkingMemType());
        DType quantized = DType.valueOf(modelConfig.getQuantizedMemType());
        AutoModelForCausaLm.Builder builder = AutoModelForCausaLm.newBuilder(fetch);
        WrappedForkJoinPool pool = new WrappedForkJoinPool(WrappedForkJoinPool.autoSizeByCores());
        NativeSimdTensorOperations operation = new NativeSimdTensorOperations(new ConfigurableTensorProvider(builder.getCache(), pool).get());
        builder.withWorkingQuantType(working);
        builder.withWorkingQuantType(quantized);
        builder.withTensorProvider(new ConfigurableTensorProvider(operation));

        try (AbstractModel model = builder.build()) {
            System.out.println("Chat with deliverance! Type 'undeliver' to quit.");
            System.out.print(">> ");
            BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
            try {
                String inputLine;
                while (!(inputLine = br.readLine()).equalsIgnoreCase("undeliver")) {
                    PromptSupport.Builder b = model.promptSupport().get().builder();
                    b.addUserMessage(inputLine);
                    var uuid = UUID.randomUUID();
                    Response k = model.generate(uuid, b.build(), new GeneratorParameters()

                            .withIncludeStopStrInOutput(false)
                            .withStopWords(List.of("<|eot_id|>"))
                            .withTemperature(0.2f)
                            .withSeed(99998), (int next, String tok, String s1, float f1) -> {
                        System.out.print( model.getTokenRenderer().tokenizerToRendered(s1));
                    });
                    System.out.println();
                    System.out.print(">> ");
                }
            } catch (IOException e) {
                throw new MojoExecutionException(e);
            }
        }
    }


}
