package io.teknek.deliverance.vibrant;

import com.codahale.metrics.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.generator.GeneratorParameters;
import io.teknek.deliverance.generator.Response;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.AutoModelForCausaLm;
import io.teknek.deliverance.model.ModelSupport;
import io.teknek.deliverance.safetensors.fetch.ModelFetcher;
import io.teknek.deliverance.safetensors.prompt.PromptSupport;
import io.teknek.deliverance.tensor.KvBufferCacheSettings;
import io.teknek.deliverance.tensor.TensorCache;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;
import io.teknek.deliverance.tensor.operations.NativeSimdTensorOperations;
import org.apache.maven.plugin.AbstractMojo;
import org.apache.maven.plugin.MojoExecutionException;
import org.apache.maven.plugin.MojoFailureException;
import org.apache.maven.plugins.annotations.LifecyclePhase;
import org.apache.maven.plugins.annotations.Mojo;
import org.apache.maven.plugins.annotations.Parameter;
import org.apache.maven.project.MavenProject;

import java.io.File;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.List;
import java.util.UUID;

@Mojo(name = "generate", defaultPhase = LifecyclePhase.GENERATE_SOURCES)
public class VibrantMojo extends AbstractMojo {

    @Parameter(defaultValue = "${project}", required = true, readonly = true)
    MavenProject project;

    @Parameter(property = "outputDirectory", defaultValue = "${project.build.directory}/generated-sources/vibrant")
    private File outputDirectory;

    @Parameter
    private List<VibeSpec> vibeSpecs;

    @Parameter(name="modelConfig")
    private ModelConfig modelConfig;


    public void runSingleSpec(AbstractModel model, VibeSpec spec) throws MojoExecutionException {
        if (!spec.enabled) {
            return;
        }
        File specDir;
        if (spec.getGenerateTo() == null) {
            spec.setGenerateTo("generated-source");
        }
        if ("generated-source".equalsIgnoreCase(spec.getGenerateTo())) {
            boolean ignore = outputDirectory.mkdirs();
            specDir = new File(outputDirectory, spec.id);
            boolean ignore2 = specDir.mkdirs();
            project.addCompileSourceRoot(specDir.getAbsolutePath());
            getLog().info("Generated source directory: " + specDir.getAbsolutePath());
        } else if ("generated-test-source".equalsIgnoreCase(spec.getGenerateTo())) {
            boolean ignore = outputDirectory.mkdirs();
            specDir = new File(outputDirectory, spec.id);
            boolean ignore2 = specDir.mkdirs();
            project.addTestCompileSourceRoot(specDir.getAbsolutePath());
            getLog().info("Generated test-source directory: " + specDir.getAbsolutePath());
        } else if ("existing-source".equalsIgnoreCase(spec.getGenerateTo())) {
            List<String> compileSourceRoots = project.getCompileSourceRoots();
            specDir = new File(compileSourceRoots.get(0));
            getLog().info("Using source directory: " + specDir.getAbsolutePath());
        } else if ("existing-test-source".equalsIgnoreCase(spec.getGenerateTo())) {
            List<String> compileSourceRoots = project.getTestCompileSourceRoots();
            specDir = new File(compileSourceRoots.get(0));
            getLog().info("Using test-source directory: " + specDir.getAbsolutePath());
        } else {
            throw new MojoExecutionException("unsupported mode " + spec.getGenerateTo());
        }

        PromptSupport.Builder b = model.promptSupport().get().builder();
        spec.userMessages.forEach(b::addUserMessage);
        spec.systemMessages.forEach(b::addSystemMessage);
        var uuid = UUID.randomUUID();
        Response k = model.generate(uuid, b.build(), new GeneratorParameters()
                .withNtokens(512)
                .withIncludeStopStrInOutput(false)
                .withStopWords(List.of("<|eot_id|>"))
                .withTemperature(0.2f)
                .withSeed(99998), (int next, String tok, String s1, float f1) -> {
        });
        getLog().debug("inference engine reply: " + k.responseText);
        Path p = Path.of(specDir.toURI());
        Path child = p.resolve("raw.txt");
        MessageTransformer t;
        try {
            if (spec.getMessageTransformer() == null){
                spec.setMessageTransformer("io.teknek.deliverance.vibrant.JavaResponseTransformer");
            }
            t = (MessageTransformer) Class.forName(spec.getMessageTransformer()).newInstance();
        } catch (InstantiationException | IllegalAccessException | ClassNotFoundException e) {
            throw new MojoExecutionException(e);
        }
        t.transform(k.responseText, p, spec);
        try {
            if (spec.isOverwrite()){
                Files.writeString(child, k.responseText);
            } else {
                Files.writeString(child, k.responseText, StandardOpenOption.CREATE_NEW);
            }
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    @Override
    public void execute() throws MojoExecutionException, MojoFailureException {
        if (modelConfig == null) {
            modelConfig = new ModelConfig();
        }
        boolean anyEnabled = vibeSpecs.stream().anyMatch(VibeSpec::isEnabled);
        if (!anyEnabled){
            getLog().info("No VibeSpec are found or enabled");
            return;
        }

        ModelFetcher fetch = new ModelFetcher(modelConfig.getOwner(), modelConfig.getModelName());
        DType working = DType.valueOf(modelConfig.getWorkingMemType());
        DType quantized = DType.valueOf(modelConfig.getQuantizedMemType());
        AutoModelForCausaLm.Builder builder = AutoModelForCausaLm.newBuilder(fetch);
        NativeSimdTensorOperations operation = new NativeSimdTensorOperations(new ConfigurableTensorProvider(builder.getCache()).get());
        builder.withWorkingQuantType(working);
        builder.withWorkingQuantType(quantized);
        builder.withTensorProvider(new ConfigurableTensorProvider(operation));

        try (AbstractModel m = builder.build()) {
            for (VibeSpec spec : vibeSpecs) {
                runSingleSpec(m, spec);
            }
        }
    }

    private void debugConfig(){
        System.out.println("Project: " + project.getGroupId() + ":"
                + project.getArtifactId() + ":" + project.getVersion());
        System.out.println("vibe specs: " + vibeSpecs);
        System.out.println("modelConfig: " + modelConfig);
    }
}
