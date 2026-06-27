package io.teknek.deliverance.nanocode;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.testcontainers.containers.Container;
import org.testcontainers.containers.GenericContainer;
import org.testcontainers.images.builder.Transferable;
import org.testcontainers.utility.DockerImageName;

import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.Iterator;
import java.util.Map;
import java.util.Optional;

final class JavaSandboxTool {
    private static final ObjectMapper JSON = new ObjectMapper();
    private static final String DEFAULT_IMAGE = "eclipse-temurin:25-jdk";
    private static final int DEFAULT_TIMEOUT_SECONDS = 10;
    private static final int MAX_TIMEOUT_SECONDS = 60;
    private static final int DEFAULT_MAX_OUTPUT_CHARS = 12_000;

    private JavaSandboxTool() {
    }

    static String run(JsonNode args) throws Exception {
        String image = env("NANOCODE_JAVA_SANDBOX_IMAGE", DEFAULT_IMAGE);
        int timeoutSeconds = clamp(args.path("timeoutSeconds").asInt(DEFAULT_TIMEOUT_SECONDS), 1, MAX_TIMEOUT_SECONDS);
        int maxOutputChars = Math.max(1, args.path("maxOutputChars").asInt(DEFAULT_MAX_OUTPUT_CHARS));
        String mode = args.path("mode").asText("single-file");

        try (GenericContainer<?> container = new GenericContainer<>(DockerImageName.parse(image))
                .withCreateContainerCmdModifier(cmd -> {
                    cmd.withNetworkDisabled(true);
                    cmd.getHostConfig()
                            .withMemory(512L * 1024L * 1024L)
                            .withCpuCount(1L);
                })
                .withWorkingDirectory("/workspace")
                .withStartupTimeout(Duration.ofSeconds(30))
                .withCommand("sh", "-c", "sleep infinity")) {
            container.start();
            Optional<String> singleJavaFile = copyFiles(container, args.path("files"));
            Command command = command(mode, args.path("mainClass").asText("Main"), singleJavaFile);
            long start = System.nanoTime();
            Container.ExecResult result = container.execInContainer("timeout", timeoutSeconds + "s",
                    "sh", "-lc", command.value());
            long durationMs = (System.nanoTime() - start) / 1_000_000L;
            return JSON.writeValueAsString(new SandboxResult(
                    result.getExitCode(),
                    truncate(result.getStdout(), maxOutputChars),
                    truncate(result.getStderr(), maxOutputChars),
                    result.getExitCode() == 124,
                    durationMs,
                    command.value()));
        }
    }

    private static Optional<String> copyFiles(GenericContainer<?> container, JsonNode files) {
        if (!files.isObject()) {
            throw new IllegalArgumentException("java_sandbox requires object field 'files'");
        }
        Iterator<Map.Entry<String, JsonNode>> fields = files.fields();
        String javaFile = null;
        int javaFileCount = 0;
        while (fields.hasNext()) {
            Map.Entry<String, JsonNode> field = fields.next();
            String path = field.getKey();
            if (path.startsWith("/") || path.contains("..")) {
                throw new IllegalArgumentException("sandbox file path must be relative and must not contain '..': " + path);
            }
            if (path.endsWith(".java")) {
                javaFile = path;
                javaFileCount++;
            }
            container.copyFileToContainer(Transferable.of(field.getValue().asText().getBytes(StandardCharsets.UTF_8)),
                    "/workspace/" + path);
        }
        return javaFileCount == 1 ? Optional.of(javaFile) : Optional.empty();
    }

    static Command command(String mode, String mainClass, Optional<String> singleJavaFile) {
        return switch (mode) {
            case "single-file" -> singleFileCommand(mainClass, singleJavaFile);
            case "maven-test" -> new Command("mvn -q test");
            default -> throw new IllegalArgumentException("unsupported java_sandbox mode: " + mode);
        };
    }

    private static Command singleFileCommand(String mainClass, Optional<String> singleJavaFile) {
        String mainClassFile = mainClass + ".java";
        if (singleJavaFile.isPresent() && !singleJavaFile.get().equals(mainClassFile)) {
            return new Command("cp " + shell(singleJavaFile.get()) + " " + shell(mainClassFile)
                    + " && javac " + shell(mainClassFile) + " && java " + shell(mainClass));
        }
        return new Command("javac " + shell(mainClassFile) + " && java " + shell(mainClass));
    }

    private static String shell(String value) {
        return "'" + value.replace("'", "'\\''") + "'";
    }

    private static int clamp(int value, int min, int max) {
        return Math.max(min, Math.min(max, value));
    }

    private static String truncate(String value, int max) {
        if (value == null || value.length() <= max) {
            return value == null ? "" : value;
        }
        return value.substring(0, max) + "\n... (truncated, " + (value.length() - max) + " chars omitted)";
    }

    private static String env(String name, String defaultValue) {
        String value = System.getenv(name);
        return value == null || value.isBlank() ? defaultValue : value;
    }

    record Command(String value) {
    }

    private record SandboxResult(int exitCode, String stdout, String stderr, boolean timedOut, long durationMs,
                                 String command) {
    }
}
