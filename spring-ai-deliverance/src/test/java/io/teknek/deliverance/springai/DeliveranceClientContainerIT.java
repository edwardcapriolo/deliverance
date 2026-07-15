package io.teknek.deliverance.springai;

import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;
import org.testcontainers.containers.GenericContainer;
import org.testcontainers.images.builder.ImageFromDockerfile;
import org.testcontainers.junit.jupiter.Testcontainers;
import org.testcontainers.utility.DockerImageName;

import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertTrue;

@Testcontainers(disabledWithoutDocker = true)
class DeliveranceClientContainerIT {

    @Test
    void clientModeCanCallReleasedDeliveranceContainer() {
        Assumptions.assumeTrue(Boolean.getBoolean("deliverance.springai.it.container"));
        String image = System.getProperty("deliverance.springai.testcontainer.image", "ecapriolo/deliverance:latest");

        try (GenericContainer<?> container = new GenericContainer<>(DockerImageName.parse(image)).withExposedPorts(8080)) {
            container.start();
            String baseUrl = "http://" + container.getHost() + ":" + container.getMappedPort(8080);
            DeliveranceChatModel model = new DeliveranceChatModel(
                    DeliveranceApi.create(baseUrl, null),
                    new com.fasterxml.jackson.databind.ObjectMapper(),
                    DeliveranceChatOptions.builder()
                            .model(System.getProperty("deliverance.springai.testcontainer.model", "test-model"))
                            .temperature(0.0)
                            .maxTokens(16)
                            .build());

            String response = model.call("Say hello in one short sentence.");
            assertTrue(response != null && !response.isBlank());
        }
    }

    @Test
    void clientModeCanCallLocalDeliveranceImage() {
        Assumptions.assumeTrue(Boolean.getBoolean("deliverance.springai.it.local-image"));

        ImageFromDockerfile image = new ImageFromDockerfile("deliverance:spring-ai-it", false)
                .withDockerfile(Path.of("docker", "Dockerfile"));
        try (GenericContainer<?> container = new GenericContainer<>(image).withExposedPorts(8080)) {
            container.start();
            assertTrue(container.isRunning());
        }
    }
}
