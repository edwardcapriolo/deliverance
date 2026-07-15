package net.deliverance.http;

import io.teknek.deliverance.model.CreateEmbeddingRequest;
import io.teknek.deliverance.model.CreateEmbeddingRequestInput;
import io.teknek.deliverance.model.CreateEmbeddingRequestModel;
import io.teknek.deliverance.model.ListModelsResponse;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.webmvc.test.autoconfigure.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.core.env.Environment;
import org.springframework.http.MediaType;
import org.springframework.test.context.junit.jupiter.SpringExtension;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.request.MockMvcRequestBuilders;

import static org.springframework.security.test.web.servlet.request.SecurityMockMvcRequestPostProcessors.httpBasic;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.content;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

@ExtendWith(SpringExtension.class)
@SpringBootTest(args = "--enable-native-access=ALL-UNNAMED --add-modules jdk.incubator.vector", 
        properties = {"deliverance.tensor.operations.type=jvector"})
@AutoConfigureMockMvc
public class ModelControllerTest {
    @Autowired
    MockMvc mockMvc;

    @Test
    public void listModels(@Autowired Environment env) throws Exception {
        String expected = """
                {"object":"list","data":[{"id":"TinyLlama-1.1B-Chat-v1.0-Jlama-Q4","created":0,"object":"model","owned_by":"tjake"},{"id":"all-MiniLM-L6-v2","created":0,"object":"model","owned_by":"sentence-transformers"}]}""";
        mockMvc.perform(MockMvcRequestBuilders.get("/models")
                        .with(httpBasic("1","2"))
                        .contentType(MediaType.APPLICATION_JSON))
                .andExpect(status().isOk())
                .andExpect(content().string(expected));
    }
}
