package net.deliverance.http;

import io.teknek.deliverance.model.CreateEmbeddingRequest;
import io.teknek.deliverance.model.CreateEmbeddingRequestInput;
import io.teknek.deliverance.model.CreateEmbeddingRequestModel;
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
public class EmbeddingControllerTest {
    @Autowired
    MockMvc mockMvc;

    @Test
    public void whenUsingSpringBootTestArgs_thenCommandLineArgSet(@Autowired Environment env) throws Exception {
        CreateEmbeddingRequest request = new CreateEmbeddingRequest();
        request.setModel(new CreateEmbeddingRequestModel("all-MiniLM-L6-v2"));
        request.setInput(new CreateEmbeddingRequestInput("This is it"));
        io.teknek.deliverance.JSON j = new io.teknek.deliverance.JSON();
        String s = j.getMapper().writeValueAsString(request);
        mockMvc.perform(MockMvcRequestBuilders.post("/embeddings")
                        .with(httpBasic("1","2"))
                        .contentType(MediaType.APPLICATION_JSON).content(s))
                .andExpect(status().isOk());
    }
}
