package net.deliverance.http;

import io.teknek.deliverance.generator.Generator;
import io.teknek.deliverance.generator.Response;
import io.teknek.deliverance.model.*;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mockito;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;

import org.springframework.core.env.Environment;
import org.springframework.http.MediaType;
import org.springframework.test.context.bean.override.mockito.MockitoBean;
import org.springframework.test.context.junit.jupiter.SpringExtension;

import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.request.MockMvcRequestBuilders;

import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

import static org.mockito.Mockito.when;

@ExtendWith(SpringExtension.class)
//@SpringBootTest(args = "--add-modules jdk.incubator.vector", properties = {"deliverance.tensor.operations.type=jvector"})
@SpringBootTest(args = "--add-modules jdk.incubator.vector")
@AutoConfigureMockMvc
public class DeliveranceControllerTest {

    @Autowired
    MockMvc mockMvc;

    @Test
    public void whenUsingSpringBootTestArgs_thenCommandLineArgSet(@Autowired Environment env) throws Exception {
        Response r = new Response("yo", null,
                null,0,0, 0, 0);
        CreateChatCompletionRequest request = new CreateChatCompletionRequest()
                .model("TinyLlama-1.1B-Chat-v1.0-Jlama-Q4")
                .stop(null);
        request.addMessagesItem(new ChatCompletionRequestMessage(
                new ChatCompletionRequestUserMessage().content(
                        new ChatCompletionRequestUserMessageContent("Generate the first letter of the alphabet is"))));

        io.teknek.deliverance.JSON j = new io.teknek.deliverance.JSON();
        String s = j.getMapper().writeValueAsString(request);
        mockMvc.perform(MockMvcRequestBuilders.post("/chat/completions")
                .contentType(MediaType.APPLICATION_JSON).content(s))
                .andExpect(status().isOk());
    }
}