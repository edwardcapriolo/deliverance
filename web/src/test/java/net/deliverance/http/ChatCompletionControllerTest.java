package net.deliverance.http;

import com.fasterxml.jackson.core.JsonProcessingException;
import io.teknek.deliverance.JSON;
import io.teknek.deliverance.generator.Response;
import io.teknek.deliverance.model.*;
import io.teknek.deliverance.model.Error;
import io.teknek.deliverance.safetensors.prompt.Tool;
import io.teknek.dysfx.Either;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mockito;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.core.env.Environment;
import org.springframework.http.MediaType;
import org.springframework.test.context.junit.jupiter.SpringExtension;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.request.MockMvcRequestBuilders;

import java.math.BigDecimal;
import java.util.*;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.springframework.security.test.web.servlet.request.SecurityMockMvcRequestPostProcessors.httpBasic;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;


@ExtendWith(SpringExtension.class)
//@SpringBootTest(args = "--add-modules jdk.incubator.vector", properties = {"deliverance.tensor.operations.type=jvector"})
@SpringBootTest(args = "--add-modules jdk.incubator.vector" )
@AutoConfigureMockMvc
public class ChatCompletionControllerTest {

    @Autowired
    MockMvc mockMvc;


    @Test
    public void whenUsingSpringBootTestArgs_thenCommandLineArgSet(@Autowired Environment env) throws Exception {
        CreateChatCompletionRequest request = new CreateChatCompletionRequest()
                .model("TinyLlama-1.1B-Chat-v1.0-Jlama-Q4")
                .stop(null);
        request.addMessagesItem(new ChatCompletionRequestMessage(
                new ChatCompletionRequestUserMessage().content(
                        new ChatCompletionRequestUserMessageContent("Generate the first letter of the alphabet is"))));

        JSON j = new JSON();
        String s = j.getMapper().writeValueAsString(request);
            mockMvc.perform(MockMvcRequestBuilders.post("/chat/completions")
                            .with(httpBasic("1","2"))
                .contentType(MediaType.APPLICATION_JSON).content(s))
                .andExpect(status().isOk());
    }


    @Test
    public void toolsSetup(@Autowired Environment env) throws Exception {

        /*
            auto (Default): The model automatically decides whether to call a tool (or multiple tools) and which tool to use based on the prompt.
    none: The model is forced to not use any tools and instead responds with a text message.
    any / required: The model is forced to call at least one tool, though it chooses which one.
    tool / Specific Function: You can explicitly name a specific tool the model must use, forcing the output to follow that tool's structure.
         */
        CreateChatCompletionRequest request = new CreateChatCompletionRequest()
                .model("TinyLlama-1.1B-Chat-v1.0-Jlama-Q4")
                .stop(null);
        request.addMessagesItem(new ChatCompletionRequestMessage(
                new ChatCompletionRequestUserMessage().content(
                        new ChatCompletionRequestUserMessageContent("Generate the first letter of the alphabet is"))));

        request.addToolsItem(new ChatCompletionTool().function( new FunctionObject().name("some tool")));
        CausalLanguageModel m = Mockito.mock(CausalLanguageModel.class);
        JSON j = new JSON();

        Either<Error, PreparedRequest> response = ChatCompletionService.mapRequest(new HashMap<>(), m , request);
        //Right<Error,PreparedRequest> right = (Right<Error,PreparedRequest>) response;
        //PreparedRequest req = (PreparedRequest) response.productElement(0);

    }

    @Test
    public void requestMapsLogprobFlags() {
        CreateChatCompletionRequest request = new CreateChatCompletionRequest()
                .model("TinyLlama-1.1B-Chat-v1.0-Jlama-Q4")
                .logprobs(true)
                .topLogprobs(5)
                .stop(null);
        request.addMessagesItem(new ChatCompletionRequestMessage(
                new ChatCompletionRequestUserMessage().content(
                        new ChatCompletionRequestUserMessageContent("Generate the first letter of the alphabet is"))));

        CausalLanguageModel m = Mockito.mock(CausalLanguageModel.class);
        io.teknek.deliverance.safetensors.prompt.PromptSupport promptSupport = Mockito.mock(io.teknek.deliverance.safetensors.prompt.PromptSupport.class);
        io.teknek.deliverance.safetensors.prompt.PromptSupport.Builder builder = Mockito.mock(io.teknek.deliverance.safetensors.prompt.PromptSupport.Builder.class);
        Mockito.when(m.promptSupport()).thenReturn(Optional.of(promptSupport));
        Mockito.when(promptSupport.builder()).thenReturn(builder);
        Mockito.when(builder.addUserMessage(Mockito.anyString())).thenReturn(builder);

        Either<Error, PreparedRequest> response = ChatCompletionService.mapRequest(new HashMap<>(), m, request);
        PreparedRequest prepared = (PreparedRequest) response.productElement(0);
        assertEquals(true, prepared.generatorParameters().logProbs.orElse(false));
        assertEquals(5, prepared.generatorParameters().topLogProbs.orElse(0));
    }

    @Test
    public void responseMapsSamplerReturnsIntoChoiceLogprobs() {
        PriorityQueue<IndexValueToken> top = new PriorityQueue<>();
        IndexValueToken chosen = new IndexValueToken(973, 20.0f, " //");
        chosen.logProb = -1.25f;
        IndexValueToken alt = new IndexValueToken(18171, 19.0f, "~~");
        alt.logProb = -2.25f;
        top.add(chosen);
        top.add(alt);

        Response response = new Response(" //", " //", io.teknek.deliverance.generator.FinishReason.MAX_TOKENS,
                1, List.of(973), 0, 0, List.of(new SamplerReturn(973, top)));

        CreateChatCompletionResponseChoicesInnerLogprobs logprobs = ChatCompletionController.toLogProbs(response);
        assertTrue(logprobs != null && logprobs.getContent() != null);
        assertEquals(" //", logprobs.getContent().get(0).getToken());
        assertEquals(BigDecimal.valueOf(-1.25f), logprobs.getContent().get(0).getLogprob());
        assertEquals(2, logprobs.getContent().get(0).getTopLogprobs().size());
    }

    @Test
    public void toolParsing() throws JsonProcessingException {
        String tool = """
                  {
                    "type": "function",
                    "function": {
                      "name": "get_current_weather",
                      "description": "Get the current weather in a given location",
                      "parameters": {
                        "type": "object",
                        "properties": {
                          "city": { "type": "string", "description": "City name" },
                          "state": { "type": "string", "description": "State abbreviation" },
                          "unit": { "type": "string", "enum": ["celsius", "fahrenheit"] }
                        },
                        "required": ["city", "state", "unit"]
                      }
                    }
                  }
                """;
        ChatCompletionTool t = JSON.getDefault().getMapper().readValue(tool, ChatCompletionTool.class);

        Tool prompTool = ChatCompletionService.convert(t);

        assertEquals("get_current_weather", prompTool.getFunction().getName());
        assertEquals("Get the current weather in a given location", prompTool.getFunction().getDescription());
        assertEquals(Arrays.asList("city", "state", "unit"), prompTool.getFunction().getParameters().getRequired());
        assertEquals("object", prompTool.getFunction().getParameters().getType());
        //TODO this looks dubious s there is another layer of properties in here.
        //assertEquals( Arrays.asList("string", "string"), prompTool.getFunction().getParameters().getProperties());

    }



}
