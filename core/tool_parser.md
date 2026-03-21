## Tool Parsers

Inference engines can be told about Tools (previously called functions). They do not call the tools directly during inferencing, instead
the reply with special instructions for the user's "agent" to use the tool to complete a task.

### Notice: Tool support 
Not all models support tools. Even if a model supports tools, if it is smaller(weaker) it might struggle 
to find the right tool to call or call it correctly.

### Before you start
You may want to read [inference flow](inference_flow.md) so you understand the processing steps
and where tools fit in.

### Tool model
Fist, you must describe the tool and say what it does. For example we made a simple 
tool to flip a coin. 
```
Tool tool = Tool.from(Function.builder().name("flip_coin").description("This methods will flip a coin. The result will be H for he
try (AbstractModel m = ModelSupport.loadModel(f, DType.F32, DType.I8, new ConfigurableTensorProvider(operation),                  
        new MetricRegistry(), tensorCache, new KvBufferCacheSettings(true), fetch, new TokenizerRenderer())) {                    
    String prompt = "I would like to decide who goes first by a coin flip";                                                       
    PromptSupport.Builder g = m.promptSupport().get().builder()                                                                   
            .useChatTemplate(text)                                                                                                
            .addUserMessage(prompt);    
...
```
### PromptContext Jinja
Remember each model has a different Jinja template and will insert tools differently. You will see below that the 
LLama template grows quite significantly when a tool is listed. Note: this means a larger prompt.

```jija
template:<|start_header_id|>system<|end_header_id|>

Environment: ipython
Cutting Knowledge Date: December 2023
Today Date: 26 Jul 2024

<|eot_id|><|start_header_id|>user<|end_header_id|>

Given the following functions, please respond with a JSON for a function call with its proper arguments that best answers the given prompt.

Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.Do not use variables.

{"type": "function", "function": {"name": "flip_coin", "description": "This methods will flip a coin. The result will be H for heads or T for tails.", "parameters": {"type": "object", "properties": {}, "required": []}}}

I would like to decide who goes first by a coin flip<|eot_id|><|start_header_id|>assistant<|end_header_id|>


```

### Model detects need for tool

The model uses it's intelligence to determine if it should use a tool.  For LLAMA some json is simply included in the response like so:

```
The JSON for the function call with its proper arguments is:

{"name": "flip_coin", "parameters": {}}<|end_header_id|>  # This will flip a coin and return the result, which will determine who goes first.

In this case, the function "flip_coin" is called with no arguments. The result of the coin flip will determine who goes first. If the result is "H", then the first player goes first, and if the result is "T", then the second player goes first.
```
Qwen wraps the json in XML <tool> to make it easier to pick out :)

### ToolCallParser.java

As we can see above the ToolCall is included directly in the text stream. 

```
{"name": "flip_coin", "parameters": {}}
```
There can be 0-many ToolCall(s) in a response. Thus our interface has to interrogate the 
response and pick them out. 

```
public interface ToolCallParser {
    /**
     *
     * @param response the response directly from the AbstractModel
     * @return a list of all tool calls found in the response 
     */
    List<ToolCall> extract(Response response);
}
```

### HTTP response

Remember the "Agent" is the consumer of the response. They agent is looking for a finish_reason = "tool_calls". 
This tells the agent "inference engine is finished generating", and "agent call this tool".

```json
{
  
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "tool_calls": [
          {
            "id": "call_123",
            "type": "function",
            "function": {
              "name": "coin_flip",
              "arguments": ""
            }
          }
        ]
      },
      "finish_reason": "tool_calls"
    }
  ]
}

```
The code that handles this is found in the controller.  
```java
List<ToolCall> tcs = model.getToolCallParser().extract(resp);
CreateChatCompletionResponseChoicesInner z2 = new CreateChatCompletionResponseChoicesInner()
        .message(new ChatCompletionResponseMessage().content(resp.responseTextWithSpecialTokens))
        .index(0);
if (!tcs.isEmpty()) {
  z2.setFinishReason(CreateChatCompletionResponseChoicesInner.FinishReasonEnum.TOOL_CALLS);
} 
tcs.forEach(tc -> {
    ChatCompletionMessageToolCall t = new ChatCompletionMessageToolCall();
    t.id(tc.getId());
    t.function(new ChatCompletionMessageToolCallFunction().name(tc.getName()));
    try {
        String paramsAsString = JsonUtils.om.writeValueAsString(tc.getParameters());
        t.getFunction().arguments(paramsAsString);
    } catch (JsonProcessingException e) {
        throw new RuntimeException(e);
    }
    z2.getMessage().addToolCallsItem(t);
});
response.addChoicesItem(z2);
```

### Quirky stuff

- The model argument of the tool are Map<String,Object>, but in the JSON it is a flat string.
- The FINISH_REASON is "tools_call". However the model can continue generating a lot of text, and hit stop_words or max_tokens
- The code is responsible for creating ids for each tool call ""id": "call_123"


