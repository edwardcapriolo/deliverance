
## Flow of an inference engine

Many users are familiar with the JSON interface of popular services for ai chat, so lets start there.
```
http://bla/chat/completion -> 
  { "model":"granite", 
      "messages": [ 
         { "content": "Tell me about space", "role": "user"  }
      ]
  }
```

Not all inference engine flows may start as HTTP/JSON. It is possible that the request is programatic like using the
API directly or from a simple telnet like text interface.

### Jinja template 

There is transformation layer that prepares the request for the inference engine. 
This is done by jinja templates: Here is the first few lines of the llama template.

```jinja
template: {{- bos_token }}
{%- if custom_tools is defined %}
{%- set tools = custom_tools %}
{%- endif %}
{%- if not tools_in_user_message is defined %}
{%- set tools_in_user_message = true %}
{%- endif %}
...
``` 
At this point the flow looks something like this:

```
{ "model":"granite" ... } -> java code -> jinja template

```

In the jinja template above is "bos_token" aka 'begin' are special fields the model is trained on. Templates vary between models. 
For example, in llama models users can supply a "system" prompt such as, "You are an assistant that programs well". 
Gemma2 models have no "system" message, those 
instructions are sent as a"user" prompts. Some systems support multiple "bos_token". The point to get across is there 
is no one single way to do this generically. If the developer is aware Gemma2 can't support a system 
message it is possible to pre-process instead of let the request fail.

### Jinja -> prompt template (string)

Jinja makes a "prompt template", a specialized string prepared for a model. 
Here is an example:
```java
   ctx = ps.builder()
    .addSystemMessage("You are a chatbot that writes short correct responses.")
    .addUserMessage("What is the best season to plant avocados?").build();

    String expected = """
      <|system|>
      You are a chatbot that writes short correct responses.</s>
      <|user|>
      What is the best season to plant avocados?</s>
      <|assistant|>
      """;

```
You notice that in for this model there was a system turn, then a user turn. The final
string is "<|assistant|>". This is the magic, it signifies to the model that it is now the models
turn to generate! 

### Tokenizer: prompt to tokens

Models use a "tokenizer" to turn the text to Integers. It isn't the case that every word
is one token. Each tokenizer has different approaches.

```java
   List<String> tokens = m.getTokenizer().tokenize("show me the money!");
   assertEquals(List.of("show me the money!"), tokens);
   long[] encode = m.getTokenizer().encode("show me!");
   assertArrayEquals(new long[]{4294, 35, 1004, 29991}, encode);
   assertEquals("show", m.getTokenizer().decode(4294));
   assertEquals("me", m.getTokenizer().decode(1004));
   assertEquals("!", m.getTokenizer().decode(29991));
```

A word like "Edward" might be in the tokenizers vocabulary as 21891, however for another tokenizer 
it could make two tokes "Ed" 3434 "ward" 1232. 

### Generator: (input) tokens -> generator

Now that we have tokenized the input we can feed it into the heart of the inference engine and
let it cook!

The important thing to understand is the code of the inference engine controls the flow. Let's take the most
basic example the stopping condition. The user might supply a max token limit, in that case the engine
is counting the result tokens and stops processing. The models will return EOT (end of text) when they have reached
a conclusion. (It isn't the jinja's job to handle this flow)

```java
try (KvBufferCache.KvBuffer kvmem = kvBufferCache.getKvBuffer(sessionId.toString())) {
  int startPos = kvmem.getCurrentContextPosition();
  try (AbstractTensor logits = makeDenseTensor(config.vocabularySize)) {
    int [] promptTokens = constructPromptTokens(encoded);
    AbstractTensor last = batchForward(promptTokens, startPos, kvmem);
  ...
```

#### Generator -> tokens -> text
The generator generates integers. Some are words and some are special tokens. Ultimately they need
to become text for the end user, but we generally also need them before that as the user might
be supplying "stop words" that would terminate the request before the model is done.

```
process:
  while (not EOT)
    next token

output:
  4 3 2 34 99   
    
  tokenizer.decode(4)=Tomatoes 
  tokenizer.decode(3)=are
  tokenizer.decode(2)=a
  tokenizer.decord(34)=fruit
  tokenizer.decode(99)=<|eot_id|>

Output-->
  Tomatoes are a fruit<|eot_id|>
```

###  Generate Result (tokens + text) -> JSON (HTTP)

Generally the text from the response and the finish reasons is captured into a response.

```
{ "model":"granite3",
   choices[ 
     { "index":0, content:"Tomatoes are a fruit" }
   ],
   finish_reason: "length"
 }

```


