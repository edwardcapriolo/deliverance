# Spring AI Deliverance

Spring AI integration for Deliverance.

Deliverance can be used as a pure Java local inference engine or as a normal chat-completion HTTP server. This module exposes Deliverance through Spring AI's `ChatModel` interface.

## Modes

### Client Mode

Client mode is the default. It calls a running Deliverance HTTP server.

```yaml
spring:
  ai:
    deliverance:
      mode: client
      model: edwardcapriolo/Qwen3-4B-JQ4
      base-url: http://localhost:8080
```

The configured `model` value is sent with each `/chat/completions` request.

### Embedded Mode

Embedded mode loads Deliverance in the Spring application JVM.

```yaml
spring:
  ai:
    deliverance:
      mode: embedded
      model: edwardcapriolo/Qwen3-4B-JQ4
      auto-pull: true
      huggingface:
        token: ${HF_TOKEN:}
      model-config: classpath:/models/qwen3-4b-jq4.json
```

Embedded mode is opt-in because it may download model weights and allocate native/direct memory.

## Options

Use Spring AI generic options where possible. Deliverance-specific options are available through `DeliveranceChatOptions`:

```java
Prompt prompt = new Prompt(
        "Create a ticket id.",
        DeliveranceChatOptions.builder()
                .model("edwardcapriolo/Qwen3-4B-JQ4")
                .temperature(0.0)
                .maxTokens(32)
                .guidedRegex("TICKET-[0-9]{4}")
                .build());
```

Supported Deliverance-specific options include:

```text
seed
topK
logprobs
topLogprobs
guidedChoice
guidedRegex
guidedJson
```

## Demo

See `spring-ai-deliverance-demo` for a PR review service that exposes HTTP endpoints backed by Spring AI and Deliverance.

The demo is designed around a realistic private-code workflow rather than a weather-tool example.

## Tests

Normal tests cover:

- request mapping to generated Deliverance client models
- embedded mode with a fake `CausalLanguageModel`
- demo HTTP controller behavior with a fake `ChatModel`

Real server tests are scaffolded with Testcontainers and are opt-in:

```sh
mvn -pl spring-ai-deliverance -Ddeliverance.springai.it.container=true test
```

Use a released image:

```sh
-Ddeliverance.springai.testcontainer.image=ecapriolo/deliverance:0.0.x
```

Or build a local image from this checkout:

```sh
-Ddeliverance.springai.it.local-image=true
```
