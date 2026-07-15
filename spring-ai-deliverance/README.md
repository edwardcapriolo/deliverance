# Spring AI Deliverance

Spring AI integration for Deliverance.

This module exposes a running Deliverance chat-completion HTTP server through Spring AI's `ChatModel` interface.

It targets Spring AI 2.x and Java 17. Embedded Deliverance inference is available from the separate `spring-ai-deliverance-embedded` Java 25 add-on module.

## Client Mode

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

## Embedded Add-On

Add `spring-ai-deliverance-embedded` when the Spring application should load Deliverance in-process.
This add-on requires Java 25 because it depends on Deliverance core inference.

```xml
<dependency>
  <groupId>io.teknek.deliverance</groupId>
  <artifactId>spring-ai-deliverance-embedded</artifactId>
  <version>${deliverance.version}</version>
</dependency>
```

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
- demo HTTP controller behavior with a fake `ChatModel`
- embedded add-on behavior with a fake `CausalLanguageModel`

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
