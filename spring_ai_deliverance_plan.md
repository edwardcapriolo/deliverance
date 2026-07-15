# Spring AI Deliverance Plan

## Goal

Add first-class Spring AI support for Deliverance and provide a compelling real-world demo app.

Positioning:

```text
AI without the black box:
pure Java, local models, no provider lock-in, embedded or client mode, modern inference features.
```

The core idea is that Spring developers should be able to use Deliverance through Spring AI interfaces while still keeping model execution local, inspectable, and provider-free.

## Modules

Add:

```text
spring-ai-deliverance
spring-ai-deliverance-demo
```

### spring-ai-deliverance

Spring AI integration module.

Responsibilities:

- Implement Spring AI `ChatModel`.
- Add streaming support later through Spring AI streaming interfaces.
- Support embedded Deliverance mode.
- Support client mode against a running Deliverance HTTP server.
- Map Spring AI options to Deliverance `GeneratorParameters`.
- Expose Deliverance-specific options for guided decoding.
- Provide Spring Boot auto-configuration.

### spring-ai-deliverance-demo

Spring Boot HTTP application demonstrating a real workflow.

Responsibilities:

- Expose REST endpoints for code review workflows.
- Use Spring AI `ChatModel` backed by Deliverance.
- Use prompt templates from resources.
- Demonstrate guided JSON, guided choice, and normal chat generation.
- Include Bitbucket Pipeline integration examples.

## Runtime Modes

### Embedded Mode

Runs Deliverance in the same Spring Boot process.

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

Notes:

- `model` is always one string. In embedded mode, `owner/name` is split internally for `ModelFetcher`.
- `auto-pull` behaves like Ollama-style model pull ergonomics.
- `huggingface.token` supports gated/private model downloads.
- `model-config` should reuse Deliverance's existing model config JSON format instead of mirroring every builder option as Spring properties.
- Embedded mode is opt-in. It is a key differentiator, but it should not be the default path for tests or simple client integrations.

### Client Mode

Uses a running Deliverance HTTP server.

```yaml
spring:
  ai:
    deliverance:
      mode: client
      base-url: http://localhost:8080
      model: edwardcapriolo/Qwen3-4B-JQ4
```

This mode keeps the Spring application lightweight and lets multiple apps share one Deliverance server.

Client mode still sends the configured `model` string in each chat completion request. The server interprets the model id according to its own loaded model configuration.

Client mode should be the default for the Spring AI integration because it is lightweight for application users. Tests should still include real Deliverance server coverage, not only mocked HTTP responses.

## Spring AI API Surface

Initial implementation:

```text
ChatModel
```

Future implementation:

```text
StreamingChatModel
EmbeddingModel
```

Mapping:

```text
Spring AI Prompt
  -> Deliverance prompt/messages
  -> GeneratorParameters
  -> Deliverance model.generate(...)
  -> Spring AI ChatResponse
```

For client mode:

```text
Spring AI Prompt
  -> /chat/completions request
  -> Deliverance server
  -> Spring AI ChatResponse
```

## Options

Use Spring AI generic options where available and add Deliverance-specific options for features Spring AI does not model directly.

Deliverance-specific options should include:

```text
guidedChoice
guidedRegex
guidedJson
seed
topK
topLogprobs
xtcThreshold
xtcProbability
```

Do not mirror every `AutoModelForCausaLm.Builder` setting as a Spring property. Use `model-config` for advanced model/runtime configuration.

Use one model property across modes:

```yaml
spring.ai.deliverance.model=edwardcapriolo/Qwen3-4B-JQ4
```

Do not introduce separate `owner` and `name` properties unless a concrete need appears later.

## Demo App: Local PR Review Service

The demo should be a real Spring HTTP service, not a command-line runner.

Tagline:

```text
Private code review with local AI: no provider, no code leaves your network.
```

Architecture:

```text
Bitbucket Pipeline
  -> HTTP POST review payload
  -> Spring Boot demo app
  -> Spring AI ChatModel
  -> spring-ai-deliverance
  -> embedded Deliverance or Deliverance server
  -> local model
  -> guided JSON response
  -> normal JSON API response
```

## Demo HTTP API

### Review Pull Request

```http
POST /api/reviews/pull-request
Content-Type: application/json
```

Request:

```json
{
  "repository": "payments-service",
  "pullRequestId": "PR-428",
  "title": "Add guided JSON support",
  "description": "Adds structured output support for JSON schema and regex.",
  "sourceBranch": "feature/guided-json",
  "targetBranch": "main",
  "diff": "... unified diff ...",
  "testOutput": "... maven test output ...",
  "changedFiles": [
    "core/src/main/java/...",
    "web/deliverance_specification.yaml"
  ]
}
```

Response:

```json
{
  "summary": "Adds guided JSON support through schema-to-regex and index-based token masking.",
  "riskLevel": "medium",
  "findings": [
    {
      "severity": "medium",
      "file": "LogitsProcessorFactory.java",
      "line": 49,
      "message": "Index construction may be expensive for repeated schemas unless cached."
    }
  ],
  "recommendedTests": [
    "Add web request mapping test for guided_json.",
    "Run GemmaPromptIT guided JSON case."
  ],
  "releaseNote": "Adds guided JSON structured output support."
}
```

This endpoint should use guided JSON so the app receives a machine-readable review response.

### Release Note

```http
POST /api/reviews/release-note
```

Returns a concise release note from a PR payload.

### Reviewer Checklist

```http
POST /api/reviews/checklist
```

Response:

```json
{
  "items": [
    "Verify request fields are mapped to GeneratorParameters.",
    "Confirm generated API sources were refreshed.",
    "Run guided decoding benchmark."
  ]
}
```

### Classify Pull Request

```http
POST /api/reviews/classify
```

Uses guided choice:

```text
feature | bugfix | performance | docs | internal
```

## Prompt Templates

Use resource templates instead of hard-coded prompts.

```text
spring-ai-deliverance-demo/src/main/resources/prompts/pr-review.st
spring-ai-deliverance-demo/src/main/resources/prompts/release-note.st
spring-ai-deliverance-demo/src/main/resources/prompts/checklist.st
spring-ai-deliverance-demo/src/main/resources/prompts/classify.st
```

Application flow:

```text
Controller
  -> Service
  -> prompt template
  -> Spring AI ChatModel
  -> Deliverance
```

## Bitbucket Pipeline Example

Example `bitbucket-pipelines.yml` step:

```yaml
pipelines:
  pull-requests:
    "**":
      - step:
          name: Local AI PR Review
          image: eclipse-temurin:25
          script:
            - apt-get update && apt-get install -y git curl jq
            - git fetch origin "$BITBUCKET_PR_DESTINATION_BRANCH"
            - git diff "origin/$BITBUCKET_PR_DESTINATION_BRANCH"...HEAD > pr.diff
            - mvn test | tee test-output.txt || true
            - |
              jq -n \
                --arg repo "$BITBUCKET_REPO_SLUG" \
                --arg pr "$BITBUCKET_PR_ID" \
                --arg title "$BITBUCKET_PR_TITLE" \
                --rawfile diff pr.diff \
                --rawfile tests test-output.txt \
                '{
                  repository: $repo,
                  pullRequestId: $pr,
                  title: $title,
                  diff: $diff,
                  testOutput: $tests
                }' > review-request.json
            - |
              curl -sS -X POST http://deliverance-review.internal/api/reviews/pull-request \
                -H 'Content-Type: application/json' \
                --data @review-request.json \
                | tee review.json
            - jq . review.json
```

Optional future step: post the review result back as a Bitbucket PR comment.

## Why This Demo Works

This is more compelling than a generic weather tool demo because it shows a real enterprise use case:

- private code review
- local AI
- no provider key
- no source code leaving the network
- Spring AI standard interfaces
- guided JSON for machine-readable output
- guided choice for classification
- embedded or client/server Deliverance

## Implementation Order

1. Inspect `spring-ai-ollama` in Spring AI for module conventions.
2. Add `spring-ai-deliverance` module skeleton.
3. Implement `DeliveranceChatOptions`.
4. Implement client-mode `DeliveranceChatModel` first.
5. Add Spring Boot auto-configuration for client mode.
6. Add small mapping tests for request/response conversion.
7. Add real client-mode integration tests using a Deliverance Testcontainer.
8. Support two Testcontainer modes: released Docker Hub image and locally built image from this checkout.
9. Implement embedded-mode `DeliveranceChatModel`.
10. Add embedded auto-configuration, off by default.
11. Add streaming support.
12. Add `spring-ai-deliverance-demo` module.
13. Implement PR review endpoints and prompt templates.
14. Add README and Bitbucket pipeline docs.
15. Add demo integration tests that call the HTTP endpoints against real Deliverance in client mode or embedded mode.

## Upstream Spring AI Path

If this is contributed upstream to Spring AI, client mode is likely the easiest first contribution because it looks like a standard provider integration.

Embedded mode may remain a Deliverance-specific starter if Spring AI prefers provider/client integrations over embedding a full inference engine inside the application process.

## Testing Strategy

Do not rely on a large set of hallucinated mock-provider tests. Because Deliverance is in this repository, integration tests should exercise real Deliverance whenever practical.

### Mapping Unit Tests

Keep a small number of unit tests that verify object mapping only:

- Spring AI prompt/options -> Deliverance request object.
- Deliverance response object -> Spring AI response type.
- Deliverance-specific options such as `guidedJson`, `guidedRegex`, and `topK` map correctly.

These tests are not proof that the provider works. They only protect local mapping code.

### Client Mode With Testcontainers

Use Testcontainers for real client-mode integration tests.

Two modes should be supported:

```text
released image
local image from current checkout
```

Released image mode:

```text
-Ddeliverance.springai.testcontainer.mode=released
-Ddeliverance.springai.testcontainer.image=ecapriolo/deliverance:0.0.x
```

Local image mode:

```text
-Ddeliverance.springai.testcontainer.mode=local
```

Local image mode should build the Docker image from this repository and run the Spring AI client tests against it. This is preferable for validating changes in the current checkout instead of only validating compatibility with a previously published image.

Client-mode integration tests should cover:

- simple `ChatModel.call(...)` against `/chat/completions`.
- guided JSON request through Spring AI options.
- guided regex request through Spring AI options.
- streaming response once streaming support exists.
- auth/base URL/model property behavior.

### Embedded Mode Tests

Embedded mode should be a supported feature, but it should remain opt-in for tests because it can download/load models and allocate native memory.

Use layers:

- Auto-configuration smoke test with a fake or test `CausalLanguageModel` bean.
- Disabled/profile-gated real embedded model test.
- Explicit local-model test profile for developers who have the model cache available.

Example flags:

```text
-Dspring.ai.deliverance.embedded.it=true
-Dspring.ai.deliverance.model=edwardcapriolo/Qwen3-4B-JQ4
```

### Demo App Tests

The demo module should test the real HTTP endpoints:

```text
POST /api/reviews/pull-request
POST /api/reviews/release-note
POST /api/reviews/checklist
POST /api/reviews/classify
```

For confidence, at least one profile should run the demo app against a real Deliverance Testcontainer and verify that a Bitbucket-style PR payload returns structured JSON.
