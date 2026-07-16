# nanocode-deliverance

Tiny terminal coding agent inspired by `nanocode.java`, backed by a running Deliverance HTTP server.

![nanocode-deliverance terminal screenshot](nanocode_large.png)

This module intentionally stays small:

* Java 17.
* Depends on `deliverance-client`.
* Provides minimal tools: `read`, `write`, `edit`, `glob`, `grep`, `web_fetch`, `java_sandbox`.
* Risky/eval-prone `bash` is disabled unless explicitly enabled.

## Build

```sh
mvn -pl nanocode-deliverance -am -DskipTests package
```

Runnable jar:

```sh
java -jar nanocode-deliverance/target/nanocode-deliverance-0.0.10-SNAPSHOT-all.jar
```

## Start A Local Deliverance Server

From this directory:

```sh
sh run-server-llama-3.2-3b-instruct.sh
```

or:

```sh
sh run-server-tinyllama-chat.sh
```

Both scripts run the web jar from `../web/target` and default to `DELIVERANCE_PORT=8085`.
The client scripts read their server URL and model settings from their JSON config files.

## Run The Client

For Llama 3.2:

```sh
sh run_llama32_client.sh
```

For TinyLlama:

```sh
sh run_tiny_llama_client.sh
```

## Configuration

Nanocode uses one runtime switch:

```sh
--config FILE
```

Example:

```sh
java -jar target/nanocode-deliverance-0.0.10-SNAPSHOT-all.jar --config config-qwen3-4b-jq4.json
```

Config files are plain JSON:

```json
{
  "baseUrl": "http://localhost:8085",
  "model": "Qwen3-4B-JQ4",
  "ntokens": null,
  "maxTokens": 512,
  "maxToolResultChars": 2000,
  "maxToolRounds": 3,
  "temperature": 0.0,
  "toolsEnabled": true,
  "allowRiskyTools": false,
  "streamEnabled": true,
  "javaSandboxImage": "eclipse-temurin:25-jdk",
  "enableThinking": true
}
```

Fields:

* `baseUrl`: Deliverance server base URL.
* `model`: model name to send to `/chat/completions`.
* `ntokens`: optional total prompt+generation token budget; use `null` to omit.
* `maxTokens`: max response tokens.
* `maxToolResultChars`: max tool result chars kept in context.
* `maxToolRounds`: max assistant tool-use rounds per user turn, default `3` in the sample configs.
* `temperature`: sampling temperature.
* `toolsEnabled`: whether to send tool definitions.
* `allowRiskyTools`: enables `bash`; without it, shell execution is not advertised to the model and direct calls are rejected.
* `streamEnabled`: use streaming chat completions.
* `javaSandboxImage`: Testcontainers image for `java_sandbox`.
* `enableThinking`: send `chat_template_kwargs.enable_thinking` to the server.

During a REPL session, change mutable settings with `/config`:

```text
/config get rounds
/config set rounds 3
/config get thinking
/config set thinking off
```

Checked-in client configs:

```sh
config-qwen3-4b-jq4.json
config-llama32.json
config-tinyllama.json
```

Each assistant response with one or more tool calls counts as one tool round. Nanocode stops the current user turn after `maxToolRounds` rounds and prints a warning, so small models can retry tool use but cannot loop forever.

## Java Sandbox Tool

`java_sandbox` runs one-shot Java commands in a Testcontainers container with network disabled by default. It accepts files, copies them into `/workspace`, runs either a single Java file or Maven tests, and returns structured JSON with `exitCode`, `stdout`, `stderr`, `timedOut`, and `durationMs`.

Example tool arguments:

```json
{
  "mode": "single-file",
  "mainClass": "Main",
  "files": {
    "Main.java": "public class Main { public static void main(String[] args) { System.out.println(1 + 1); } }"
  },
  "timeoutSeconds": 10
}
```

## Notes

The module uses the generated `deliverance-client` `ChatApi` with a Jackson mix-in for request message serialization.
