# Inference Benchmark

`InferenceBenchmark` is a command-line benchmark runner for long, mixed-category inference prompts. It is not a unit
test. It records per-turn prompt size, generated token count, total latency, generation latency, and generated tokens per
second.

The built-in suite is a small MT-Bench-derived subset with reasoning, math, coding, extraction, STEM, humanities, and
writing prompts. For a larger suite, download FastChat MT-Bench questions and pass them as JSONL:

```sh
curl -L -o /tmp/mt_bench_question.jsonl \
  https://raw.githubusercontent.com/lm-sys/FastChat/main/fastchat/llm_judge/data/mt_bench/question.jsonl
```

## Deliverance

```sh
MAVEN_OPTS="-XX:TieredStopAtLevel=1" mvn -q -pl core -am -DskipTests compile

MAVEN_OPTS="-XX:TieredStopAtLevel=1" mvn -q -pl core \
  -Dexec.classpathScope=test \
  -Dexec.executable=java \
  -Dexec.args="--add-modules jdk.incubator.vector,jdk.httpserver,java.net.http --add-opens java.base/java.nio=ALL-UNNAMED --enable-native-access=ALL-UNNAMED -cp %classpath io.teknek.deliverance.benchmark.InferenceBenchmark --engine deliverance --owner tjake --model gemma-2-2b-it-JQ4 --max-tokens 256 --output target/deliverance-benchmark.csv --jsonl-output target/deliverance-benchmark.jsonl" \
  org.codehaus.mojo:exec-maven-plugin:3.5.0:exec
```

## Ollama

Start Ollama and pull the comparison model first:

```sh
ollama pull llama3.2
```

Then run:

```sh
MAVEN_OPTS="-XX:TieredStopAtLevel=1" mvn -q -pl core -am -DskipTests compile

MAVEN_OPTS="-XX:TieredStopAtLevel=1" mvn -q -pl core \
  -Dexec.classpathScope=test \
  -Dexec.executable=java \
  -Dexec.args="--add-modules jdk.incubator.vector,jdk.httpserver,java.net.http --add-opens java.base/java.nio=ALL-UNNAMED --enable-native-access=ALL-UNNAMED -cp %classpath io.teknek.deliverance.benchmark.InferenceBenchmark --engine ollama --ollama-model llama3.2 --max-tokens 256 --output target/ollama-benchmark.csv --jsonl-output target/ollama-benchmark.jsonl" \
  org.codehaus.mojo:exec-maven-plugin:3.5.0:exec
```

## Same Suite, Both Engines

```sh
MAVEN_OPTS="-XX:TieredStopAtLevel=1" mvn -q -pl core -am -DskipTests compile

MAVEN_OPTS="-XX:TieredStopAtLevel=1" mvn -q -pl core \
  -Dexec.classpathScope=test \
  -Dexec.executable=java \
  -Dexec.args="--add-modules jdk.incubator.vector,jdk.httpserver,java.net.http --add-opens java.base/java.nio=ALL-UNNAMED --enable-native-access=ALL-UNNAMED -cp %classpath io.teknek.deliverance.benchmark.InferenceBenchmark --engine both --owner tjake --model gemma-2-2b-it-JQ4 --ollama-model llama3.2 --suite-file /tmp/mt_bench_question.jsonl --max-cases 20 --max-tokens 256 --output target/inference-benchmark.csv --jsonl-output target/inference-benchmark.jsonl" \
  org.codehaus.mojo:exec-maven-plugin:3.5.0:exec
```

Use `exec:exec`, not `exec:java`, for this benchmark. The benchmark depends on `jdk.incubator.vector`, and `exec:exec`
lets the command launch `java` with the required `--add-modules` flags. Do not include `-am` on the run command. `-am`
adds upstream reactor projects and Maven may execute against the parent project before `core`, causing
`ClassNotFoundException`.

Use `-Dexec.classpathScope=test` for now. The `core` module currently declares the optional `native` module and
`slf4j-simple` binding as test-scope dependencies, so the default exec classpath falls back to Panama and suppresses the
native-provider warning logs.

## Local Tensor Parallel

Use `--tensor-parallel-size 2` to run Deliverance through an in-process tensor-parallel group with two local rank models
and a coordinator model. This mirrors the tensor-parallel integration-test shape without starting gossip workers.

```sh
MAVEN_OPTS="-XX:TieredStopAtLevel=1" mvn -q -pl core \
  -Dexec.classpathScope=test \
  -Dexec.executable=java \
  -Dexec.args="--add-modules jdk.incubator.vector,jdk.httpserver,java.net.http --add-opens java.base/java.nio=ALL-UNNAMED --enable-native-access=ALL-UNNAMED -cp %classpath io.teknek.deliverance.benchmark.InferenceBenchmark --engine deliverance --owner tjake --model gemma-2-2b-it-JQ4 --tensor-parallel-size 2 --max-tokens 256 --output target/deliverance-tp2-benchmark.csv --jsonl-output target/deliverance-tp2-benchmark.jsonl" \
  org.codehaus.mojo:exec-maven-plugin:3.5.0:exec
```

## Output Columns

* `engine`: `deliverance` or `ollama`
* `model`: model identifier
* `case_id`, `category`, `turn`: prompt-suite identifiers
* `prompt_chars`: rendered prompt/request size in characters
* `prompt_tokens`: Deliverance `Response.promptTokens` or Ollama `prompt_eval_count`
* `generated_tokens`: Deliverance generated token count or Ollama `eval_count`
* `total_ms`: full request latency
* `generation_ms`: generated-token decode time estimate. For Ollama this is `eval_duration`; for Deliverance this is
  `total_ms - time_to_first_token_ms`, so it excludes prefill plus first-token latency and is approximate.
* `tokens_per_second`: generated tokens per second using `generation_ms`
* `response_chars`: response text size
* `finish_reason`: engine finish reason when available

Ollama's tokens/sec is computed from `eval_count / eval_duration`. Deliverance's tokens/sec is computed from generated
token count after the first token divided by `total_ms - time_to_first_token_ms`, because Deliverance does not yet expose
a separate decode-only duration field.

## JSONL Transcripts

Pass `--jsonl-output target/inference-benchmark.jsonl` to write one JSON object per turn. Each record includes the chat
messages so far, rendered Deliverance prompt or Ollama request JSON, response text, response text with special tokens, and
the same metrics written to CSV. The JSONL file is flushed after every turn so partial results survive interrupted runs.
