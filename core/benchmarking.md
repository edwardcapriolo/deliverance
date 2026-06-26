# Benchmarking

Deliverance includes a benchmark runner and several shell scripts for repeatable local inference tests. The runner is `io.teknek.deliverance.benchmark.InferenceBenchmark`; the scripts wrap it with the JVM flags, native-library paths, and model defaults we use most often.

For the complete CLI reference, see [Inference Benchmark](InferenceBenchmark.md). This page is the practical guide for the scripts and the benchmark workflow.

## Before Running

Build or verify the native module before using scripts that expect native SIMD:

```sh
(cd native && mvn test)
```

The native scripts detect the current platform and use a classifier-specific library directory, for example:

```text
native/target/native-lib-only/osx-aarch_64
native/target/native-lib-only/linux-aarch_64
```

You can override classifier detection with:

```sh
DELIVERANCE_NATIVE_CLASSIFIER=osx-aarch_64 ./benchmarks/run-qwen-single-benchmark.sh
```

Most scripts accept extra benchmark arguments through `DELIVERANCE_BENCHMARK_ARGS`. Setting this variable replaces the script's default benchmark argument block.

## Main Scripts

### Qwen Single-Model Benchmark

```sh
./benchmarks/run-qwen-single-benchmark.sh
```

Default model:

```text
Qwen/Qwen3-0.6B
```

Default benchmark options include:

```text
--output-head-quantization Q4
--pool-size 16
--max-tokens 256
--warmup-cases 0
--profile-stages
--output target/deliverance-single-benchmark.csv
--jsonl-output target/deliverance-single-benchmark.jsonl
```

Use this script for quick Qwen3 smoke/performance checks. For QOD-generated models, pass the generated target model name:

```sh
DELIVERANCE_BENCHMARK_ARGS="--output-head-quantization Q4 --pool-size 16 --max-tokens 256 --warmup-cases 0 --profile-stages --output target/qwen4b-qod.csv --jsonl-output target/qwen4b-qod.jsonl" \
./benchmarks/run-qwen-single-benchmark.sh
```

If you need to change owner/model, run `InferenceBenchmark` directly or adjust the script; the script currently hard-codes `--owner Qwen --model Qwen3-0.6B` before `DELIVERANCE_BENCHMARK_ARGS`.

### Gemma2 Single-Model Benchmark

```sh
./benchmarks/run-deliverance-single-benchmark.sh
```

Default model:

```text
tjake/gemma-2-2b-it-JQ4
```

This is the baseline single-process native SIMD benchmark. It writes CSV and JSONL outputs under `core/target` by default.

### Gemma2 Tensor-Parallel Benchmark

```sh
./benchmarks/run-deliverance-benchmark.sh
```

Default tensor-parallel options include:

```text
--tensor-parallel-size 4
--tensor-parallel-max-ranks-per-worker 2
--output-head-quantization Q4
--pool-size 16
--max-tokens 256
--warmup-cases 0
--profile-stages
```

This runs the local in-process tensor-parallel path with HTTP collectives. It is useful for checking tensor-parallel overhead and rank/coordinator behavior without starting separate worker JVMs.

### Netty Tensor-Parallel Benchmark

```sh
./run-deliverance-netty-benchmark.sh
```

This is the same broad shape as `benchmarks/run-deliverance-benchmark.sh`, but uses:

```text
--tensor-parallel-collective-transport netty
```

Use it when comparing collective transport overhead.

### Panama Benchmark

```sh
./run-deliverance-panama-benchmark.sh
```

This script does not set `java.library.path` or `-Dexec.classpathScope=test`. It is useful when you want to compare the Panama/vector path rather than native SIMD.

### Mixtral Benchmark

```sh
./run-deliverance-mixtral-benchmark.sh
```

Default model:

```text
tjake/Mixtral-8x7B-Instruct-v0.1-JQ4
```

Default max tokens are lower, `128`, because the model is much larger.

## Running A Different Model

For anything beyond the script defaults, call the benchmark runner directly. Example for a QOD-generated Qwen3 4B target:

```sh
mvn -q -pl core \
  -Dexec.classpathScope=test \
  -Dexec.executable=java \
  -Dexec.args="-Djava.library.path=native/target/native-lib-only/osx-aarch_64 --add-modules jdk.incubator.vector,jdk.httpserver,java.net.http --add-opens java.base/java.nio=ALL-UNNAMED --enable-native-access=ALL-UNNAMED -cp %classpath io.teknek.deliverance.benchmark.InferenceBenchmark --engine deliverance --owner Qwen --model Qwen3-4B-JQ4 --output-head-quantization Q4 --pool-size 16 --max-tokens 256 --warmup-cases 0 --profile-stages --output target/qwen3-4b-jq4.csv --jsonl-output target/qwen3-4b-jq4.jsonl" \
  org.codehaus.mojo:exec-maven-plugin:3.5.0:exec
```

Use `exec:exec`, not `exec:java`, so the forked JVM receives the `--add-modules` and native-access flags.

## QOD Benchmark Workflow

The benchmark runner loads model names from the Deliverance cache. It does not currently create a QOD target itself.

The practical QOD benchmark workflow is:

1. Generate the QOD target once through a test or small harness using `withQuantizeOnDemand(...)`.
2. Verify the generated cache directory exists, such as `~/.deliverance/Qwen_Qwen3-4B-JQ4`.
3. Benchmark the generated model by passing `--owner Qwen --model Qwen3-4B-JQ4`.
4. Use `--output-head-quantization Q4` if you want load-time output-head Q4 too.
5. Compare against dense or non-output-head runs with the same prompt suite and token limits.

QOD conversion time should not be mixed into inference throughput numbers unless the benchmark question is specifically "first-run model preparation plus inference".

## Output Files

CSV output is for aggregate analysis. Important columns include:

- `engine`
- `model`
- `case_id`
- `category`
- `turn`
- `prompt_tokens`
- `generated_tokens`
- `total_ms`
- `generation_ms`
- `tokens_per_second`
- `finish_reason`

JSONL output is for qualitative review. Each line records the prompt/request, response text, special-token response text, timing, and metadata for one turn. JSONL is flushed after every turn, so interrupted benchmark runs still preserve partial transcripts.

## Console Output

Each recorded turn prints a compact progress line like:

```text
[deliverance] model=Qwen/Qwen3-0.6B-JQ4 case=builtin-reasoning-1 category=reasoning turn=2 prompt_tokens=405 generated=256 total_ms=13460.8 tok_s=24.62 finish=MAX_TOKENS
```

Read this as:

- `prompt_tokens`: rendered prompt length in model tokens.
- `generated`: generated token count.
- `total_ms`: full request wall time.
- `tok_s`: decode-rate estimate, not including all prefill behavior.
- `finish`: `STOP_TOKEN`, `MAX_TOKENS`, or another finish reason.

With `--profile-stages`, the runner prints stage timing after each turn:

```text
[profile] causalselfattention.forward count=7196 total_ms=7779.666 mean_us=1081.110
[profile] sampler.output_projection   count=256  total_ms=1808.581 mean_us=7064.769
[profile-counter] sampler.output_weight_Q4 count=256
```

These profiles are useful for identifying whether time is going into attention, MLP, output projection, sampling, prefill, or decode. Counters also confirm dtype paths, for example whether the output projection actually used Q4 weights.

## Interpreting Token Rates

Deliverance response objects include total time and time-to-first-token. For short generations, total average milliseconds per token can look worse because it includes prefill and first-token setup.

For a more useful decode estimate:

```text
decode_ms = total_ms - time_to_first_token_ms
decode_tok_s = generated_tokens / (decode_ms / 1000.0)
```

Use the CSV `tokens_per_second` column for consistent comparisons across benchmark runs, and use JSONL transcripts to verify output quality.

## Benchmark Hygiene

- Keep model, prompt suite, `max_tokens`, temperature, output-head quantization, native/Panama path, and pool size fixed when comparing runs.
- Use `--warmup-cases` when measuring steady-state behavior.
- Use `--warmup-cases 0` when you want to see cold-ish first measured cases.
- Do not run multiple Maven benchmark processes against the same module target directory while diagnosing classpath or build issues.
- Native build outputs are classifier-specific; make sure the script is using the same platform classifier you built.
- Review JSONL transcripts because higher token/sec is not useful if the model quality regresses.
