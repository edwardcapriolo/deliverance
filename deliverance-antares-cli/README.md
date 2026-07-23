# Deliverance Antares CLI

`deliverance-antares-cli` is a Java command-line helper for running Antares-style vulnerability localization against a Deliverance `/v1/completions` server.

It streams the model's investigation, shows tool calls as they happen, asks before running terminal commands, and returns exact vulnerable file paths.

## Quick Demo

Start the Antares-1B JQ4 server from the Deliverance repository root:

```bash
./deliverance-antares-cli/run-server-antares-1b.sh
```

Run the bundled CWE-78 showcase in another terminal:

```bash
./deliverance-antares-cli/run-showcase-cwe78.sh
```

Equivalent direct command:

```bash
java -jar deliverance-antares-cli/target/deliverance-antares-cli-0.0.12-SNAPSHOT-all.jar \
  --repo nanocode-deliverance/showcase-security-repo \
  --endpoint http://127.0.0.1:18085/v1 \
  --model antares-1b-JQ4 \
  --cwe CWE-78 \
  --query "Look for user-controlled input reaching Runtime.exec, ProcessBuilder, shell commands, or command string concatenation." \
  --max-tool-calls 4
```

The showcase repository contains this vulnerable Java code:

```java
public Process createArchive(String userName) throws IOException {
    String command = "tar -czf /tmp/" + userName + ".tgz /srv/uploads/" + userName;
    return Runtime.getRuntime().exec(command);
}
```

Condensed successful run:

```text
[antares] repo root .../nanocode-deliverance/showcase-security-repo
[antares] model turn 1, tool calls 0/8
I’ll first identify the language and major files, then search for OS command execution APIs and command-line construction patterns.

[tool] terminal rg -n "\.exec\(" .
[approval] y/N: y
[result]
./src/main/java/demo/ArchiveController.java:8:        return Runtime.getRuntime().exec(command);

[tool] read_file src/main/java/demo/ArchiveController.java
[result]
1: package demo;
...
6:     public Process createArchive(String userName) throws IOException {
7:         String command = "tar -czf /tmp/" + userName + ".tgz /srv/uploads/" + userName;
8:         return Runtime.getRuntime().exec(command);
9:     }

[tool] submit_vulnerable_files {ranked_files=[src/main/java/demo/ArchiveController.java]}
{
  "vulnerabilityFound" : true,
  "rankedFiles" : [ "src/main/java/demo/ArchiveController.java" ],
  "summary" : "Submitted vulnerable files: [src/main/java/demo/ArchiveController.java]"
}
```

That is the point: Antares uses terminal-style tools to inspect a repository, finds the command-injection sink, reads the file, and submits the vulnerable file path.

## Why This Matters

Cisco's announcement, [Introducing Antares: The Most Efficient Open-Weight AI Models for Vulnerability Localization](https://blogs.cisco.com/ai/introducing-antares-the-most-efficient-open-weight-ai-models-for-vulnerability-localization), describes Antares-1B as an open-weight model specialized for vulnerability localization in real-world codebases and trained to navigate repositories through terminal-style tool use.

That is also a strong validation point for Deliverance: a compact specialized model can run locally behind a Java `/v1/completions` server, inspect a repository through tools, and return a useful file-level security finding.

Cisco also emphasizes the practical deployment angle: security teams need capable tooling that is practical to run, compact models reduce inference costs, local or on-premises operation can keep sensitive source code inside an organization's environment, and AI-assisted security becomes more accessible to universities, public sector institutions, and smaller security teams.

## Safety

Antares can request terminal commands. By default this CLI asks for `y/N` approval before every `terminal` or `bash` command:

```text
[approval] repo: /path/to/repository
[approval] run terminal command?
rg -n "Runtime\.getRuntime|ProcessBuilder|\.exec" .
[approval] y/N:
```

If Antares asks to run something you do not want, answer `n`. The model can recover and continue with safer commands.

Only use unattended mode inside a sandbox/container or another environment you are willing to let the model inspect:

```bash
./deliverance-antares-cli/run-showcase-cwe78.sh --yes-run-commands
```

`read_file` is constrained to repository-relative files. `submit_vulnerable_files` rejects globs, placeholders, directories, missing files, absolute paths, and parent traversal.

## Model And Server

The default Antares server profile uses the uploaded JQ4 model:

```text
edwardcapriolo/antares-1b-JQ4
```

and exposes the OpenAI-compatible model name:

```text
antares-1b-JQ4
```

Quick server check:

```bash
curl -s -H 'Content-Type: application/json' http://127.0.0.1:18085/models
```

This CLI uses raw OpenAI-compatible `/v1/completions`, not chat completions. Antares expects its own prompt/tool format; server-side chat templates change that prompt and are not equivalent.

## What This Module Preserves

- Antares/Granite prompt serialization
- streamed model output so the terminal keeps moving during long generations
- `terminal`, `read_file`, `submit_vulnerable_files`, and `submit_no_vulnerability_found`
- exact repository-relative file validation for submissions
- command approval before model-generated terminal commands by default

## VLoc Smoke Benchmark

Cisco also published the [Vulnerability Localization Benchmark](https://github.com/cisco-foundation-ai/vulnerability-localization-benchmark), a 500-task agentic benchmark scored with file-level F1 and true-negative rate.

This module includes a tiny local smoke runner with the same basic scoring shape: run one repository/CWE task, compare submitted files to expected vulnerable files, and write a JSONL row.

```bash
./deliverance-antares-cli/run-vloc-showcase-benchmark.sh
```

The showcase benchmark expects:

```text
src/main/java/demo/ArchiveController.java
```

The output row includes `precision`, `recall`, `file_f1`, `submitted_files`, `expected_files`, and elapsed time. This is not the full 500-task VLoc Bench harness; it is the first local Deliverance benchmark scaffold for that workflow.

## Upstream CLI

The full-featured reference implementation is the official Python [`antares-cli`](https://github.com/fdtn-ai/antares-cli) project. It includes reports, SARIF, run history, read-only repository snapshots, and investigation traces. This Java module is a compact Deliverance-side CLI for the same local workflow.

To compare with the upstream CLI, configure a profile that points at the Deliverance completions endpoint:

```toml
[profiles.deliverance-antares]
display_name = "Deliverance Antares JQ4"
model = "antares-1b-JQ4"
backend = "remote"
endpoint_env = "ANTARES_ENDPOINT"
context_window = 16384
remote_timeout_seconds = 300

[profiles.deliverance-antares.generation]
max_tokens = 4096
temperature = 0.3
top_p = 1.0
frequency_penalty = 0.3
stop_tokens = ["<|end_of_text|>", "<|start_of_role|>"]
use_completions_api = true
```

Then run:

```bash
export ANTARES_ENDPOINT="http://127.0.0.1:18085/v1/completions"

uv run antares query ../deliverance/nanocode-deliverance/showcase-security-repo \
  --cwe CWE-78 \
  --query "Look for user-controlled input reaching Runtime.exec, ProcessBuilder, shell commands, or command string concatenation." \
  --profile deliverance-antares \
  --tool-budget 4
```
