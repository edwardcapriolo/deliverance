# Antares Showcase

This demo runs the local quantized Antares-1B JQ4 model behind the Deliverance OpenAI-compatible `/v1/completions` server and uses `deliverance-antares-cli` as the terminal security agent.

Antares is trained for vulnerability localization: given a CWE category and a repository, it should inspect files through terminal-style tools and return candidate vulnerable file paths for human review.

The current Java CLI lives in [`../deliverance-antares-cli`](../deliverance-antares-cli/README.md). The official upstream reference implementation is the Python `antares-cli` project.

## Start Antares

```sh
mvn -pl web,deliverance-antares-cli -am -DskipTests package
./deliverance-antares-cli/run-server-antares-1b.sh
```

The server defaults to `http://127.0.0.1:18085` and loads `edwardcapriolo/antares-1b-JQ4` from the Deliverance model cache.

Quick server check:

```sh
curl -s -H 'Content-Type: application/json' http://127.0.0.1:18085/models
```

## Run The Showcase

In another terminal:

```sh
./deliverance-antares-cli/run-showcase-cwe78.sh
```

By default the Java CLI asks for `y/N` approval before running every model-requested terminal command. For unattended runs, use only inside a sandbox/container:

```sh
./deliverance-antares-cli/run-showcase-cwe78.sh --yes-run-commands
```

## Security Finding Showcase

This module includes a tiny intentionally vulnerable repository at:

```text
nanocode-deliverance/showcase-security-repo
```

The script runs this CWE-78 query:

```text
Look for user-controlled input reaching Runtime.exec, ProcessBuilder, shell commands, or command string concatenation.
```

Expected finding:

```text
src/main/java/demo/ArchiveController.java
```

Why: it concatenates `userName` into a shell command string passed to `Runtime.getRuntime().exec(...)`.

Expected behavior: Antares should stream progress, request terminal/read-file tools, and submit the exact vulnerable file path. See [`../deliverance-antares-cli/README.md`](../deliverance-antares-cli/README.md) for a condensed demo transcript and Python CLI comparison commands.
