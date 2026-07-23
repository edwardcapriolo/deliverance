# Antares Nanocode Showcase

This demo runs the local quantized Antares-1B JQ4 model behind the Deliverance OpenAI-compatible server and uses `nanocode-deliverance` as a tiny terminal security agent.

Antares is trained for vulnerability localization: given a CWE category and a repository, it should inspect files through terminal-style tools and return candidate vulnerable file paths for human review.

## Start Antares

```sh
mvn -pl web,nanocode-deliverance -am -DskipTests package
cd nanocode-deliverance
sh run-server-antares-1b.sh
```

The server defaults to `http://127.0.0.1:18085` and loads `edwardcapriolo/antares-1b-JQ4` from the Deliverance model cache.

## Run Nanocode

In another terminal:

```sh
cd nanocode-deliverance
sh run_antares_client.sh
```

Quick server check:

```sh
curl -s -H 'Content-Type: application/json' http://127.0.0.1:18085/models
```

## Security Finding Showcase

This module includes a tiny intentionally vulnerable repository at:

```text
nanocode-deliverance/showcase-security-repo
```

Run this prompt in the Nanocode REPL:

```text
Security localization task:

Repository: ./showcase-security-repo

CWE-78: Improper Neutralization of Special Elements used in an OS Command ('OS Command Injection'). The software constructs all or part of an OS command using externally influenced input, but does not neutralize special elements that could modify the intended command.

Use the available tools to inspect the repository. Return the file path or paths most likely to contain the vulnerability and cite the specific evidence you found. Do not provide exploit instructions.
```

Expected finding:

```text
showcase-security-repo/src/main/java/demo/ArchiveController.java
```

Why: it concatenates `userName` into a shell command string passed to `Runtime.getRuntime().exec(...)`.

## Additional Prompts

Ask Antares to inspect Nanocode's own tool implementation:

```text
Find the Java files in nanocode-deliverance that implement tools, then summarize the available tools.
```

Ask Antares to run a simple shell command. `config-antares-1b.json` explicitly enables `bash`, so keep this for trusted local demos only:

```text
Use bash to print the current directory and list the first five files, then explain what command you ran.
```

Ask Antares to verify code behavior without shell access by using the regular Nanocode file/search tools:

```text
Search for the bash tool implementation and tell me what safety gate prevents it from running by default.
```

Expected behavior: Antares should emit tool calls, Nanocode should execute them, and the final answer should summarize observed file or command evidence instead of hallucinating it.
