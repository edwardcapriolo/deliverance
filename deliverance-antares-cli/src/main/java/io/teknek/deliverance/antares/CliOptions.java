package io.teknek.deliverance.antares;

import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;

final class CliOptions {
    final Path repo;
    final String endpoint;
    final String model;
    final String cwe;
    final String query;
    final int maxToolCalls;
    final int maxSubmitTurns;
    final int maxIterations;
    final int maxTokens;
    final float temperature;
    final float topP;
    final boolean yesRunCommands;

    private CliOptions(Path repo, String endpoint, String model, String cwe, String query, int maxToolCalls, int maxSubmitTurns,
            int maxIterations, int maxTokens, float temperature, float topP, boolean yesRunCommands) {
        this.repo = repo;
        this.endpoint = endpoint;
        this.model = model;
        this.cwe = cwe;
        this.query = query;
        this.maxToolCalls = maxToolCalls;
        this.maxSubmitTurns = maxSubmitTurns;
        this.maxIterations = maxIterations;
        this.maxTokens = maxTokens;
        this.temperature = temperature;
        this.topP = topP;
        this.yesRunCommands = yesRunCommands;
    }

    static CliOptions parse(String[] args) {
        Map<String, String> values = new HashMap<>();
        for (int i = 0; i < args.length; i++) {
            String arg = args[i];
            if ("--help".equals(arg) || "-h".equals(arg)) {
                throw new UsageException(usage());
            }
            if ("--yes-run-commands".equals(arg)) {
                values.put("yes-run-commands", "true");
                continue;
            }
            if (!arg.startsWith("--") || i + 1 >= args.length) {
                throw new UsageException(usage());
            }
            values.put(arg.substring(2), args[++i]);
        }
        String repo = required(values, "repo");
        String endpoint = values.getOrDefault("endpoint", "http://127.0.0.1:18085/v1");
        String model = values.getOrDefault("model", "antares-1b-JQ4");
        String query = values.getOrDefault("query", defaultQuery());
        String cwe = values.getOrDefault("cwe", "");
        return new CliOptions(
                Path.of(repo),
                endpoint,
                model,
                cwe,
                query,
                parseInt(values, "max-tool-calls", 20),
                parseInt(values, "max-submit-turns", 3),
                parseInt(values, "max-iterations", 50),
                parseInt(values, "max-tokens", 4096),
                parseFloat(values, "temperature", 0.3f),
                parseFloat(values, "top-p", 1.0f),
                Boolean.parseBoolean(values.getOrDefault("yes-run-commands", "false")));
    }

    private static String required(Map<String, String> values, String name) {
        String value = values.get(name);
        if (value == null || value.isBlank()) {
            throw new UsageException("Missing --" + name + "\n" + usage());
        }
        return value;
    }

    private static int parseInt(Map<String, String> values, String name, int defaultValue) {
        String value = values.get(name);
        return value == null ? defaultValue : Integer.parseInt(value);
    }

    private static float parseFloat(Map<String, String> values, String name, float defaultValue) {
        String value = values.get(name);
        return value == null ? defaultValue : Float.parseFloat(value);
    }

    private static String defaultQuery() {
        return "Search this repository for security vulnerabilities. Focus on CWE-89 SQL Injection, "
                + "CWE-78 Command Injection, CWE-79 XSS, CWE-798 Hardcoded Credentials, CWE-22 Path Traversal, "
                + "CWE-502 Deserialization, and CWE-306 Missing Authentication. Read source files and submit ranked vulnerable file paths only.";
    }

    static String usage() {
        return "usage: java -jar deliverance-antares-cli.jar --repo <path> [--endpoint http://host:port/v1] "
                + "[--model antares-1b-JQ4] [--cwe CWE-78] [--query text] [--max-tool-calls 20] [--max-submit-turns 3] [--max-iterations 50] [--max-tokens 4096] [--yes-run-commands]\n"
                + "By default, the CLI asks for y/N approval before every model-requested terminal command. "
                + "Use --yes-run-commands only in a sandbox/container.";
    }

    static final class UsageException extends RuntimeException {
        UsageException(String message) {
            super(message);
        }
    }
}
