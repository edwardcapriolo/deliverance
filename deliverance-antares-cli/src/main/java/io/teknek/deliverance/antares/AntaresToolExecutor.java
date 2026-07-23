package io.teknek.deliverance.antares;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.TimeUnit;
import java.util.regex.Pattern;

final class AntaresToolExecutor {
    private static final int DEFAULT_MAX_CHARS = 4_000;
    private static final int ABSOLUTE_MAX_CHARS = 20_000;
    private static final Pattern UNSAFE_SHELL = Pattern.compile("(^|\\s)(>|>>|<|tee|xargs|curl|wget|python|python3|perl|ruby|node|npm|mvn|gradle|git|ssh|scp|nc|dd|rm|mv|cp|chmod|chown|mkdir|touch|truncate|sed\\s+-i)(\\s|$)");
    private static final Set<String> ALLOWED_COMMANDS = Set.of(
            "ls", "tree", "find", "cat", "head", "tail", "sed", "grep", "rg", "wc", "sort", "uniq", "cut",
            "file", "stat", "du", "pwd", "nl", "basename", "dirname", "realpath", "diff", "echo", "true", "false");

    private final Path repoRoot;
    private final int maxToolCalls;
    private final ToolApproval toolApproval;
    private int toolCallsUsed;
    private ToolCall lastToolCall;

    AntaresToolExecutor(Path repoRoot, int maxToolCalls) throws IOException {
        this(repoRoot, maxToolCalls, ToolApproval.approveAll());
    }

    AntaresToolExecutor(Path repoRoot, int maxToolCalls, ToolApproval toolApproval) throws IOException {
        this.repoRoot = repoRoot.toRealPath();
        this.maxToolCalls = maxToolCalls;
        this.toolApproval = toolApproval;
        if (!Files.isDirectory(this.repoRoot)) {
            throw new IOException("Repository path is not a directory: " + this.repoRoot);
        }
    }

    int toolCallsUsed() {
        return toolCallsUsed;
    }

    Path repoRoot() {
        return repoRoot;
    }

    ToolCall lastToolCall() {
        return lastToolCall;
    }

    ToolResult execute(ToolCall call) {
        String name = call.name();
        if ("submit_vulnerable_files".equals(name)) {
            return submitVulnerableFiles(call.arguments());
        }
        if ("submit_no_vulnerability_found".equals(name)) {
            return ToolResult.submitted(AgentResult.none());
        }
        if (toolCallsUsed >= maxToolCalls) {
            return ToolResult.response("Terminal call budget exhausted (" + maxToolCalls + "/" + maxToolCalls
                    + "). Submit your answer now.");
        }
        if ("terminal".equals(name) || "bash".equals(name)) {
            toolCallsUsed++;
            lastToolCall = call;
            return ToolResult.response(runTerminal(call.arguments()));
        }
        if ("read_file".equals(name)) {
            toolCallsUsed++;
            lastToolCall = call;
            return ToolResult.response(readFile(call.arguments()));
        }
        return ToolResult.response("Tool " + name + " failed: unknown tool. Available tools: terminal, read_file, "
                + "submit_vulnerable_files, submit_no_vulnerability_found.");
    }

    private ToolResult submitVulnerableFiles(Map<String, Object> args) {
        List<String> files = rankedFiles(args);
        List<String> invalid = new ArrayList<>();
        List<String> valid = new ArrayList<>();
        for (String file : files) {
            String error = submittedFileValidationError(file);
            if (error == null) {
                valid.add(file);
            } else {
                invalid.add(file + " (" + error + ")");
            }
        }
        if (!invalid.isEmpty() || valid.isEmpty()) {
            return ToolResult.response("submit_vulnerable_files failed: ranked_files must be existing repository-relative file paths, "
                    + "not globs, placeholders, or descriptions. Invalid entries: " + invalid);
        }
        return ToolResult.submitted(AgentResult.vulnerable(valid));
    }

    private String runTerminal(Map<String, Object> args) {
        String command = stringArg(args, "command", "").trim();
        int maxChars = intArg(args, "max_chars", DEFAULT_MAX_CHARS);
        if (maxChars <= 0) {
            maxChars = DEFAULT_MAX_CHARS;
        }
        maxChars = Math.min(maxChars, ABSOLUTE_MAX_CHARS);
        if (command.isBlank()) {
            return "Tool terminal failed: empty command";
        }
        if (!toolApproval.approveTerminalCommand(repoRoot, command)) {
            return "Tool terminal declined by user: " + command;
        }
        ProcessBuilder pb = new ProcessBuilder("/bin/sh", "-lc", command);
        pb.directory(repoRoot.toFile());
        try {
            Process process = pb.start();
            boolean finished = process.waitFor(10, TimeUnit.SECONDS);
            if (!finished) {
                process.destroyForcibly();
                return "Tool terminal failed: command timed out after 10 seconds";
            }
            int exitCode = process.exitValue();
            String stdout = new String(process.getInputStream().readAllBytes(), StandardCharsets.UTF_8);
            String stderr = new String(process.getErrorStream().readAllBytes(), StandardCharsets.UTF_8);
            String output = stdout;
            if (!stderr.isBlank()) {
                output += "\n[stderr]: " + stderr;
            }
            if (output.isBlank()) {
                output = emptyCommandResult(command, exitCode);
            } else if (exitCode != 0) {
                output += "\n[exit=" + exitCode + "]";
            }
            return truncate(output, maxChars);
        } catch (IOException e) {
            return "Tool terminal failed: " + e.getMessage();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            return "Tool terminal failed: interrupted";
        }
    }

    private String readFile(Map<String, Object> args) {
        String pathText = firstStringArg(args, "path", "file_path", "file");
        if (pathText == null || pathText.isBlank()) {
            return "Tool read_file failed: missing path";
        }
        try {
            Path path = resolveRepoPath(pathText);
            if (!Files.isRegularFile(path)) {
                return "Tool read_file failed: not a regular file: " + pathText;
            }
            List<String> lines = Files.readAllLines(path, StandardCharsets.UTF_8);
            int start = Math.max(1, intArg(args, "start_line", 1));
            int end = Math.min(lines.size(), intArg(args, "end_line", lines.size()));
            if (end < start) {
                return "";
            }
            StringBuilder out = new StringBuilder();
            for (int i = start; i <= end; i++) {
                out.append(i).append(": ").append(lines.get(i - 1)).append('\n');
            }
            return truncate(out.toString(), ABSOLUTE_MAX_CHARS);
        } catch (IOException e) {
            return "Tool read_file failed: " + e.getMessage();
        }
    }

    Path resolveRepoPath(String pathText) throws IOException {
        Path normalized = repoRoot.resolve(pathText).normalize();
        if (!normalized.startsWith(repoRoot)) {
            throw new IOException("path escapes repository root: " + pathText);
        }
        Path resolved = normalized.toRealPath();
        if (!resolved.startsWith(repoRoot)) {
            throw new IOException("path escapes repository root: " + pathText);
        }
        return resolved;
    }

    String terminalSafetyError(String command) {
        if (command.isBlank()) {
            return "empty command";
        }
        String lower = command.toLowerCase(Locale.ROOT);
        if (UNSAFE_SHELL.matcher(lower).find()) {
            return "command contains a blocked write, interpreter, network, or shell redirection token";
        }
        for (String segment : shellSegments(lower)) {
            String trimmed = segment.trim();
            if (trimmed.isEmpty()) {
                continue;
            }
            String first = trimmed.split("\\s+", 2)[0];
            if (!ALLOWED_COMMANDS.contains(first)) {
                return "command starts with blocked executable: " + first;
            }
        }
        return null;
    }

    private List<String> shellSegments(String command) {
        List<String> segments = new ArrayList<>();
        StringBuilder current = new StringBuilder();
        boolean singleQuoted = false;
        boolean doubleQuoted = false;
        boolean escaped = false;
        for (int i = 0; i < command.length(); i++) {
            char c = command.charAt(i);
            if (escaped) {
                current.append(c);
                escaped = false;
                continue;
            }
            if (c == '\\') {
                current.append(c);
                escaped = true;
                continue;
            }
            if (c == '\'' && !doubleQuoted) {
                singleQuoted = !singleQuoted;
                current.append(c);
                continue;
            }
            if (c == '"' && !singleQuoted) {
                doubleQuoted = !doubleQuoted;
                current.append(c);
                continue;
            }
            if (!singleQuoted && !doubleQuoted) {
                if (c == ';' || c == '|') {
                    segments.add(current.toString());
                    current.setLength(0);
                    if (c == '|' && i + 1 < command.length() && command.charAt(i + 1) == '|') {
                        i++;
                    }
                    continue;
                }
                if (c == '&' && i + 1 < command.length() && command.charAt(i + 1) == '&') {
                    segments.add(current.toString());
                    current.setLength(0);
                    i++;
                    continue;
                }
            }
            current.append(c);
        }
        segments.add(current.toString());
        return segments;
    }

    private List<String> rankedFiles(Map<String, Object> args) {
        Object value = args.get("ranked_files");
        if (value == null) {
            value = args.get("files");
        }
        if (value == null) {
            value = args.get("file_paths");
        }
        LinkedHashSet<String> files = new LinkedHashSet<>();
        if (value instanceof List<?> list) {
            for (Object item : list) {
                if (item != null && !item.toString().isBlank()) {
                    addRankedFile(files, item.toString());
                }
            }
        } else if (value != null && !value.toString().isBlank()) {
            addRankedFile(files, value.toString());
        }
        return new ArrayList<>(files);
    }

    private void addRankedFile(LinkedHashSet<String> files, String path) {
        String normalized = path.strip();
        files.add(normalized);
    }

    String submittedFileValidationError(String path) {
        if (path == null || path.isBlank()) {
            return "blank path";
        }
        String normalized = path.strip();
        if (normalized.equals("path/to/file") || normalized.equals("path/to/vulnerable_file")) {
            return "placeholder path";
        }
        if (normalized.contains("*") || normalized.contains("?") || normalized.contains("[")
                || normalized.contains("]") || normalized.contains("{") || normalized.contains("}")) {
            return "glob patterns are not file paths";
        }
        if (normalized.startsWith("/") || normalized.contains("..")) {
            return "path must stay inside repository";
        }
        try {
            Path resolved = resolveRepoPath(normalized);
            if (!Files.isRegularFile(resolved)) {
                return "not a regular file";
            }
            return null;
        } catch (IOException e) {
            return e.getMessage();
        }
    }

    private static String stringArg(Map<String, Object> args, String name, String defaultValue) {
        Object value = args.get(name);
        return value == null ? defaultValue : value.toString();
    }

    private static String firstStringArg(Map<String, Object> args, String... names) {
        for (String name : names) {
            Object value = args.get(name);
            if (value != null) {
                return value.toString();
            }
        }
        return null;
    }

    private static int intArg(Map<String, Object> args, String name, int defaultValue) {
        Object value = args.get(name);
        if (value instanceof Number number) {
            return number.intValue();
        }
        if (value != null) {
            return Integer.parseInt(value.toString());
        }
        return defaultValue;
    }

    private static String truncate(String value, int maxChars) {
        if (value.length() <= maxChars) {
            return value;
        }
        return value.substring(0, maxChars) + "\n\n[OUTPUT TRUNCATED: showing first " + maxChars
                + " characters. Use read_file with line ranges or narrower terminal commands.]";
    }

    private static String emptyCommandResult(String command, int exitCode) {
        String trimmed = command.stripLeading();
        if (trimmed.startsWith("rg ") && exitCode == 1) {
            return "[empty output; exit=1, ripgrep found no matches]";
        }
        return "[empty output; exit=" + exitCode + "]";
    }
}
