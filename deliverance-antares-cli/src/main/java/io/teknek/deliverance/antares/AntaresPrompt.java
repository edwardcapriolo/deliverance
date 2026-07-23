package io.teknek.deliverance.antares;

import java.util.List;
import java.util.regex.Pattern;

final class AntaresPrompt {
    private static final String ASSISTANT_PREFILL = "<|start_of_role|>assistant<|end_of_role|><think>\n";
    private static final List<String> GRANITE_CONTROL_TOKENS = List.of(
            "<|start_of_role|>", "<|end_of_role|>", "<|end_of_text|>", "<|endoftext|>", "<|eot_id|>");
    private static final Pattern TOOL_RESPONSE_DELIMITER = Pattern.compile("</?tool_response(?:\\s+[^>]*)?>",
            Pattern.CASE_INSENSITIVE);

    private AntaresPrompt() {
    }

    static List<Message> initialMessages(String query, int terminalCallBudget) {
        return List.of(
                new Message("system", systemPrompt(terminalCallBudget)),
                new Message("user", query));
    }

    static String render(List<Message> messages) {
        StringBuilder sb = new StringBuilder();
        for (Message message : messages) {
            if (sb.length() > 0) {
                sb.append('\n');
            }
            sb.append(serialize(message));
        }
        if (sb.length() > 0) {
            sb.append('\n');
        }
        sb.append(ASSISTANT_PREFILL);
        return sb.toString();
    }

    static Message toolResponse(String response) {
        return new Message("tool_response", response);
    }

    static Message noToolRetry(int iterationIndex) {
        String content;
        if (iterationIndex == 0) {
            content = "You must use tools to investigate before reporting. Start by calling terminal to examine the code. "
                    + "When finished, submit file paths with submit_vulnerable_files or call submit_no_vulnerability_found.";
        } else if (iterationIndex >= 4) {
            content = "Call submit_vulnerable_files with the file paths you identified, "
                    + "or call submit_no_vulnerability_found if you found nothing.";
        } else {
            content = "Continue investigating. Use terminal to read more source files.";
        }
        return new Message("tool_response", content);
    }

    static Message duplicateRetry(boolean forceSubmit) {
        String content;
        if (forceSubmit) {
            content = "You have repeated the same tool calls 3 times. Stop investigating. Summarize what you found and "
                    + "submit file-level results now using submit_vulnerable_files. If you found nothing, call submit_no_vulnerability_found.";
        } else {
            content = "You already called these tools with these exact arguments. Please try a different approach.";
        }
        return new Message("tool_response", content);
    }

    static Message submitOnlyRetry(int terminalCallBudget, List<String> candidateFiles) {
        String candidates = candidateFiles.isEmpty() ? "No candidate files have been recorded."
                : "Observed candidate files: " + candidateFiles;
        return new Message("tool_response", submissionRequiredNudge(terminalCallBudget) + "\n" + candidates);
    }

    static String submissionRequiredNudge(int terminalCallBudget) {
        return "\nERROR: Repository tool budget exhausted (" + terminalCallBudget + "/" + terminalCallBudget + "). "
                + "Submit now using either "
                + "<tool_call>{\"name\":\"submit_vulnerable_files\",\"arguments\":{\"ranked_files\":[\"path/to/file\"]}}</tool_call> or "
                + "<tool_call>{\"name\":\"submit_no_vulnerability_found\",\"arguments\":{}}</tool_call>.";
    }

    private static String serialize(Message message) {
        String role = message.role();
        String content = escapeGraniteControlTokens(message.content());
        if ("assistant".equals(role)) {
            String prefixed = content.startsWith("<think>") ? content : "<think>\n" + content;
            return "<|start_of_role|>assistant<|end_of_role|>" + prefixed + "<|end_of_text|>";
        }
        if ("tool_response".equals(role)) {
            content = TOOL_RESPONSE_DELIMITER.matcher(content).replaceAll("[escaped tool-response delimiter]");
            return "<|start_of_role|>user<|end_of_role|>\n<tool_response>\n" + content
                    + "\n</tool_response><|end_of_text|>";
        }
        return "<|start_of_role|>" + role + "<|end_of_role|>" + content + "<|end_of_text|>";
    }

    private static String escapeGraniteControlTokens(String content) {
        String escaped = content;
        for (String token : GRANITE_CONTROL_TOKENS) {
            String tokenName = token.substring(2, token.length() - 2);
            escaped = escaped.replace(token, "[escaped Granite control token: " + tokenName + "]");
        }
        return escaped;
    }

    private static String systemPrompt(int terminalCallBudget) {
        return "You are a security vulnerability localization agent. You have read-only terminal access to a codebase.\n\n"
                + "Given a CWE (Common Weakness Enumeration) description, your task is to determine whether this codebase "
                + "contains a vulnerability matching that CWE class, and if so, identify which source file(s) are vulnerable.\n\n"
                + "You can explore the codebase using the `terminal` tool — it runs read-only commands "
                + "(ls, find, cat, head, tail, grep, rg, tree, etc.) inside the repository. "
                + "You have up to " + terminalCallBudget + " repository tool calls.\n\n"
                + "When you're done exploring:\n"
                + "- If you found vulnerable file(s): call `submit_vulnerable_files` with a ranked list of file paths "
                + "(most likely vulnerable first).\n"
                + "- If you believe this codebase does NOT contain the described vulnerability: call `submit_no_vulnerability_found`.\n\n"
                + "You may be looking at code that has already been patched — in that case, the correct answer is to submit nothing. "
                + "Do not guess or hallucinate files. Only submit files you have evidence for.\n\n"
                + "NOTE: Submitted paths must be exact file paths (e.g. src/utils.js), never globs or wildcards.\n\n"
                + "You are a helpful assistant with access to the following tools. You may call one or more tools to assist with the user query.\n\n"
                + "You are provided with function signatures within <tools></tools> XML tags:\n"
                + "<tools>\n"
                + "{\"type\":\"function\",\"function\":{\"name\":\"terminal\",\"description\":\"Execute a read-only terminal command in the repository. Allowed: ls, tree, find, cat, head, tail, sed, grep, rg, wc, sort, uniq, cut, file, stat, du, pwd, nl, basename, dirname, realpath, diff, echo, true, false. Pipes and read-only &&, ||, and ; chains are OK. No interpreters, writes, redirects, or network commands.\",\"parameters\":{\"type\":\"object\",\"properties\":{\"command\":{\"type\":\"string\",\"description\":\"The shell command to run\"},\"max_chars\":{\"type\":\"integer\",\"description\":\"Max output chars (default 2000)\",\"default\":2000}},\"required\":[\"command\"]}}}\n"
                + "{\"type\":\"function\",\"function\":{\"name\":\"read_file\",\"description\":\"Read one repository-relative file with optional line bounds.\",\"parameters\":{\"type\":\"object\",\"properties\":{\"path\":{\"type\":\"string\"},\"start_line\":{\"type\":\"integer\"},\"end_line\":{\"type\":\"integer\"}},\"required\":[\"path\"]}}}\n"
                + "{\"type\":\"function\",\"function\":{\"name\":\"submit_vulnerable_files\",\"description\":\"Submit your answer: a ranked list of vulnerable file paths.\",\"parameters\":{\"type\":\"object\",\"properties\":{\"ranked_files\":{\"type\":\"array\",\"items\":{\"type\":\"string\"}}},\"required\":[\"ranked_files\"]}}}\n"
                + "{\"type\":\"function\",\"function\":{\"name\":\"submit_no_vulnerability_found\",\"description\":\"Declare that no vulnerability matching the CWE description was found.\",\"parameters\":{\"type\":\"object\",\"properties\":{},\"required\":[]}}}\n"
                + "</tools>\n\n"
                + "For each tool call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n"
                + "<tool_call>\n"
                + "{\"name\": <function-name>, \"arguments\": <args-json-object>}\n"
                + "</tool_call>. If a tool does not exist in the provided list of tools, "
                + "notify the user that you do not have the ability to fulfill the request.";
    }
}
