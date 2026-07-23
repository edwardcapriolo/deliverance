package io.teknek.deliverance.antares;

final class CwePrompts {
    private CwePrompts() {
    }

    static String analysisPrompt(String cwe, String query) {
        if (cwe == null || cwe.isBlank()) {
            return query;
        }
        String normalized = cwe.strip().toUpperCase();
        String context = switch (normalized) {
            case "CWE-78" -> "CWE-78: Improper Neutralization of Special Elements used in an OS Command ('OS Command Injection')\n"
                    + "Likelihood of Exploit: High\n"
                    + "The product constructs all or part of an OS command using externally-influenced input, "
                    + "but it does not neutralize or incorrectly neutralizes special elements that could modify the intended command.";
            default -> normalized;
        };
        String base = "Analyze this codebase for the following vulnerability class:\n\n"
                + context
                + "\n\nUse the terminal tool to explore and determine if this vulnerability exists. "
                + "Then either submit the vulnerable file(s) or declare no vulnerability found.";
        if (query == null || query.isBlank()) {
            return base;
        }
        return base + "\n\nAdditional instructions:\n" + query;
    }
}
