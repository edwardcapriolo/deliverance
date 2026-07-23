package io.teknek.deliverance.antares;

import java.util.List;

record AgentResult(boolean vulnerabilityFound, List<String> rankedFiles, String summary) {
    static AgentResult vulnerable(List<String> rankedFiles) {
        return new AgentResult(true, List.copyOf(rankedFiles), "Submitted vulnerable files: " + rankedFiles);
    }

    static AgentResult fallbackVulnerable(List<String> rankedFiles, String reason) {
        return new AgentResult(true, List.copyOf(rankedFiles), reason + " Using observed candidate files: " + rankedFiles);
    }

    static AgentResult none() {
        return new AgentResult(false, List.of(), "Submitted no vulnerability found.");
    }

    static AgentResult incomplete(String summary) {
        return new AgentResult(false, List.of(), summary);
    }
}
