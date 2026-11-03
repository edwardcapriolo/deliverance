package io.teknek.reckoning.antares;

import java.nio.file.Path;

interface ToolApproval {
    boolean approveTerminalCommand(Path repoRoot, String command);

    static ToolApproval approveAll() {
        return (repoRoot, command) -> true;
    }
}
