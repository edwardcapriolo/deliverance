package io.teknek.deliverance.antares;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.file.Path;
import java.util.Locale;

final class ConsoleToolApproval implements ToolApproval {
    private final BufferedReader input = new BufferedReader(new InputStreamReader(System.in));

    @Override
    public boolean approveTerminalCommand(Path repoRoot, String command) {
        System.err.println("[approval] repo: " + repoRoot);
        System.err.println("[approval] run terminal command?");
        System.err.println(command);
        System.err.print("[approval] y/N: ");
        System.err.flush();
        try {
            String answer = input.readLine();
            if (answer == null) {
                return false;
            }
            String normalized = answer.trim().toLowerCase(Locale.ROOT);
            return "y".equals(normalized) || "yes".equals(normalized);
        } catch (IOException e) {
            return false;
        }
    }
}
