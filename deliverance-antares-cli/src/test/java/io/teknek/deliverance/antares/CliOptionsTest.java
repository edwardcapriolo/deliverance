package io.teknek.deliverance.antares;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class CliOptionsTest {
    @Test
    void terminalApprovalIsDefaultUnlessExplicitlyDisabled() {
        CliOptions defaults = CliOptions.parse(new String[]{"--repo", "."});
        CliOptions unattended = CliOptions.parse(new String[]{"--repo", ".", "--yes-run-commands"});

        assertFalse(defaults.yesRunCommands);
        assertTrue(unattended.yesRunCommands);
    }
}
