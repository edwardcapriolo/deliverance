package io.teknek.deliverance.nanocode;

import org.junit.jupiter.api.Test;

import java.util.Optional;

import static org.junit.jupiter.api.Assertions.assertEquals;

class JavaSandboxToolTest {

    @Test
    void singleFileModeCompilesOnlySuppliedJavaFileWhenClassNameDiffers() {
        JavaSandboxTool.Command command = JavaSandboxTool.command("single-file", "HelloWorld", Optional.of("main.java"));

        assertEquals("cp 'main.java' 'HelloWorld.java' && javac 'HelloWorld.java' && java 'HelloWorld'", command.value());
    }

    @Test
    void singleFileModeFallsBackToMainClassFileName() {
        JavaSandboxTool.Command command = JavaSandboxTool.command("single-file", "HelloWorld", Optional.empty());

        assertEquals("javac 'HelloWorld.java' && java 'HelloWorld'", command.value());
    }
}
