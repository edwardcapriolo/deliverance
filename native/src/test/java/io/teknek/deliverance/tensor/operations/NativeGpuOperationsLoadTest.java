package io.teknek.deliverance.tensor.operations;

import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class NativeGpuOperationsLoadTest {

    @Test
    public void nativeGpuOperationsLoadsAndReportsProviderName() {
        Assumptions.assumeTrue(hasNativeLibrary("webgpu_dawn"),
                "webgpu_dawn native library is not available on java.library.path");
        Assumptions.assumeTrue(hasNativeLibrary("deliverancegpu"),
                "deliverancegpu native library is not available on java.library.path");
        NativeGPUTensorOperations operations = new NativeGPUTensorOperations();

        assertEquals("Native GPU Operations", operations.name());
        assertEquals(1, operations.parallelSplitSize());
    }

    private static boolean hasNativeLibrary(String libname) {
        String mapped = System.mapLibraryName(libname);
        String libraryPath = System.getProperty("java.library.path", "");
        return Arrays.stream(libraryPath.split(java.io.File.pathSeparator))
                .map(Path::of)
                .anyMatch(path -> Files.isRegularFile(path.resolve(mapped)));
    }
}
