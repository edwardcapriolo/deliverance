package io.teknek.deliverance.tensor2;

import org.junit.jupiter.api.extension.AfterTestExecutionCallback;
import org.junit.jupiter.api.extension.BeforeTestExecutionCallback;
import org.junit.jupiter.api.extension.ExtensionContext;

/** Publishes the exact JUnit invocation ID to the tensor timing collector. */
final class Tensor2PerformanceCaseExtension implements BeforeTestExecutionCallback, AfterTestExecutionCallback {
    @Override
    public void beforeTestExecution(ExtensionContext context) {
        String method = context.getTestMethod().map(java.lang.reflect.Method::getName)
                .orElse(context.getDisplayName());
        String display = context.getDisplayName()
                .replace(",", "%2C")
                .replace("\n", "%0A")
                .replace("\r", "%0D");
        Tensor2TimingCsv.setTestIdentity(method + "::" + display, context.getUniqueId());
    }

    @Override
    public void afterTestExecution(ExtensionContext context) {
        Tensor2TimingCsv.clearTestIdentity();
    }
}
