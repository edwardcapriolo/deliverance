package io.teknek.deliverance.tensor2;

import org.junit.platform.suite.api.SelectClasses;
import org.junit.platform.suite.api.Suite;

/** Entry point for persistent tensor2 performance gates. */
@Suite
@SelectClasses(Tensor2PerformanceGateTest.class)
public class Tensor2PerformanceGateSuite {
}
