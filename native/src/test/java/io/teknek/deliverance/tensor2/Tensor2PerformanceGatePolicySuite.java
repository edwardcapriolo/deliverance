package io.teknek.deliverance.tensor2;

import org.junit.platform.suite.api.SelectClasses;
import org.junit.platform.suite.api.Suite;

/** Focused deterministic tests for tensor2 performance-gate decisions. */
@Suite
@SelectClasses(Tensor2PerformanceGatePolicyTest.class)
public class Tensor2PerformanceGatePolicySuite {
}
