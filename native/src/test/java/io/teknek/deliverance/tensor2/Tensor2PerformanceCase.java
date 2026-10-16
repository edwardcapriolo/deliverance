package io.teknek.deliverance.tensor2;

import org.junit.jupiter.api.extension.ExtendWith;

import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

/** Marks tests whose tensor timing rows must retain their JUnit invocation identity. */
@Target({ElementType.METHOD, ElementType.TYPE})
@Retention(RetentionPolicy.RUNTIME)
@ExtendWith(Tensor2PerformanceCaseExtension.class)
public @interface Tensor2PerformanceCase {
}
