package io.teknek.deliverance.math;


@FunctionalInterface
public interface BiIntConsumer {
    /**
     * Performs this operation on the given arguments.
     *
     * @param value the input argument
     */
    void accept(int value, int value2);
}