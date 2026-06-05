package io.teknek.deliverance.model;

/**
 * Explicit generation key/value state.
 *
 * <p>Committed KV positions should be treated as immutable. Writes should happen through scoped append/write operations
 * rather than arbitrary read/write tensor access.</p>
 *
 * <p>This interface is introduced as a typed generation-boundary concept first. Concrete storage and write-session APIs
 * should be added deliberately.</p>
 */
public interface PastKeyValues extends AutoCloseable {

    /**
     * Returns the number of committed token positions available for reads.
     *
     * <p>The value is a logical sequence length, not a storage capacity. For a cache containing positions
     * {@code 0..N-1}, this method returns {@code N}. Implementations should advance this value only after KV writes for
     * a position are committed. Uncommitted or partially written positions must not be counted.</p>
     */
    int sequenceLength();

    /**
     * Releases any memory, mapped files, or other resources owned by this cache state.
     */
    @Override
    void close();
}
