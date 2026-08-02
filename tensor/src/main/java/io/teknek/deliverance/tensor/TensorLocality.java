package io.teknek.deliverance.tensor;

import java.util.List;
import java.util.Objects;

/**
 * Advisory locality metadata for a tensor's backing memory.
 *
 * <p>This metadata is an observation used by schedulers. It is not a correctness contract: operating systems may move
 * pages and unsupported platforms may report unknown locality.</p>
 */
public record TensorLocality(
        long virtualAddress,
        long byteSize,
        int numaNode,
        List<Integer> preferredCpus,
        long observedAtMillis,
        String source
) {
    public static final int UNKNOWN_NUMA_NODE = -1;

    public TensorLocality {
        preferredCpus = List.copyOf(Objects.requireNonNull(preferredCpus, "preferredCpus"));
        source = Objects.requireNonNull(source, "source");
    }

    public boolean numaKnown() {
        return numaNode >= 0;
    }
}
