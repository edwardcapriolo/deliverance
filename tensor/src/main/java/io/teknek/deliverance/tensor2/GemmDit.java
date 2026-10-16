package io.teknek.deliverance.tensor2;

import io.teknek.deliverance.DType;
import io.teknek.deliverance.tensorlib.TensorPlanAdaptiveSplitTuner;

import java.util.List;
import java.util.Map;

/** Adaptive selection of Tensor2 row-GEMM execution strategies. */
public final class GemmDit {
    private static final List<Candidate> CANDIDATES = List.of(
            new Candidate("ROW/NONE", GemmKernel.ROW, Threading.NONE),
            new Candidate("ROW/PROVIDER", GemmKernel.ROW, Threading.PROVIDER),
            new Candidate("CHUNK/NONE", GemmKernel.CHUNK, Threading.NONE),
            new Candidate("CHUNK/PROVIDER", GemmKernel.CHUNK, Threading.PROVIDER));

    private GemmDit() {
    }

    public static void dotProductRows(Lighter lighter, TensorPlanAdaptiveSplitTuner tuner, String planName,
            TensorRef output, TensorRef input, TensorRef weights, int inputColumnStart, int weightColumnStart,
            int columnLength, int weightRowStart, int weightRowCount, int outputColumnStart) {
        if (tuner == null || output.dType() != DType.F32
                || (input.dType() != DType.F32 && input.dType() != DType.I8)
                || weights.dType() != DType.Q4) {
            lighter.dotProductRows(output, input, weights, inputColumnStart, weightColumnStart, columnLength,
                    weightRowStart, weightRowCount, outputColumnStart);
            return;
        }
        List<String> names = candidateNames(input.dType(), weights.dType(), inputColumnStart, weightColumnStart,
                columnLength);
        String selected = tuner.chooseAlternate(planName, names);
        Candidate candidate = CANDIDATES.stream().filter(value -> value.name().equals(selected)).findFirst()
                .orElse(CANDIDATES.getFirst());
        long start = System.nanoTime();
        try {
            lighter.dotProductRows(output, input, weights, inputColumnStart, weightColumnStart, columnLength,
                    weightRowStart, weightRowCount, outputColumnStart, candidate.kernel(), candidate.threading());
            tuner.observeAlternate(planName, candidate.name(), System.nanoTime() - start,
                    Map.of("kernel", candidate.kernel().name(), "threading", candidate.threading().name(),
                            "input", input.dType().name(), "weights", weights.dType().name(),
                            "rows", Integer.toString(weightRowCount)));
        } catch (RuntimeException | Error error) {
            tuner.observeAlternate(planName, candidate.name(), Long.MAX_VALUE);
            throw error;
        }
    }

    static List<String> candidateNames(DType inputType, DType weightType, int inputColumnStart,
            int weightColumnStart, int columnLength) {
        boolean chunkSupported = (inputType == DType.F32 || inputType == DType.I8)
                && weightType == DType.Q4
                || inputType == DType.BF16 && weightType == DType.Q4;
        return CANDIDATES.stream()
                .filter(candidate -> chunkSupported || candidate.kernel() != GemmKernel.CHUNK)
                .map(Candidate::name)
                .toList();
    }

    private record Candidate(String name, GemmKernel kernel, Threading threading) {
    }
}
