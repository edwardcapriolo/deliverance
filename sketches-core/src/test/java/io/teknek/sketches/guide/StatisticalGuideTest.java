package io.teknek.sketches.guide;

import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;

class StatisticalGuideTest {

    @Test
    void generatedLengthsMatchExpectedDistributionMoments() {
        double[] p0 = {0.2, 0.6, 0.2};
        int[] states = {1, 2, 3};
        int samples = 250;
        String regex = "11[01]+|0[01]*";

        NextToken markov = new NextToken(this::probMarkov, p0, states, 30127);
        NextToken nonMarkov = new NextToken(this::probNonMarkov, p0, states, 24601);

        double[] lengths1 = new double[samples];
        double[] lengths2 = new double[samples];
        for (int i = 0; i < samples; i++) {
            lengths1[i] = generate(markov, regex).size() - 1;
            lengths2[i] = generate(nonMarkov, regex).size() - 1;
        }

        // These are intentionally loose Java-port equivalents of outlines-core's seeded statistical checks. The exact
        // values differ because java.util.Random and numpy's generator do not produce the same sample stream.
        assertEquals(4.833333, mean(lengths1), 0.8);
        assertEquals(4.833333, mean(lengths2), 0.8);
        assertEquals(8.527777, variance(lengths1), 3.5);
        assertEquals(8.527777, variance(lengths2), 3.5);
    }

    private List<Integer> generate(NextToken model, String regex) {
        Vocabulary vocabulary = new Vocabulary(3, Map.of("0", List.of(1), "1", List.of(2)));
        IndexGuide guide = new IndexGuide(new Index(regex, vocabulary));

        List<Integer> tokens = new ArrayList<>();
        List<Integer> allowed = guide.getTokens();
        while (true) {
            tokens = model.next(tokens, allowed);
            int token = tokens.getLast();
            if (token == 3) {
                break;
            }
            allowed = guide.advance(token);
        }
        return tokens;
    }

    private double[] probNonMarkov(List<Integer> tokens) {
        long n0 = tokens.stream().filter(token -> token == 1).count();
        long n1 = tokens.size() - n0;
        double[] p = {
                1.0 + Math.exp(n1 - n0),
                1.0 + Math.exp(n0 - n1),
                0.0
        };
        normalize(p);
        return new double[] {
                0.7 * p[0],
                0.7 * p[1],
                0.7 * p[2] + 0.3
        };
    }

    private double[] probMarkov(List<Integer> tokens) {
        return switch (tokens.getLast()) {
            case 1 -> new double[] {0.2, 0.5, 0.3};
            case 2 -> new double[] {0.3, 0.4, 0.3};
            default -> throw new IllegalArgumentException("Unexpected token " + tokens.getLast());
        };
    }

    private static void normalize(double[] values) {
        double sum = 0.0;
        for (double value : values) {
            sum += value;
        }
        for (int i = 0; i < values.length; i++) {
            values[i] /= sum;
        }
    }

    private static double mean(double[] values) {
        double sum = 0.0;
        for (double value : values) {
            sum += value;
        }
        return sum / values.length;
    }

    private static double variance(double[] values) {
        double mean = mean(values);
        double sum = 0.0;
        for (double value : values) {
            double diff = value - mean;
            sum += diff * diff;
        }
        return sum / values.length;
    }

    private interface ProbabilityFunction {
        double[] probabilities(List<Integer> tokens);
    }

    private static final class NextToken {
        private final ProbabilityFunction probabilityFunction;
        private final double[] p0;
        private final int[] states;
        private final Random random;

        private NextToken(ProbabilityFunction probabilityFunction, double[] p0, int[] states, long seed) {
            this.probabilityFunction = probabilityFunction;
            this.p0 = p0;
            this.states = states;
            this.random = new Random(seed);
        }

        private List<Integer> next(List<Integer> tokens, List<Integer> mask) {
            double[] probabilities = tokens.isEmpty() ? p0.clone() : probabilityFunction.probabilities(tokens);
            double sum = 0.0;
            for (int i = 0; i < states.length; i++) {
                if (!mask.contains(states[i])) {
                    probabilities[i] = 0.0;
                }
                sum += probabilities[i];
            }
            if (sum == 0.0) {
                throw new IllegalStateException("No probability mass remains after applying guide mask");
            }

            double pick = random.nextDouble() * sum;
            int selected = states[states.length - 1];
            for (int i = 0; i < states.length; i++) {
                pick -= probabilities[i];
                if (pick <= 0.0) {
                    selected = states[i];
                    break;
                }
            }
            ArrayList<Integer> nextTokens = new ArrayList<>(tokens.size() + 1);
            nextTokens.addAll(tokens);
            nextTokens.add(selected);
            return nextTokens;
        }
    }
}
