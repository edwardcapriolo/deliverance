package io.teknek.deliverance.grace;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;

public final class GraceTokenizerBenchmarkMain {
    private GraceTokenizerBenchmarkMain() {
    }

    public static void main(String[] args) throws Exception {
        Options options = Options.parse(args);
        String base = Files.readString(options.textFile);
        String text = base.repeat(options.repeat);
        PreTrainedTokenizer tokenizer = AutoTokenizer.fromPretrained(options.modelDir);

        int[] ids = tokenizer.encode(text).inputIds();
        int chars = text.codePointCount(0, text.length());
        int bytes = text.getBytes(StandardCharsets.UTF_8).length;
        System.out.println("grace.ids_sha256=" + sha256(ids));
        System.out.println("grace.tokens=" + ids.length);
        System.out.println("grace.chars=" + chars);
        System.out.println("grace.bytes=" + bytes);

        for (int i = 0; i < options.warmup; i++) {
            tokenizer.encode(text);
        }

        long start = System.nanoTime();
        long tokenCount = 0;
        for (int i = 0; i < options.iterations; i++) {
            tokenCount += tokenizer.encode(text).inputIds().length;
        }
        long elapsed = System.nanoTime() - start;
        printResult("grace", chars, tokenCount, options.iterations, elapsed);
    }

    private static void printResult(String engine, int charsPerIteration, long tokenCount, int iterations, long elapsedNanos) {
        double seconds = elapsedNanos / 1_000_000_000.0;
        double meanMs = (elapsedNanos / 1_000_000.0) / iterations;
        double charsPerSecond = (charsPerIteration * (double) iterations) / seconds;
        double tokensPerSecond = tokenCount / seconds;
        System.out.printf(java.util.Locale.ROOT,
                "%s iterations=%d mean_ms=%.3f chars_s=%.1f tokens_s=%.1f%n",
                engine, iterations, meanMs, charsPerSecond, tokensPerSecond);
    }

    private static String sha256(int[] ids) throws Exception {
        java.security.MessageDigest digest = java.security.MessageDigest.getInstance("SHA-256");
        for (int id : ids) {
            digest.update((byte) (id >>> 24));
            digest.update((byte) (id >>> 16));
            digest.update((byte) (id >>> 8));
            digest.update((byte) id);
        }
        byte[] bytes = digest.digest();
        StringBuilder out = new StringBuilder();
        for (byte b : bytes) {
            out.append(String.format("%02x", b));
        }
        return out.toString();
    }

    private record Options(Path modelDir, Path textFile, int repeat, int warmup, int iterations) {
        static Options parse(String[] args) {
            Path modelDir = null;
            Path textFile = Path.of("grace/src/test/resources/tokenizer-showdown.txt");
            int repeat = 64;
            int warmup = 20;
            int iterations = 200;
            for (int i = 0; i < args.length; i++) {
                switch (args[i]) {
                    case "--model-dir" -> modelDir = Path.of(args[++i]);
                    case "--text-file" -> textFile = Path.of(args[++i]);
                    case "--repeat" -> repeat = Integer.parseInt(args[++i]);
                    case "--warmup" -> warmup = Integer.parseInt(args[++i]);
                    case "--iterations" -> iterations = Integer.parseInt(args[++i]);
                    default -> throw new IllegalArgumentException("Unknown argument " + args[i]
                            + " in " + Arrays.toString(args));
                }
            }
            if (modelDir == null) {
                throw new IllegalArgumentException("--model-dir is required");
            }
            return new Options(modelDir, textFile, repeat, warmup, iterations);
        }
    }
}
