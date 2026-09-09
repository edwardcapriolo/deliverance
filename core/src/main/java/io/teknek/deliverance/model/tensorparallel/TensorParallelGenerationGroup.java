package io.teknek.deliverance.model.tensorparallel;

import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.GenerateEvent;
import io.teknek.deliverance.model.GenerationBackend;
import io.teknek.deliverance.model.GenerationCursor;
import io.teknek.deliverance.generator.GeneratorParameters;
import io.teknek.deliverance.generator.Response;
import io.teknek.deliverance.model.tensorparallel.transport.SharedKvPrefixRestoreRequest;
import io.teknek.deliverance.model.tensorparallel.transport.SharedKvPrefixRestoreResult;
import io.teknek.deliverance.model.tensorparallel.transport.SharedKvPrefixStoreRequest;
import io.teknek.deliverance.model.tensorparallel.transport.TensorParallelRankService;
import io.teknek.deliverance.safetensors.prompt.PromptContext;
import io.teknek.deliverance.tensor.AbstractTensor;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;
import java.util.UUID;
import java.time.Duration;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

/**
 * Coordinates tensor-parallel rank endpoints for generation building blocks and full generation.
 *
 * <p>This class owns a set of rank services that share a tensor-parallel assignment. It does not perform membership
 * discovery or leader election. Those concerns are handled before endpoints are supplied. This class only coordinates
 * rank execution.</p>
 */
public class TensorParallelGenerationGroup implements AutoCloseable {
    private final UUID sessionId = UUID.randomUUID();
    private final List<RankEndpoint> endpoints;
    private final ExecutorService executor;
    private final TensorParallelTimeoutSettings timeoutSettings;

    public TensorParallelGenerationGroup(List<AbstractModel> models) {
        if (models.isEmpty()) {
            throw new IllegalArgumentException("models must not be empty");
        }
        List<AbstractModel> sortedModels = models.stream()
                .sorted(Comparator.comparingInt(model -> model.getTensorParallelContext().rank()))
                .toList();
        validateModelRanks(sortedModels);
        this.endpoints = sortedModels.stream()
                .map(model -> new RankEndpoint(model.getTensorParallelContext().rank(), model.getTensorParallelContext().size(),
                        new InProcessTensorParallelRankService(model), true))
                .toList();
        this.executor = Executors.newFixedThreadPool(this.endpoints.size());
        this.timeoutSettings = TensorParallelTimeoutSettings.DEFAULT;
    }

    public static TensorParallelGenerationGroup fromEndpoints(List<RankEndpoint> endpoints) {
        return fromEndpoints(endpoints, TensorParallelTimeoutSettings.DEFAULT);
    }

    public static TensorParallelGenerationGroup fromEndpoints(List<RankEndpoint> endpoints,
            TensorParallelTimeoutSettings timeoutSettings) {
        return new TensorParallelGenerationGroup(endpoints, EndpointConstructor.INSTANCE, timeoutSettings);
    }

    private TensorParallelGenerationGroup(List<RankEndpoint> endpoints, EndpointConstructor ignored,
            TensorParallelTimeoutSettings timeoutSettings) {
        if (endpoints.isEmpty()) {
            throw new IllegalArgumentException("endpoints must not be empty");
        }
        this.endpoints = endpoints.stream().sorted(Comparator.comparingInt(RankEndpoint::rank)).toList();
        validateEndpointRanks(this.endpoints);
        this.executor = Executors.newFixedThreadPool(this.endpoints.size());
        this.timeoutSettings = timeoutSettings == null ? TensorParallelTimeoutSettings.DEFAULT : timeoutSettings;
    }

    /**
     * Runs prompt/prefix forward on every rank and returns rank 0's reduced output.
     *
     * <p>All rank models must enter the same collective calls in the same order. Non-zero rank outputs are closed before
     * this method returns. The caller owns the returned rank 0 output tensor.</p>
     */
    public AbstractTensor batchForward(int[] tokenIds, int startPosition) {
        List<AbstractTensor> outputs = forwardAllRanks((index, model) ->
                model.batchForward(sessionId, tokenIds, startPosition));
        AbstractTensor rankZero = outputs.get(0);
        for (int i = 1; i < outputs.size(); i++) {
            outputs.get(i).close();
        }
        return rankZero;
    }

    /**
     * Runs prompt/prefix forward on every rank and returns every rank's reduced output.
     *
     * <p>The caller owns every returned tensor. This method is useful for tests that need to verify rank outputs agree.</p>
     */
    public List<AbstractTensor> batchForwardAllRanks(int[] tokenIds, int startPosition) {
        return forwardAllRanks((index, model) -> model.batchForward(sessionId, tokenIds, startPosition));
    }

    /**
     * Runs one decode-token forward step on every rank and returns rank 0's reduced output.
     */
    public AbstractTensor forward(int tokenId, int position) {
        List<AbstractTensor> outputs = forwardAllRanks((index, model) ->
                model.forward(sessionId, tokenId, position));
        AbstractTensor rankZero = outputs.get(0);
        for (int i = 1; i < outputs.size(); i++) {
            outputs.get(i).close();
        }
        return rankZero;
    }

    /**
     * Runs one decode-token forward step on every rank and returns every rank's reduced output.
     */
    public List<AbstractTensor> forwardAllRanks(int tokenId, int position) {
        return forwardAllRanks((index, model) -> model.forward(sessionId, tokenId, position));
    }

    /**
     * Generates text through the tensor-parallel rank endpoints.
     *
     * <p>The supplied coordinator model owns tokenizer, output projection, sampling, stop handling, and response
     * post-processing. This group owns distributed prompt/decode forward execution and rank-local KV state.</p>
     */
    public Response generate(AbstractModel coordinatorModel, PromptContext promptContext,
                             GeneratorParameters generatorParameters, GenerateEvent eventFired) {
        return generate(UUID.randomUUID(), coordinatorModel, promptContext, generatorParameters, eventFired);
    }

    public Response generate(UUID sessionId, AbstractModel coordinatorModel, PromptContext promptContext,
                              GeneratorParameters generatorParameters, GenerateEvent eventFired) {
        Objects.requireNonNull(coordinatorModel, "coordinatorModel");
        return coordinatorModel.generateWithBackend(sessionId, promptContext, generatorParameters, eventFired,
                new TensorParallelGenerationBackend(promptContext));
    }

    private final class TensorParallelGenerationBackend implements GenerationBackend {
        private TensorParallelGenerationBackend(PromptContext ignoredPromptContext) {
        }

        @Override
        public GenerationSession open(UUID sessionId, int[] promptTokens, GeneratorParameters parameters) {
            String cacheSalt = parameters.cacheSalt.orElse("");
            int prefixLength = restoreSharedKvPrefix(sessionId, promptTokens, cacheSalt);
            return new TensorParallelGenerationSession(sessionId, promptTokens, cacheSalt, prefixLength);
        }
    }

    private final class TensorParallelGenerationSession implements GenerationBackend.GenerationSession {
        private final UUID sessionId;
        private final int[] promptTokens;
        private final String cacheSalt;
        private final int prefixLength;

        private TensorParallelGenerationSession(UUID sessionId, int[] promptTokens, String cacheSalt, int prefixLength) {
            this.sessionId = sessionId;
            this.promptTokens = promptTokens;
            this.cacheSalt = cacheSalt;
            this.prefixLength = prefixLength;
        }

        @Override
        public int prefixLength() {
            return prefixLength;
        }

        @Override
        public AbstractTensor prefill(GenerationCursor cursor) {
            AbstractTensor output;
            if (cursor.hasTokensToProcess()) {
                output = TensorParallelGenerationGroup.this.batchForward(sessionId, cursor.tokensToProcess(), cursor.startPosition());
                storeSharedKvPrefix(sessionId, promptTokens, cacheSalt);
            } else {
                output = TensorParallelGenerationGroup.this.forward(sessionId, cursor.replayToken(), cursor.replayPosition());
            }
            return output;
        }

        @Override
        public AbstractTensor decode(int tokenId, int position) {
            return TensorParallelGenerationGroup.this.forward(sessionId, tokenId, position);
        }

        @Override
        public void close() {
            closeSession(sessionId);
        }
    }

    @Override
    public void close() {
        executor.shutdownNow();
        for (RankEndpoint endpoint : endpoints) {
            endpoint.closeIfOwned();
        }
    }

    private List<AbstractTensor> forwardAllRanks(RankForward forward) {
        List<Future<AbstractTensor>> futures = new ArrayList<>();
        List<AbstractTensor> outputs = new ArrayList<>();
        try {
            for (int i = 0; i < endpoints.size(); i++) {
                int index = i;
                futures.add(executor.submit(() -> forward.apply(index, endpoints.get(index).service())));
            }
            for (int i = 0; i < futures.size(); i++) {
                outputs.add(await("forward", endpoints.get(i), futures.get(i), timeoutSettings.rankOperationTimeout()));
            }
            return outputs;
        } catch (Exception e) {
            cancelAll(futures);
            for (AbstractTensor output : outputs) {
                output.close();
            }
            throw new RuntimeException("Tensor-parallel forward failed", e);
        }
    }

    private static void validateModelRanks(List<AbstractModel> models) {
        int size = models.get(0).getTensorParallelContext().size();
        Set<Integer> ranks = new HashSet<>();
        for (AbstractModel model : models) {
            if (model.getTensorParallelContext().size() != size) {
                throw new IllegalArgumentException("all rank models must have the same tensor-parallel size");
            }
            ranks.add(model.getTensorParallelContext().rank());
        }
        if (ranks.size() != size) {
            throw new IllegalArgumentException("rank model count must equal tensor-parallel size");
        }
        for (int rank = 0; rank < size; rank++) {
            if (!ranks.contains(rank)) {
                throw new IllegalArgumentException("missing tensor-parallel rank " + rank);
            }
        }
    }

    private static void validateEndpointRanks(List<RankEndpoint> endpoints) {
        int size = endpoints.get(0).size();
        Set<Integer> ranks = new HashSet<>();
        for (RankEndpoint endpoint : endpoints) {
            if (endpoint.size() != size) {
                throw new IllegalArgumentException("all rank endpoints must have the same tensor-parallel size");
            }
            ranks.add(endpoint.rank());
        }
        if (ranks.size() != size) {
            throw new IllegalArgumentException("rank endpoint count must equal tensor-parallel size");
        }
        for (int rank = 0; rank < size; rank++) {
            if (!ranks.contains(rank)) {
                throw new IllegalArgumentException("missing tensor-parallel rank " + rank);
            }
        }
    }

    public record RankEndpoint(int rank, int size, TensorParallelRankService service, boolean closeWithGroup) {
        private void closeIfOwned() {
            if (closeWithGroup && service instanceof AutoCloseable closeable) {
                try {
                    closeable.close();
                } catch (Exception e) {
                    throw new RuntimeException(e);
                }
            }
        }
    }

    private interface RankForward {
        AbstractTensor apply(int index, TensorParallelRankService model);
    }

    private AbstractTensor batchForward(UUID sessionId, int[] tokenIds, int startPosition) {
        List<AbstractTensor> outputs = forwardAllRanks((index, model) ->
                model.batchForward(sessionId, tokenIds, startPosition));
        AbstractTensor rankZero = outputs.get(0);
        for (int i = 1; i < outputs.size(); i++) {
            outputs.get(i).close();
        }
        return rankZero;
    }

    private AbstractTensor forward(UUID sessionId, int tokenId, int position) {
        List<AbstractTensor> outputs = forwardAllRanks((index, model) ->
                model.forward(sessionId, tokenId, position));
        AbstractTensor rankZero = outputs.get(0);
        for (int i = 1; i < outputs.size(); i++) {
            outputs.get(i).close();
        }
        return rankZero;
    }

    int restoreSharedKvPrefix(UUID sessionId, int[] promptTokens, String cacheSalt) {
        List<Future<SharedKvPrefixRestoreResult>> futures = new ArrayList<>();
        try {
            for (RankEndpoint endpoint : endpoints) {
                futures.add(executor.submit(() -> endpoint.service()
                        .restoreSharedKvPrefix(new SharedKvPrefixRestoreRequest(sessionId, promptTokens, cacheSalt))));
            }
            Integer agreedLength = null;
            boolean mismatch = false;
            for (int i = 0; i < futures.size(); i++) {
                SharedKvPrefixRestoreResult result = await("restoreSharedKvPrefix", endpoints.get(i), futures.get(i),
                        timeoutSettings.rankOperationTimeout());
                if (result.prefixLength() <= 0) {
                    mismatch = true;
                    continue;
                }
                if (agreedLength == null) {
                    agreedLength = result.prefixLength();
                } else if (agreedLength != result.prefixLength()) {
                    mismatch = true;
                }
            }
            if (mismatch) {
                closeSession(sessionId);
                return 0;
            }
            return agreedLength == null ? 0 : agreedLength;
        } catch (Exception e) {
            cancelAll(futures);
            closeSession(sessionId);
            throw new RuntimeException("Tensor-parallel shared KV prefix restore failed", e);
        }
    }

    void storeSharedKvPrefix(UUID sessionId, int[] promptTokens, String cacheSalt) {
        List<Future<?>> futures = new ArrayList<>();
        try {
            for (RankEndpoint endpoint : endpoints) {
                futures.add(executor.submit(() -> endpoint.service()
                        .storeSharedKvPrefix(new SharedKvPrefixStoreRequest(sessionId, promptTokens, cacheSalt))));
            }
            for (int i = 0; i < futures.size(); i++) {
                await("storeSharedKvPrefix", endpoints.get(i), futures.get(i), timeoutSettings.rankOperationTimeout());
            }
        } catch (Exception e) {
            cancelAll(futures);
            throw new RuntimeException("Tensor-parallel shared KV prefix store failed", e);
        }
    }

    private void closeSession(UUID sessionId) {
        List<Future<?>> futures = new ArrayList<>();
        for (RankEndpoint endpoint : endpoints) {
            futures.add(executor.submit(() -> endpoint.service().closeSession(sessionId)));
        }
        try {
            for (int i = 0; i < futures.size(); i++) {
                await("closeSession", endpoints.get(i), futures.get(i), timeoutSettings.rankCloseTimeout());
            }
        } catch (Exception e) {
            cancelAll(futures);
            throw new RuntimeException("Tensor-parallel session cleanup failed", e);
        }
    }

    private static void cancelAll(List<? extends Future<?>> futures) {
        for (Future<?> future : futures) {
            future.cancel(true);
        }
    }

    private <T> T await(String operation, RankEndpoint endpoint, Future<T> future, Duration timeout) throws Exception {
        try {
            return future.get(timeout.toNanos(), TimeUnit.NANOSECONDS);
        } catch (TimeoutException e) {
            future.cancel(true);
            throw new TimeoutException("Timed out waiting for tensor-parallel rank operation=" + operation
                    + " rank=" + endpoint.rank() + " size=" + endpoint.size()
                    + " timeout=" + timeout);
        } catch (Exception e) {
            future.cancel(true);
            throw new RuntimeException("Tensor-parallel rank operation failed operation=" + operation
                    + " rank=" + endpoint.rank() + " size=" + endpoint.size(), e);
        }
    }

    private enum EndpointConstructor {
        INSTANCE
    }
}
