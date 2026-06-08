package io.teknek.deliverance.model.tensorparallel;

import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.GenerateEvent;
import io.teknek.deliverance.generator.GeneratorParameters;
import io.teknek.deliverance.generator.Response;
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
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

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
    }

    public static TensorParallelGenerationGroup fromEndpoints(List<RankEndpoint> endpoints) {
        return new TensorParallelGenerationGroup(endpoints, EndpointConstructor.INSTANCE);
    }

    private TensorParallelGenerationGroup(List<RankEndpoint> endpoints, EndpointConstructor ignored) {
        if (endpoints.isEmpty()) {
            throw new IllegalArgumentException("endpoints must not be empty");
        }
        this.endpoints = endpoints.stream().sorted(Comparator.comparingInt(RankEndpoint::rank)).toList();
        validateEndpointRanks(this.endpoints);
        this.executor = Executors.newFixedThreadPool(this.endpoints.size());
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
        try {
            return coordinatorModel.generateWithForwarder(sessionId, promptContext, generatorParameters, eventFired,
                    new AbstractModel.GenerationForwarder() {
                        @Override
                        public AbstractTensor batchForward(int[] tokenIds, int startPosition) {
                            return TensorParallelGenerationGroup.this.batchForward(sessionId, tokenIds, startPosition);
                        }

                        @Override
                        public AbstractTensor forward(int tokenId, int position) {
                            return TensorParallelGenerationGroup.this.forward(sessionId, tokenId, position);
                        }
                    });
        } finally {
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
        try {
            List<Future<AbstractTensor>> futures = new ArrayList<>();
            for (int i = 0; i < endpoints.size(); i++) {
                int index = i;
                futures.add(executor.submit(() -> forward.apply(index, endpoints.get(index).service())));
            }
            List<AbstractTensor> outputs = new ArrayList<>();
            for (Future<AbstractTensor> future : futures) {
                outputs.add(future.get());
            }
            return outputs;
        } catch (Exception e) {
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

    private void closeSession(UUID sessionId) {
        List<Future<?>> futures = new ArrayList<>();
        for (RankEndpoint endpoint : endpoints) {
            futures.add(executor.submit(() -> endpoint.service().closeSession(sessionId)));
        }
        try {
            for (Future<?> future : futures) {
                future.get();
            }
        } catch (Exception e) {
            throw new RuntimeException("Tensor-parallel session cleanup failed", e);
        }
    }

    private enum EndpointConstructor {
        INSTANCE
    }
}
