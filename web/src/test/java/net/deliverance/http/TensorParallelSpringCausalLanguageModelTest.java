package net.deliverance.http;

import io.teknek.deliverance.generator.GeneratorParameters;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.DoNothingGenerateEvent;
import io.teknek.deliverance.model.tensorparallel.GossipParallelMembership;
import io.teknek.deliverance.model.tensorparallel.TensorParallelAssignment;
import io.teknek.deliverance.model.tensorparallel.TensorParallelGenerationGroup;
import io.teknek.deliverance.model.tensorparallel.TensorParallelRankAssignment;
import io.teknek.deliverance.model.tensorparallel.TensorParallelRankEndpoint;
import io.teknek.deliverance.safetensors.prompt.PromptContext;
import io.teknek.gossip.LocalMember;
import io.teknek.gossip.RemoteMember;
import org.junit.jupiter.api.Test;
import org.springframework.http.HttpStatus;
import org.springframework.web.server.ResponseStatusException;

import java.net.URI;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class TensorParallelSpringCausalLanguageModelTest {

    @Test
    void assignedNodeDownEventClosesGroupAndRejectsGeneration() {
        Fixture fixture = fixture(true, false);

        fixture.model().reconcileReadiness();

        verify(fixture.group()).close();
        ResponseStatusException thrown = assertThrows(ResponseStatusException.class,
                () -> fixture.model().generate(UUID.randomUUID(), PromptContext.of("hi"), new GeneratorParameters(),
                        new DoNothingGenerateEvent()));
        assertEquals(HttpStatus.SERVICE_UNAVAILABLE, thrown.getStatusCode());
        assertTrue(thrown.getReason().contains("gossip reports rank owner nodes down: [node-1]"));
    }

    @Test
    void unassignedNodeDownEventDoesNotCloseGroup() {
        Fixture fixture = fixture();

        fixture.model().reconcileReadiness();

        verify(fixture.group(), never()).close();
    }

    @Test
    void duplicateAssignedNodeDownEventsCloseGroupOnce() {
        Fixture fixture = fixture(true, false);

        fixture.model().reconcileReadiness();
        fixture.model().reconcileReadiness();

        verify(fixture.group()).close();
    }

    @Test
    void assignedNodeUpEventReopensGroupWhenAllRanksArePresent() {
        Fixture fixture = fixture(true, false);
        TensorParallelGenerationGroup reopened = mock(TensorParallelGenerationGroup.class);
        when(fixture.membership().openGenerationGroup()).thenReturn(reopened);

        fixture.model().reconcileReadiness();
        when(fixture.membership().liveMembers()).thenReturn(List.of(member("node-1")));
        fixture.model().reconcileReadiness();

        assertDoesNotThrow(() -> fixture.model().generate(UUID.randomUUID(), PromptContext.of("hi"),
                new GeneratorParameters(), new DoNothingGenerateEvent()));
        verify(fixture.group()).close();
        verify(fixture.membership()).openGenerationGroup();
        verify(reopened).generate(any(), any(), any(), any(), any());
    }

    @Test
    void assignedNodeUpEventDoesNotReopenGroupUntilAllRanksArePresent() {
        Fixture fixture = fixture(false, true);

        fixture.model().reconcileReadiness();

        ResponseStatusException thrown = assertThrows(ResponseStatusException.class,
                () -> fixture.model().generate(UUID.randomUUID(), PromptContext.of("hi"), new GeneratorParameters(),
                        new DoNothingGenerateEvent()));
        assertEquals(HttpStatus.SERVICE_UNAVAILABLE, thrown.getStatusCode());
        assertTrue(thrown.getReason().contains("Expected 2 rank endpoints but found 1"));
        verify(fixture.group()).close();
        verify(fixture.membership(), never()).openGenerationGroup();
    }

    @Test
    void reconcilerDoesNotCloseGroupUntilInFlightGenerationFinishes() throws Exception {
        Fixture fixture = fixture();
        CountDownLatch generationEntered = new CountDownLatch(1);
        CountDownLatch finishGeneration = new CountDownLatch(1);
        when(fixture.group().generate(any(), any(), any(), any(), any())).thenAnswer(invocation -> {
            generationEntered.countDown();
            assertTrue(finishGeneration.await(5, TimeUnit.SECONDS));
            return null;
        });
        ExecutorService executor = Executors.newFixedThreadPool(2);
        try {
            Future<?> generation = executor.submit(() -> fixture.model().generate(UUID.randomUUID(), PromptContext.of("hi"),
                    new GeneratorParameters(), new DoNothingGenerateEvent()));
            assertTrue(generationEntered.await(5, TimeUnit.SECONDS));

            when(fixture.membership().liveMembers()).thenReturn(List.of());
            fixture.model().reconcileReadiness();
            verify(fixture.group(), never()).close();

            finishGeneration.countDown();
            generation.get(5, TimeUnit.SECONDS);
            fixture.model().reconcileReadiness();
            verify(fixture.group()).close();
        } finally {
            executor.shutdownNow();
        }
    }

    private static Fixture fixture() {
        return fixture(true, true);
    }

    private static Fixture fixture(boolean allRankEndpointsPresent, boolean assignedNodeLive) {
        AbstractModel coordinator = mock(AbstractModel.class);
        TensorParallelGenerationGroup group = mock(TensorParallelGenerationGroup.class);
        GossipParallelMembership membership = mock(GossipParallelMembership.class);
        when(membership.findAssignment()).thenReturn(new TensorParallelAssignment("demo", "node-0", 2,
                "hash", List.of(new TensorParallelRankAssignment(0, "node-0"),
                new TensorParallelRankAssignment(1, "node-1"))));
        when(membership.findCollectiveUri()).thenReturn(URI.create("netty://127.0.0.1:42699"));
        when(membership.localNodeId()).thenReturn("node-0");
        when(membership.liveMembers()).thenReturn(assignedNodeLive ? List.of(member("node-1")) : List.of());
        List<TensorParallelRankEndpoint> endpoints = allRankEndpointsPresent
                ? List.of(new TensorParallelRankEndpoint(0, "node-0", "http://127.0.0.1:42600"),
                new TensorParallelRankEndpoint(1, "node-1", "http://127.0.0.1:42601"))
                : List.of(new TensorParallelRankEndpoint(0, "node-0", "http://127.0.0.1:42600"));
        when(membership.rankEndpointsForAssignment()).thenAnswer(invocation -> {
            if (endpoints.size() != 2) {
                throw new IllegalStateException("Expected 2 rank endpoints but found " + endpoints.size());
            }
            return endpoints;
        });
        TensorParallelSpringCausalLanguageModel model = new TensorParallelSpringCausalLanguageModel(coordinator, group,
                membership, false);
        return new Fixture(model, group, membership);
    }

    private static LocalMember member(String nodeId) {
        return new LocalMember("cluster", URI.create("udp://127.0.0.1:42605"), nodeId, System.nanoTime(),
                Map.of(), 100, 2_000, "exponential");
    }

    private record Fixture(TensorParallelSpringCausalLanguageModel model, TensorParallelGenerationGroup group,
            GossipParallelMembership membership) {
    }
}
