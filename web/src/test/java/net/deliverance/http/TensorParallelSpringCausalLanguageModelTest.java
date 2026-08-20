package net.deliverance.http;

import io.teknek.deliverance.generator.GeneratorParameters;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.DoNothingGenerateEvent;
import io.teknek.deliverance.model.tensorparallel.GossipParallelMembership;
import io.teknek.deliverance.model.tensorparallel.TensorParallelAssignment;
import io.teknek.deliverance.model.tensorparallel.TensorParallelGenerationGroup;
import io.teknek.deliverance.model.tensorparallel.TensorParallelRankAssignment;
import io.teknek.deliverance.safetensors.prompt.PromptContext;
import io.teknek.gossip.RemoteMember;
import io.teknek.gossip.event.GossipListener;
import io.teknek.gossip.event.GossipState;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;
import org.springframework.http.HttpStatus;
import org.springframework.web.server.ResponseStatusException;

import java.net.URI;
import java.util.List;
import java.util.UUID;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class TensorParallelSpringCausalLanguageModelTest {

    @Test
    void assignedNodeDownEventClosesGroupAndRejectsGeneration() {
        Fixture fixture = fixture();

        fixture.listener().gossipEvent(new RemoteMember("cluster", URI.create("udp://127.0.0.1:42605"),
                "node-1"), GossipState.DOWN);

        verify(fixture.group()).close();
        ResponseStatusException thrown = assertThrows(ResponseStatusException.class,
                () -> fixture.model().generate(UUID.randomUUID(), PromptContext.of("hi"), new GeneratorParameters(),
                        new DoNothingGenerateEvent()));
        assertEquals(HttpStatus.SERVICE_UNAVAILABLE, thrown.getStatusCode());
        assertTrue(thrown.getReason().contains("gossip reports assigned rank owner down: node-1"));
    }

    @Test
    void unassignedNodeDownEventDoesNotCloseGroup() {
        Fixture fixture = fixture();

        fixture.listener().gossipEvent(new RemoteMember("cluster", URI.create("udp://127.0.0.1:42607"),
                "node-9"), GossipState.DOWN);

        verify(fixture.group(), never()).close();
    }

    @Test
    void duplicateAssignedNodeDownEventsCloseGroupOnce() {
        Fixture fixture = fixture();
        RemoteMember node = new RemoteMember("cluster", URI.create("udp://127.0.0.1:42605"), "node-1");

        fixture.listener().gossipEvent(node, GossipState.DOWN);
        fixture.listener().gossipEvent(node, GossipState.DOWN);

        verify(fixture.group()).close();
    }

    private static Fixture fixture() {
        AbstractModel coordinator = mock(AbstractModel.class);
        TensorParallelGenerationGroup group = mock(TensorParallelGenerationGroup.class);
        GossipParallelMembership membership = mock(GossipParallelMembership.class);
        when(membership.findAssignment()).thenReturn(new TensorParallelAssignment("demo", "node-0", 2,
                "hash", List.of(new TensorParallelRankAssignment(0, "node-0"),
                new TensorParallelRankAssignment(1, "node-1"))));
        TensorParallelSpringCausalLanguageModel model = new TensorParallelSpringCausalLanguageModel(coordinator, group,
                membership);
        ArgumentCaptor<GossipListener> listener = ArgumentCaptor.forClass(GossipListener.class);
        verify(membership).registerGossipListener(listener.capture());
        return new Fixture(model, group, listener.getValue());
    }

    private record Fixture(TensorParallelSpringCausalLanguageModel model, TensorParallelGenerationGroup group,
            GossipListener listener) {
    }
}
