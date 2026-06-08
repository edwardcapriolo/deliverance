package io.teknek.deliverance.model.tensorparallel;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class TensorParallelAssignmentTest {

    @Test
    public void ranksForNodeReturnsAllAssignedRanksForPhysicalNode() {
        TensorParallelAssignment assignment = assignment();

        assertEquals(List.of(0, 1), assignment.ranksForNode("node-0"));
        assertEquals(List.of(2, 3), assignment.ranksForNode("node-1"));
        assertEquals(List.of(), assignment.ranksForNode("node-2"));
    }

    @Test
    public void validatesEveryRankAndTopologyMatch() {
        TensorParallelAssignment assignment = assignment();
        TensorParallelTopology topology = topology();

        assertTrue(assignment.assignsEveryRank());
        assertTrue(assignment.matchesTopology(topology));
        assertFalse(assignment.matchesTopology(new TensorParallelTopology("demo", 4,
                List.of("node-0", "node-1", "node-1", "node-0"), List.of(), "other")));
    }

    @Test
    public void rejectsNonContiguousRankAssignments() {
        assertThrows(IllegalArgumentException.class, () -> new TensorParallelAssignment("demo", "node-0", 2,
                "hash", List.of(new TensorParallelRankAssignment(0, "node-0"),
                new TensorParallelRankAssignment(2, "node-1"))));
    }

    private static TensorParallelAssignment assignment() {
        return new TensorParallelAssignment("demo", "node-0", 4, topology().assignmentHash(), topology().rankAssignments());
    }

    private static TensorParallelTopology topology() {
        List<String> active = List.of("node-0", "node-0", "node-1", "node-1");
        return new TensorParallelTopology("demo", 4, active, List.of(),
                TensorParallelTopology.assignmentHash("demo", 4, active));
    }
}
