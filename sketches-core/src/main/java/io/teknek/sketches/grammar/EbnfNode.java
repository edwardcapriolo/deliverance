package io.teknek.sketches.grammar;

import java.util.List;

public sealed interface EbnfNode permits EbnfNode.Alt, EbnfNode.Literal, EbnfNode.Ref, EbnfNode.Repeat, EbnfNode.Seq {
    record Literal(String value) implements EbnfNode { }

    record Ref(String name) implements EbnfNode { }

    record Seq(List<EbnfNode> parts) implements EbnfNode { }

    record Alt(List<EbnfNode> options) implements EbnfNode { }

    record Repeat(EbnfNode node, int min, int max) implements EbnfNode {
        public static final int UNBOUNDED = -1;
    }
}
