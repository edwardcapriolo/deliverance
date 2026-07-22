package io.teknek.sketches.grammar;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertThrows;

class EbnfParserTest {
    @Test
    void parsesVllmStyleRules() {
        EbnfGrammar grammar = EbnfParser.parse("""
                root ::= select_statement
                select_statement ::= "SELECT " column " from " table
                column ::= "col_1" | "col_2"
                table ::= "table_1" | "table_2"
                """);

        assertEquals(4, grammar.rules().size());
        assertInstanceOf(EbnfNode.Ref.class, grammar.rule("root"));
        assertInstanceOf(EbnfNode.Seq.class, grammar.rule("select_statement"));
        assertInstanceOf(EbnfNode.Alt.class, grammar.rule("column"));
    }

    @Test
    void parsesGroupingAndSuffixOperators() {
        EbnfGrammar grammar = EbnfParser.parse("root ::= (" + quote("a") + " | " + quote("b") + ")+ "
                + quote("c") + "?");

        EbnfNode.Seq root = assertInstanceOf(EbnfNode.Seq.class, grammar.rule("root"));
        assertEquals(2, root.parts().size());
        EbnfNode.Repeat repeatedAlt = assertInstanceOf(EbnfNode.Repeat.class, root.parts().get(0));
        assertEquals(1, repeatedAlt.min());
        assertEquals(EbnfNode.Repeat.UNBOUNDED, repeatedAlt.max());
        EbnfNode.Repeat optional = assertInstanceOf(EbnfNode.Repeat.class, root.parts().get(1));
        assertEquals(0, optional.min());
        assertEquals(1, optional.max());
    }

    @Test
    void rejectsDuplicateRules() {
        assertThrows(IllegalArgumentException.class, () -> EbnfParser.parse("root ::= \"a\"\nroot ::= \"b\""));
    }

    private static String quote(String value) {
        return "\"" + value + "\"";
    }
}
