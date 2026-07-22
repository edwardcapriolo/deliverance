package io.teknek.sketches.grammar;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class EbnfMatcherTest {
    @Test
    void matchesVllmStyleGrammar() {
        EbnfMatcher matcher = new EbnfMatcher("""
                root ::= select_statement
                select_statement ::= "SELECT " column " from " table " where " condition
                column ::= "col_1 " | "col_2 "
                table ::= "table_1 " | "table_2 "
                condition ::= column "= " number
                number ::= "1 " | "2 "
                """, "root");

        assertTrue(matcher.matches("SELECT col_1  from table_2  where col_2 = 1 "));
        assertFalse(matcher.matches("SELECT col_3  from table_2  where col_2 = 1 "));
    }

    @Test
    void boundsUnboundedRepeat() {
        EbnfMatcher matcher = new EbnfMatcher("root ::= " + quote("a") + "*", "root",
                new EbnfLimits(2, 4, 100));

        assertTrue(matcher.matches(""));
        assertTrue(matcher.matches("a"));
        assertTrue(matcher.matches("aa"));
        assertFalse(matcher.matches("aaa"));
    }

    @Test
    void boundsRecursion() {
        EbnfMatcher matcher = new EbnfMatcher("""
                root ::= item
                item ::= "x" | "(" item ")"
                """, "root", new EbnfLimits(4, 3, 1_000));

        assertTrue(matcher.matches("x"));
        assertTrue(matcher.matches("(x)"));
        assertTrue(matcher.matches("((x))"));
        assertFalse(matcher.matches("(((x)))"));
    }

    @Test
    void matchesSmallToonTabularSubset() {
        EbnfMatcher matcher = new EbnfMatcher("""
                root ::= users
                users ::= "users[" count "]{id,name,role}:\\n" rows
                count ::= "1" | "2"
                rows ::= row | row "\\n" row
                row ::= "  " id "," name "," role
                id ::= "1" | "2"
                name ::= "Ada" | "Bob"
                role ::= "admin" | "user"
                """, "root");

        assertTrue(matcher.matches("users[2]{id,name,role}:\n  1,Ada,admin\n  2,Bob,user"));
        assertTrue(matcher.matches("users[1]{id,name,role}:\n  1,Ada,admin"));
        assertFalse(matcher.matches("{\"users\":[{\"id\":1}]}"));
        assertFalse(matcher.matches("users[2]{id,name,role}:\n  1,Ada,admin\n  2,Eve,user"));
    }

    @Test
    void rejectsMissingStartRule() {
        assertThrows(IllegalArgumentException.class, () -> new EbnfMatcher("document ::= \"x\"", "root"));
    }

    private static String quote(String value) {
        return "\"" + value + "\"";
    }
}
