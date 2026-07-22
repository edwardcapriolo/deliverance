package io.teknek.sketches.grammar;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/** Parser for Deliverance EBNF v1. */
public final class EbnfParser {
    private final List<Token> tokens;
    private int position;

    private EbnfParser(String input) {
        this.tokens = new Lexer(input).lex();
    }

    public static EbnfGrammar parse(String input) {
        return new EbnfParser(input).parseGrammar();
    }

    private EbnfGrammar parseGrammar() {
        Map<String, EbnfNode> rules = new LinkedHashMap<>();
        skipSeparators();
        while (!peek(TokenType.EOF)) {
            String name = consume(TokenType.IDENTIFIER, "expected rule name").text();
            consume(TokenType.ASSIGN, "expected ::= after rule name");
            EbnfNode expression = parseExpression();
            if (rules.put(name, expression) != null) {
                throw error("duplicate EBNF rule: " + name);
            }
            if (peek(TokenType.EOF)) {
                break;
            }
            if (!peek(TokenType.NEWLINE) && !peek(TokenType.SEMICOLON)) {
                throw error("expected newline or ; after rule " + name);
            }
            skipSeparators();
        }
        if (rules.isEmpty()) {
            throw error("EBNF grammar must contain at least one rule");
        }
        return new EbnfGrammar(rules);
    }

    private EbnfNode parseExpression() {
        List<EbnfNode> options = new ArrayList<>();
        options.add(parseSequence());
        while (match(TokenType.PIPE)) {
            options.add(parseSequence());
        }
        return options.size() == 1 ? options.getFirst() : new EbnfNode.Alt(List.copyOf(options));
    }

    private EbnfNode parseSequence() {
        List<EbnfNode> parts = new ArrayList<>();
        while (!peek(TokenType.EOF) && !peek(TokenType.NEWLINE) && !peek(TokenType.SEMICOLON)
                && !peek(TokenType.RPAREN) && !peek(TokenType.PIPE)) {
            parts.add(parseTerm());
        }
        if (parts.isEmpty()) {
            throw error("expected expression term");
        }
        return parts.size() == 1 ? parts.getFirst() : new EbnfNode.Seq(List.copyOf(parts));
    }

    private EbnfNode parseTerm() {
        EbnfNode node;
        if (peek(TokenType.IDENTIFIER)) {
            node = new EbnfNode.Ref(consume(TokenType.IDENTIFIER, "expected rule reference").text());
        } else if (peek(TokenType.STRING)) {
            node = new EbnfNode.Literal(consume(TokenType.STRING, "expected literal").text());
        } else if (match(TokenType.LPAREN)) {
            node = parseExpression();
            consume(TokenType.RPAREN, "expected )");
        } else {
            throw error("expected identifier, literal, or group");
        }

        if (match(TokenType.QUESTION)) {
            return new EbnfNode.Repeat(node, 0, 1);
        }
        if (match(TokenType.STAR)) {
            return new EbnfNode.Repeat(node, 0, EbnfNode.Repeat.UNBOUNDED);
        }
        if (match(TokenType.PLUS)) {
            return new EbnfNode.Repeat(node, 1, EbnfNode.Repeat.UNBOUNDED);
        }
        return node;
    }

    private void skipSeparators() {
        while (match(TokenType.NEWLINE) || match(TokenType.SEMICOLON)) {
            // keep skipping
        }
    }

    private boolean match(TokenType type) {
        if (!peek(type)) {
            return false;
        }
        position++;
        return true;
    }

    private boolean peek(TokenType type) {
        return tokens.get(position).type() == type;
    }

    private Token consume(TokenType type, String message) {
        if (!peek(type)) {
            throw error(message);
        }
        return tokens.get(position++);
    }

    private IllegalArgumentException error(String message) {
        Token token = tokens.get(Math.min(position, tokens.size() - 1));
        return new IllegalArgumentException(message + " at line " + token.line() + ", column " + token.column());
    }

    private enum TokenType {
        IDENTIFIER, STRING, ASSIGN, PIPE, LPAREN, RPAREN, QUESTION, STAR, PLUS, NEWLINE, SEMICOLON, EOF
    }

    private record Token(TokenType type, String text, int line, int column) { }

    private static final class Lexer {
        private final String input;
        private int index;
        private int line = 1;
        private int column = 1;

        private Lexer(String input) {
            this.input = input == null ? "" : input;
        }

        private List<Token> lex() {
            List<Token> out = new ArrayList<>();
            while (!eof()) {
                char c = current();
                if (c == ' ' || c == '\t' || c == '\r') {
                    advance();
                } else if (c == '\n') {
                    out.add(token(TokenType.NEWLINE, ""));
                    advanceLine();
                } else if (c == ';') {
                    out.add(token(TokenType.SEMICOLON, ";"));
                    advance();
                } else if (c == ':' && peek(1) == ':' && peek(2) == '=') {
                    out.add(token(TokenType.ASSIGN, "::="));
                    advance();
                    advance();
                    advance();
                } else if (c == '|') {
                    out.add(token(TokenType.PIPE, "|"));
                    advance();
                } else if (c == '(') {
                    out.add(token(TokenType.LPAREN, "("));
                    advance();
                } else if (c == ')') {
                    out.add(token(TokenType.RPAREN, ")"));
                    advance();
                } else if (c == '?') {
                    out.add(token(TokenType.QUESTION, "?"));
                    advance();
                } else if (c == '*') {
                    out.add(token(TokenType.STAR, "*"));
                    advance();
                } else if (c == '+') {
                    out.add(token(TokenType.PLUS, "+"));
                    advance();
                } else if (c == '"') {
                    out.add(string());
                } else if (isIdentifierStart(c)) {
                    out.add(identifier());
                } else {
                    throw new IllegalArgumentException("unexpected EBNF character '" + c + "' at line " + line
                            + ", column " + column);
                }
            }
            out.add(new Token(TokenType.EOF, "", line, column));
            return out;
        }

        private Token identifier() {
            int startIndex = index;
            int startColumn = column;
            advance();
            while (!eof() && isIdentifierPart(current())) {
                advance();
            }
            return new Token(TokenType.IDENTIFIER, input.substring(startIndex, index), line, startColumn);
        }

        private Token string() {
            int startLine = line;
            int startColumn = column;
            advance();
            StringBuilder value = new StringBuilder();
            while (!eof()) {
                char c = current();
                if (c == '"') {
                    advance();
                    return new Token(TokenType.STRING, value.toString(), startLine, startColumn);
                }
                if (c == '\\') {
                    value.append(escape(startLine, startColumn));
                    continue;
                }
                if (c == '\n' || c == '\r') {
                    throw new IllegalArgumentException("unterminated EBNF string at line " + startLine
                            + ", column " + startColumn);
                }
                value.append(c);
                advance();
            }
            throw new IllegalArgumentException("unterminated EBNF string at line " + startLine + ", column "
                    + startColumn);
        }

        private char escape(int startLine, int startColumn) {
            advance();
            if (eof()) {
                throw new IllegalArgumentException("unterminated EBNF escape at line " + startLine + ", column "
                        + startColumn);
            }
            char c = current();
            return switch (c) {
                case '"' -> { advance(); yield '"'; }
                case '\\' -> { advance(); yield '\\'; }
                case 'n' -> { advance(); yield '\n'; }
                case 'r' -> { advance(); yield '\r'; }
                case 't' -> { advance(); yield '\t'; }
                case 'u' -> unicodeEscape(startLine, startColumn);
                default -> throw new IllegalArgumentException("unsupported EBNF escape \\" + c + " at line "
                        + line + ", column " + column);
            };
        }

        private char unicodeEscape(int startLine, int startColumn) {
            advance();
            if (index + 4 > input.length()) {
                throw new IllegalArgumentException("short EBNF unicode escape at line " + startLine + ", column "
                        + startColumn);
            }
            int value = 0;
            for (int i = 0; i < 4; i++) {
                int digit = Character.digit(current(), 16);
                if (digit < 0) {
                    throw new IllegalArgumentException("invalid EBNF unicode escape at line " + line + ", column "
                            + column);
                }
                value = (value << 4) + digit;
                advance();
            }
            return (char) value;
        }

        private Token token(TokenType type, String text) {
            return new Token(type, text, line, column);
        }

        private boolean eof() {
            return index >= input.length();
        }

        private char current() {
            return input.charAt(index);
        }

        private char peek(int ahead) {
            int next = index + ahead;
            return next >= input.length() ? 0 : input.charAt(next);
        }

        private void advance() {
            index++;
            column++;
        }

        private void advanceLine() {
            index++;
            line++;
            column = 1;
        }

        private static boolean isIdentifierStart(char c) {
            return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || c == '_';
        }

        private static boolean isIdentifierPart(char c) {
            return isIdentifierStart(c) || (c >= '0' && c <= '9');
        }
    }
}
