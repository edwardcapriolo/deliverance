package io.teknek.deliverance.model;

/**
 * Temporary no-op renderer.
 *
 * Historically this tried to paper over tokenizer-specific decode artifacts such as byte-level space
 * markers. Now that tokenizer behavior is moving toward `grace`, this extra rendering layer is more
 * likely to fight correct decode semantics than help them.
 */
public class TokenizerRenderer implements TokenRenderer {
    public String tokenizerToRendered(String token) {
        return token;
    }
}
