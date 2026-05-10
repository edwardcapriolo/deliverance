package io.teknek.deliverance.model;

import java.util.ArrayList;
import java.util.List;

/**
 * This class is a mutable context that is modified during generate. here many of the things
 * are remnants of early codebase. EG we don't need responseTextWithSpecial tokens, in theaoy
 * we can calculate that on demand. We do it here because that is the way the code was.
 * At least with ResponseContext we don't ugly the main generate by tracking too much state
 */
public class ResponseContext {
    private final AbstractModel abstractModel;
    public final StringBuilder responseText = new StringBuilder();
    public final StringBuilder responseTextWithSpecialTokens = new StringBuilder();
    public final List<Integer> generatedTokens = new ArrayList<>();
    public final List<SamplerReturn> samplerReturnList = new ArrayList<>();

    public ResponseContext(AbstractModel abstractModel) {
        this.abstractModel = abstractModel;
    }

    public void add(SamplerReturn samplerReturn, GenerateEvent event) {
        samplerReturnList.add(samplerReturn);
        int token = samplerReturn.token;
        generatedTokens.add(token);
        //todo grace toknizer
        String decoded = abstractModel.tokenizer.decode(token);
        //todo do we need this anymore since grace should handle padding right?
        String cleaned = abstractModel.tokenRenderer.tokenizerToRendered(decoded);
        if (abstractModel.tokenizer.getModel().isSpecialToken(token)) {
            responseTextWithSpecialTokens.append(cleaned);
        } else {
            event.emit(token, decoded, cleaned, 0);
            responseText.append(cleaned);
            responseTextWithSpecialTokens.append(cleaned);
        }
    }

    public StringBuilder getResponseTextWithSpecialTokens() {
        return this.responseTextWithSpecialTokens;
    }

    public StringBuilder getResponseText() {
        return responseText;
    }

    public List<Integer> getGeneratedTokens() {
        return generatedTokens;
    }

    public List<SamplerReturn> getSamplerReturnList() {
        return samplerReturnList;
    }
}
