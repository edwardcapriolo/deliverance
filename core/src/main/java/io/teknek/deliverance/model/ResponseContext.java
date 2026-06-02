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
        String decoded = abstractModel.decodeToken(token);
        if (abstractModel.isSpecialToken(token)) {
            responseTextWithSpecialTokens.append(decoded);
        } else {
            event.emit(token, decoded, decoded, 0);
            responseText.append(decoded);
            responseTextWithSpecialTokens.append(decoded);
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
