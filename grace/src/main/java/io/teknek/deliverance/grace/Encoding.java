package io.teknek.deliverance.grace;

public record Encoding(int[] inputIds, int[] attentionMask, int[] specialTokensMask) {
    public Encoding {
        inputIds = inputIds.clone();
        attentionMask = attentionMask.clone();
        specialTokensMask = specialTokensMask.clone();
    }

    public int length() {
        return inputIds.length;
    }

    public TokenIds tokenIds() {
        return new TokenIds(inputIds);
    }
}
