package io.teknek.deliverance.grace;

public record EncodeOptions(boolean addSpecialTokens, PaddingOptions padding, TruncationOptions truncation) {
    public static EncodeOptions defaults() {
        return new EncodeOptions(true, null, null);
    }

    public EncodeOptions withoutSpecialTokens() {
        return new EncodeOptions(false, padding, truncation);
    }

    public EncodeOptions withPadding(PaddingOptions value) {
        return new EncodeOptions(addSpecialTokens, value, truncation);
    }

    public EncodeOptions withTruncation(TruncationOptions value) {
        return new EncodeOptions(addSpecialTokens, padding, value);
    }
}
