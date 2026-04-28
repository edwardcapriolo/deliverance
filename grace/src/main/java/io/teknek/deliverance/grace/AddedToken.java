package io.teknek.deliverance.grace;

public record AddedToken(
        String content,
        boolean singleWord,
        boolean lstrip,
        boolean rstrip,
        boolean special,
        boolean normalized) {

    public AddedToken(String content, boolean singleWord, boolean lstrip, boolean rstrip, boolean special,
                      Boolean normalized) {
        this(content, singleWord, lstrip, rstrip, special, normalized != null ? normalized : !special);
    }
}
