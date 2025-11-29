package io.teknek.deliverance.grace;

public class AddedToken {
    String content;
    boolean singleWord;
    boolean lstrip;
    boolean rstrip;
    boolean special;
    Boolean normalized;

    public AddedToken(String content, boolean singleWord, boolean lstrip, boolean rstrip, boolean special,
                      Boolean normalized) {
        this.content = content;
        this.singleWord = singleWord;
        this.lstrip = lstrip;
        this.rstrip = rstrip;
        this.special = special;
        this.normalized = normalized != null ? normalized : !special;
    }

}
