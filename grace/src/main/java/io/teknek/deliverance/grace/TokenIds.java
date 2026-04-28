package io.teknek.deliverance.grace;

public class TokenIds {
    private final int input;
    private final int[] inputList;

    public TokenIds(int input) {
        this.input = input;
        this.inputList = null;
    }

    public TokenIds(int[] inputs) {
        this.input = -1;
        this.inputList = inputs.clone();
    }

    public boolean isScalar() {
        return inputList == null;
    }

    public int getInput() {
        return input;
    }

    public int[] getInputList() {
        return inputList == null ? null : inputList.clone();
    }

    public int[] asArray() {
        return isScalar() ? new int[]{input} : inputList.clone();
    }

    public int length() {
        return isScalar() ? 1 : inputList.length;
    }
}
