package io.teknek.deliverance.grace;

import java.util.List;

public class TokenIds {
    private final int input;
    private final int [] inputList;

    public TokenIds(int input){
        this.input = input;
        inputList = null;
    }
    public TokenIds(int [] inputs){
        this.input = -1;
        this.inputList = inputs;
    }
    public boolean isScalar(){
        return inputList == null;
    }

    public int getInput() {
        return input;
    }

    public int[] getInputList() {
        return inputList;
    }
}
