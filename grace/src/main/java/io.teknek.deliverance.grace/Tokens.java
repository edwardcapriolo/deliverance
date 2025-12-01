package io.teknek.deliverance.grace;

public class Tokens {
    private  final String input;
    private  final String [] inputs;
    public Tokens(String input){
        this.input = input;
        this.inputs = null;
    }
    public Tokens(String [] inputs){
        this.inputs = inputs;
        this.input = null;

    }
    public boolean isScalar(){
        return inputs == null;
    }

    public String getInput() {
        return input;
    }

    public String[] getInputs() {
        return inputs;
    }
}
