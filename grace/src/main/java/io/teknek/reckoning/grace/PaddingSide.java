package io.teknek.reckoning.grace;

public enum PaddingSide {
    LEFT("left"),
    RIGHT("right");
    private String side;
    PaddingSide(String side){
        this.side = side;
    }
    public String getSide(){
        return side;
    }
}
