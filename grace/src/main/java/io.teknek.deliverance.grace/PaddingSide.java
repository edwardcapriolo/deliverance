package io.teknek.deliverance.grace;

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
