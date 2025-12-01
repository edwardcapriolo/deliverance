package io.teknek.deliverance.grace;

public enum TruncationSide {
    LEFT("left"),
    RIGHT("right");
    private String side;
    TruncationSide(String side){
        this.side = side;
    }
    public String getSide(){
        return side;
    }
}
