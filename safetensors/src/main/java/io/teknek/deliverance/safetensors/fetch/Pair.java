package io.teknek.deliverance.safetensors.fetch;

public class Pair <L,R>{

    public static<T,S> Pair<T,S> of(T t, S s){
        return new Pair<T,S> (t,s);
    }
    private final L left;
    private final R right;

    public Pair(L l, R r){
        left= l;
        right = r;
    }

    public L getLeft() {
        return left;
    }

    public R getRight() {
        return right;
    }
    public L left(){return left;}
    public R right(){return right;}
}
