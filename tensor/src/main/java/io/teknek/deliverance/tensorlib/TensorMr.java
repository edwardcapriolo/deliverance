package io.teknek.deliverance.tensorlib;

import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.dysfx.Maybe;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ForkJoinTask;
import java.util.stream.LongStream;
/*
public class TensorMr {

    private WrappedForkJoinPool pool;
    public TensorMr(WrappedForkJoinPool pool) {
        this.pool = pool;
    }


    public<V,R> Maybe<R> exec(TensorMap<V> map, Reduce<V, R> reduce,
                              long offset, long length, int splitSize, AbstractTensor t) {
        long splits = Math.min(length, splitSize);
        long chunkSize = length / splits;
        long remainder = length % chunkSize;
        List<TensorSplit> tsplits = LongStream.range(0, splits).mapToObj(i -> new TensorSplit(offset + (i * chunkSize),
                remainder > 0 && i == splits - 1 ? chunkSize + remainder : chunkSize)).toList();
        List<ForkJoinTask<V>> tasks = new ArrayList<>();
        tsplits.forEach(split ->{
            ForkJoinTask<V> task = pool.getUnderlying().submit(() -> map.map(t, split.offset, split.length));
            tasks.add(task);
        });
        List<V> z = tasks.stream().map(ForkJoinTask::join).toList();
        return reduce.reduce(z);
    }
}*/
