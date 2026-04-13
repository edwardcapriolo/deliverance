package io.teknek.deliverance.tensorlib;

import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.dysfx.Maybe;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ForkJoinTask;
import java.util.stream.LongStream;

public class TensorLib {

    public static List<TensorSplit> calculateTSplits(long offset, long length, int splitSize){
        long splits = Math.min(length, splitSize);
        long chunkSize = length / splits;
        long remainder = length % chunkSize;
        return LongStream.range(0, splits).mapToObj(i ->
                new TensorSplit(offset + (i * chunkSize),
                        remainder > 0 && i == splits - 1 ? chunkSize + remainder : chunkSize)).toList();
    }
    private final WrappedForkJoinPool pool;

    public TensorLib(WrappedForkJoinPool pool) {
        this.pool = pool;
    }

    public class Unary {
        private final AbstractTensor original;

        public Unary(AbstractTensor t) {
            original = t;
        }


        public class Mapper<V1> {
            TensorMap<V1> mapper;

            public Mapper(TensorMap<V1> mapper) {
                this.mapper = mapper;
            }

            public List<V1> execute(long offset, long length, int splitSize) {
                List<TensorSplit> tsplits = calculateTSplits(offset, length, splitSize);
                List<ForkJoinTask<V1>> tasks = new ArrayList<>();
                for (TensorSplit split : tsplits) {
                    ForkJoinTask<V1> task = pool.getUnderlying().submit(() -> mapper.map(original, split.offset, split.length));
                    tasks.add(task);
                }
                return tasks.stream().map(ForkJoinTask::join).toList();
            }

            public class Prepared{
                List<TensorSplit> tsplits;
                public Prepared(List<TensorSplit> tsplits ){
                    this.tsplits = tsplits;
                }

                public <R1> Maybe<R1> reduce(Reduce<V1,R1> reduce){
                    List<ForkJoinTask<V1>> tasks = new ArrayList<>();
                    for (TensorSplit split : tsplits) {
                        tasks.add(pool.getUnderlying().submit(() -> mapper.map(original, split.offset, split.length)));
                    }
                    List<V1> taskResults = tasks.stream().map(ForkJoinTask::join).toList();
                    if (taskResults.isEmpty()){
                        return Maybe.nothing();
                    }
                    return reduce.reduce(taskResults);
                }
            }
            public Prepared prepare(long offset, long length, int splitSize) {
                List<TensorSplit> tsplits = calculateTSplits(offset, length, splitSize);
                return new Prepared(tsplits);
            }
        }

        public <V> Mapper<V> mapper(TensorMap<V> mapper) {
            return new Mapper<>(mapper);
        }

        public class ReadOnlyMapper<V1> {
            ReadOnlyTensorMap<V1> mapper;
            ReadOnlyMapper(ReadOnlyTensorMap<V1> mapper){
                this.mapper = mapper;
            }

            /**
             * Creates a map-only job with one list value for each split
             * @param offset
             * @param length
             * @param splitSize
             * @return
             */
            public List<V1> execute(long offset, long length, int splitSize) {
                List<TensorSplit> tsplits = calculateTSplits(offset, length, splitSize);
                List<ForkJoinTask<V1>> tasks = new ArrayList<>();
                for (TensorSplit split : tsplits) {
                    ForkJoinTask<V1> task = pool.getUnderlying().submit(() -> mapper.map(original, split.offset, split.length));
                    tasks.add(task);
                }
                return tasks.stream().map(ForkJoinTask::join).toList();
            }

            public class Prepared{
                List<TensorSplit> tsplits;
                public Prepared(List<TensorSplit> tsplits ){
                    this.tsplits = tsplits;
                }

                public <R1> Maybe<R1> reduce(Reduce<V1,R1> reduce){
                    List<ForkJoinTask<V1>> tasks = new ArrayList<>();
                    for (TensorSplit split : tsplits) {
                        tasks.add(pool.getUnderlying().submit(() -> mapper.map(original, split.offset, split.length)));
                    }
                     List<V1> taskResults = tasks.stream().map(ForkJoinTask::join).toList();
                    return reduce.reduce(taskResults);
                }
            }
            public Prepared prepare(long offset, long length, int splitSize) {
                List<TensorSplit> tsplits = calculateTSplits(offset, length, splitSize);
                return new Prepared(tsplits);
            }
        }

        public <V> ReadOnlyMapper<V> readOnlyMapper(ReadOnlyTensorMap<V> mapper) {
            return new ReadOnlyMapper<>(mapper);
        }

    }

    public Unary unary(AbstractTensor t) {
        return new Unary(t);
    }
}
