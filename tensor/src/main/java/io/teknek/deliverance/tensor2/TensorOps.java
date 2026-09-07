package io.teknek.deliverance.tensor2;

import com.google.common.base.Preconditions;
import io.dropwizard.metrics5.MetricName;
import io.dropwizard.metrics5.MetricRegistry;
import io.dropwizard.metrics5.Timer;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.tensor.TensorShape;
import io.teknek.dysfx.Either;
import io.teknek.dysfx.exception.UnreachableException;

import java.lang.foreign.MemorySegment;
import java.nio.FloatBuffer;
import java.util.Map;
import java.util.concurrent.TimeUnit;

class Lighter {
    private Allocator allocator = new Allocator();
    //something like this must be initialized  so later we can pick the right one
    //    private final EnumMap<TensorProviderKind, TensorOperations> tensorOperations = new EnumMap<>(TensorProviderKind.class);
    public Lighter(){

    }
    public void multiplyAccumulate(MultiplyAccumulate multiplyAccumulate){
        //chack that a and b are same device
        //check that a and b are same size
        //this.multiple(afaf, Map.of());
    }

    public void multiplyAccumulate(MultiplyAccumulate multiplyAccumulate, Map<String, String> tags){
        //chack that a and b are same device
        //check that a and b are same size

    }

    /*
    //this would be like if we wanted to pass hints along which tensorProviderKind to chose etc
    public void multiplyAccumulate(MultiplyAccumulate multiplyAccumulate, Map<String, String> tags, ExecutionHints hints){
        //chack that a and b are same device
        //check that a and b are same size

    }*/

    public TensorRef to(TensorRef a, String device){
        //ask the allocator to do this
        return null;
    }

}
//This is an internal enum used ONLY be the tensor library to say "I cant do this. like SIMD cant do bf16* q4 operation
enum OpSupport{
    Unsupported
}


class Allocator {
    //This is be a TensorAllocator howwever to must unerstand multiple devices
    //it must always allocate the tensor its not a "cache" that hands back raw tensors when its "full"

}
enum LeaseState {
    USED,
    UNUSED
}
abstract class Tensor {

    public abstract float get(int... dims) ;
    public abstract float get(int row, int column);
    public abstract void set(float v, int row, int column) ;
    public abstract void set(float v, int... dims);
}

class F32Tensor extends Tensor{
    //final FloatBuffer underlyingByteBuffer;
    //final MemorySegment segment;

    @Override
    public float get(int... dims) {
        throw new UnsupportedOperationException("bla");
    }

    @Override
    public float get(int row, int column) {
        throw new UnsupportedOperationException("bla");
    }

    @Override
    public void set(float v, int row, int column) {
        throw new UnsupportedOperationException("bla");
    }

    @Override
    public void set(float v, int... dims) {
        throw new UnsupportedOperationException("bla");
    }
}

class TensorRef implements AutoCloseable {

    private volatile LeaseState leaseState;
    Allocator allocator;
    Tensor underlying;
    private TensorRef hiddenState;
    TensorShape shape;
    DType dType;
    int stride;
    String device;

    public TensorShape getShape(){
        return this.shape;
    }
    public TensorShape shape(){
        return this.shape;
    }

    public int dims() {
        return shape.dims();
    }

    /** The caller is done with the object and can be retuned to the pool */
    @Override
    public void close() throws Exception {
        if (hiddenState != null) {
            throw new UnreachableException("Double close on tensor");
        }
        //we make the tensor hard to use after it is closed.
        hiddenState = new TensorRef();
        hiddenState.dType = dType;
        hiddenState.stride = stride;
        hiddenState.device = device;
        hiddenState.shape = shape;
        leaseState = LeaseState.UNUSED;
        shape = null;
        dType = null;
        stride = 0;
        device = null;
        //alloator.signalFree()
    }
}

public interface TensorOps {
     Either<OpSupport, Void> multiplyAccumulate(TensorRef a, TensorRef b, int offset, int length);
}

class NaiveOps implements TensorOps {

    @Override
    public Either<OpSupport, Void> multiplyAccumulate(TensorRef a, TensorRef b, int offset, int length) {
        Preconditions.checkArgument(a.dims() == b.dims());

        boolean isBatch = b.shape().first() > 1;
        for (int ai = 0; ai < a.shape().first(); ai++) {
            //AbstractTensor as = a.slice(ai);
            //AbstractTensor bs = isBatch ? b.slice(ai) : b;
            //for (int i = offset; i < offset + length; ++i) {
            //    as.set(as.get(0, i) + bs.get(0, i), 0, i);
            //}
        }
        return Either.Right(null);
    }
}

class MeteredOps {
    private final MetricRegistry metricRegistry;
    private final TensorOps tensorOps;
    public MeteredOps(MetricRegistry metricRegistry, TensorOps ops) {
        this.metricRegistry = metricRegistry;
        this.tensorOps = ops;
    }

    public Either<OpSupport,Void> multiplyAccumulate(MultiplyAccumulate args, Map<String, String> tags){
        tags.put("ops", tensorOps.getClass().getName());
        Timer r = metricRegistry.timer( new MetricName("multiply_accumulate", tags));
        long start = System.currentTimeMillis();
        Either<OpSupport, Void> result = tensorOps.multiplyAccumulate(args.getA(), args.getB(), args.getOffset(), args.getLength());
        long end = System.currentTimeMillis();
        if (result.isRight()) {
            r.update(end-start, TimeUnit.MILLISECONDS);
        }
        return result;
    }



}

class MultiplyAccumulate {
    private final TensorRef b;
    private TensorRef a;
    private int offset;
    private int length;
    MultiplyAccumulate(TensorRef b){
        this.b = b;
    }
    MultiplyAccumulate into(TensorRef a){
        this.a = a;
        return this;
    }
    MultiplyAccumulate offsetAndLength(int offset, int length){
        this.offset = offset;
        this.length = length;
        return this;
    }

    public TensorRef getB() {
        return b;
    }

    public TensorRef getA() {
        return a;
    }

    public void setA(TensorRef a) {
        this.a = a;
    }

    public int getOffset() {
        return offset;
    }

    public void setOffset(int offset) {
        this.offset = offset;
    }

    public int getLength() {
        return length;
    }

    public void setLength(int length) {
        this.length = length;
    }
}