package io.teknek.deliverance.tensor2;

import com.google.common.base.Preconditions;
import io.dropwizard.metrics5.MetricName;
import io.dropwizard.metrics5.MetricRegistry;
import io.dropwizard.metrics5.Timer;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.tensor.TensorShape;
import io.teknek.dysfx.Either;

import java.util.EnumMap;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.TimeUnit;

public class Lighter {
    private final Allocator allocator = new Allocator();
    private final MetricRegistry metricRegistry;
    //something like this must be initialized  so later we can pick the right one
    private final EnumMap<TensorProviderKind, TensorOps> tensorOperations = new EnumMap<>(TensorProviderKind.class);
    public Lighter(){
        this(new MetricRegistry());
    }

    public Lighter(MetricRegistry metricRegistry){
        this(metricRegistry, Map.of(
                TensorProviderKind.PANAMA, new PanamaOps(),
                TensorProviderKind.NAIVE, new NaiveOps()
        ));
    }

    public Lighter(MetricRegistry metricRegistry, Map<TensorProviderKind, TensorOps> tensorOperations){
        this.metricRegistry = metricRegistry;
        this.tensorOperations.putAll(tensorOperations);
    }

    public TensorRef allocate(DType dType, TensorShape shape) {
        return allocator.allocate(dType, shape, "cpu");
    }

    public TensorRef allocate(DType dType, TensorShape shape, String device) {
        return allocator.allocate(dType, shape, device);
    }

    public void multiplyAccumulate(MultiplyAccumulate multiplyAccumulate){
        multiplyAccumulate(multiplyAccumulate, Map.of());
    }

    public void multiplyAccumulate(MultiplyAccumulate multiplyAccumulate, Map<String, String> tags){
        TensorRef source = multiplyAccumulate.getSource();
        TensorRef destination = multiplyAccumulate.getDestination();
        Preconditions.checkArgument(destination != null, "Destination tensor must be set");
        Preconditions.checkArgument(source != null, "Source tensor must be set");
        Preconditions.checkArgument(destination.device().equals(source.device()), "Tensors must be on the same device");
        Preconditions.checkArgument(destination.dims() == source.dims(), "Tensors must have the same rank");
        Preconditions.checkArgument(destination.shape().last() == source.shape().last(),
                "Tensors must have the same last dimension");
        Preconditions.checkArgument(source.shape().first() == 1 || destination.shape().first() == source.shape().first(),
                "Source tensor must be broadcastable over destination rows");

        for (Map.Entry<TensorProviderKind, TensorOps> entry : tensorOperations.entrySet()) {
            Map<String, String> metricTags = new HashMap<>(tags);
            metricTags.put("ops", entry.getKey().name());
            metricRegistry.meter(new MetricName("tensor2.multiply_accumulate", metricTags)).mark();
            Timer timer = metricRegistry.timer(new MetricName("tensor2.multiply_accumulate.time", metricTags));
            long startNanos = System.nanoTime();
            Either<OpSupport, Void> result = entry.getValue().multiplyAccumulate(destination, source,
                    multiplyAccumulate.getOffset(), multiplyAccumulate.getLength());
            if (result.isRight()) {
                timer.update(System.nanoTime() - startNanos, TimeUnit.NANOSECONDS);
                return;
            }
        }
        throw new IllegalStateException("No tensor operations support multiplyAccumulate");
    }

    public void scale(Scale scale) {
        scale(scale, Map.of());
    }

    public void scale(Scale scale, Map<String, String> tags) {
        TensorRef target = scale.getTarget();
        Preconditions.checkArgument(target != null, "Target tensor must be set");
        Preconditions.checkArgument(scale.getOffset() >= 0 && scale.getLength() >= 0
                && scale.getOffset() + scale.getLength() <= target.shape().last(), "Scale window out of bounds");

        for (Map.Entry<TensorProviderKind, TensorOps> entry : tensorOperations.entrySet()) {
            Map<String, String> metricTags = new HashMap<>(tags);
            metricTags.put("ops", entry.getKey().name());
            metricRegistry.meter(new MetricName("tensor2.scale", metricTags)).mark();
            Timer timer = metricRegistry.timer(new MetricName("tensor2.scale.time", metricTags));
            long startNanos = System.nanoTime();
            Either<OpSupport, Void> result = entry.getValue().scale(scale.getFactor(), target,
                    scale.getOffset(), scale.getLength());
            if (result.isRight()) {
                timer.update(System.nanoTime() - startNanos, TimeUnit.NANOSECONDS);
                return;
            }
        }
        throw new IllegalStateException("No tensor operations support scale");
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
