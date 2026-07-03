package io.teknek.deliverance.tensor;

import com.codahale.metrics.Meter;
import com.codahale.metrics.MetricRegistry;
import com.codahale.metrics.Timer;
import com.google.common.collect.Maps;

import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Function;

import io.teknek.deliverance.DType;
import io.teknek.deliverance.tensor.impl.*;
import org.jctools.queues.MpmcUnboundedXaddArrayQueue;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Internally uses unbounded arrays per shape well tracking the total bytes concurrently.
 */
public class ArrayQueueTensorAllocator implements TensorAllocator {

    private static final Logger logger = LoggerFactory.getLogger(ArrayQueueTensorAllocator.class);

    private final long bytesCapacity;
    private final AtomicLong currentBytes;
    private final ConcurrentMap<ShapeKey, MpmcUnboundedXaddArrayQueue<AbstractTensor>> availableByShape;

    private final MetricRegistry metricRegistry;
    private final Function<ShapeKey, MpmcUnboundedXaddArrayQueue<AbstractTensor>> queueFactory = s -> new MpmcUnboundedXaddArrayQueue<>(
            128
    );

    private final Meter dirtyGet;
    private final Meter dirtyGetHit;
    private final Meter cacheFull;

    public ArrayQueueTensorAllocator(MetricRegistry metricRegistry){
        this(100 * 1024 * 1024, metricRegistry);
    }

    public ArrayQueueTensorAllocator(long bytesCapacity, MetricRegistry metricRegistry) {
        this.bytesCapacity = bytesCapacity;
        this.currentBytes = new AtomicLong(0);
        this.availableByShape = Maps.newConcurrentMap();
        this.metricRegistry = metricRegistry;
        dirtyGet = metricRegistry.meter("tensorcache.dirtyget");
        dirtyGetHit = metricRegistry.meter("tensorcache.getdirty.hit");
        cacheFull = metricRegistry.meter("tensorcache.full");
    }

    /**
     * Allocate a tensor of the desired shape. If the available queue is not full the shape will be added 
     * @param dType
     * @param shape
     * @return
     */
    private AbstractTensor internalGet(DType dType, TensorShape shape){
        AbstractTensor t = switch (dType) {
            case F32 -> new FloatBufferTensor(shape);
            case F16 -> new Float16BufferTensor(shape);
            case BF16 -> new BFloat16BufferTensor(shape);
            case I8 -> new Q8ByteBufferTensor(shape);
            case Q4 -> new Q4ByteBufferTensor(shape);
            default -> throw new RuntimeException("Unsupported tensor type: " + dType);
        };
        if (currentBytes.addAndGet(t.size()) < bytesCapacity) {
            t.setOwnerCache(this);
        } else {
            cacheFull.mark();
            currentBytes.addAndGet(-t.size());
        }
        return t;
    }


    public AbstractTensor getDirty(DType dType, TensorShape shape){
        dirtyGet.mark();
        MpmcUnboundedXaddArrayQueue<AbstractTensor> availableQueue = availableByShape.computeIfAbsent(
                new ShapeKey(dType, shape),
                queueFactory
        );
        AbstractTensor t = availableQueue.poll();
        if (t != null) {
            dirtyGetHit.mark();
            return t;
        }
        return internalGet(dType, shape);
    }


    public AbstractTensor get(DType dType, TensorShape shape) {
        metricRegistry.meter("tensorcache.get").mark();
        MpmcUnboundedXaddArrayQueue<AbstractTensor> availableQueue = availableByShape.computeIfAbsent(
                new ShapeKey(dType, shape),
                queueFactory);
        AbstractTensor t = availableQueue.poll();
        if (t != null) {
            metricRegistry.meter("tensorcache.get.hit").mark();
            Timer timer = metricRegistry.timer("tensorcache.zero-out-time");
            timer.time(t::clear);
            return t;
        }
        return internalGet(dType, shape);
    }


    public void release(AbstractTensor b) {
        MpmcUnboundedXaddArrayQueue<AbstractTensor> availableQueue = availableByShape.computeIfAbsent(
                new ShapeKey(b.dType(), b.shape()),
                queueFactory);
        boolean ignored = availableQueue.offer(b);
    }
}
