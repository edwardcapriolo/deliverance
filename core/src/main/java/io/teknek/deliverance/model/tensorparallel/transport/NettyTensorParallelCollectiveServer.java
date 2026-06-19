package io.teknek.deliverance.model.tensorparallel.transport;

import com.codahale.metrics.MetricRegistry;
import com.codahale.metrics.Timer;
import io.netty.bootstrap.ServerBootstrap;
import io.netty.buffer.ByteBuf;
import io.netty.channel.Channel;
import io.netty.channel.ChannelFuture;
import io.netty.channel.ChannelHandlerContext;
import io.netty.channel.ChannelInitializer;
import io.netty.channel.ChannelOption;
import io.netty.channel.SimpleChannelInboundHandler;
import io.netty.channel.nio.NioEventLoopGroup;
import io.netty.channel.socket.SocketChannel;
import io.netty.channel.socket.nio.NioServerSocketChannel;
import io.netty.handler.codec.LengthFieldBasedFrameDecoder;
import io.netty.handler.codec.LengthFieldPrepender;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.JsonUtils;
import io.teknek.deliverance.model.InferenceProfiler;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.net.InetSocketAddress;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * Coordinator-hosted Netty TCP collective server for tensor-parallel all-reduce operations.
 */
public class NettyTensorParallelCollectiveServer implements AutoCloseable {
    private static final Logger LOGGER = LoggerFactory.getLogger(NettyTensorParallelCollectiveServer.class);
    private static final MetricRegistry METRICS = new MetricRegistry();
    private static final int MAX_FRAME_LENGTH = 256 * 1024 * 1024;
    private static final byte MSG_ALL_REDUCE_SUM = 1;
    private static final byte MSG_TENSOR_RESPONSE = 2;
    private static final byte MSG_ERROR = 3;

    private final InetSocketAddress address;
    private final NioEventLoopGroup bossGroup = new NioEventLoopGroup(1);
    private final NioEventLoopGroup workerGroup = new NioEventLoopGroup();
    private final ExecutorService applicationExecutor = Executors.newCachedThreadPool();
    private final Group group;
    private final TensorPayloadCodec codec = new BinaryTensorPayloadCodec();
    private Channel channel;

    public NettyTensorParallelCollectiveServer(InetSocketAddress address, Duration timeout) {
        this.address = address;
        this.group = new Group(timeout);
    }

    public void start() {
        try {
            channel = new ServerBootstrap()
                    .group(bossGroup, workerGroup)
                    .channel(NioServerSocketChannel.class)
                    .childOption(ChannelOption.TCP_NODELAY, true)
                    .childHandler(new ChannelInitializer<SocketChannel>() {
                        @Override
                        protected void initChannel(SocketChannel ch) {
                            ch.pipeline().addLast(new LengthFieldBasedFrameDecoder(MAX_FRAME_LENGTH, 0, 4, 0, 4));
                            ch.pipeline().addLast(new LengthFieldPrepender(4));
                            ch.pipeline().addLast(new Handler());
                        }
                    })
                    .bind(address)
                    .sync()
                    .channel();
            LOGGER.info("Started Netty tensor-parallel collective server uri={}", uri());
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new RuntimeException("Unable to start tensor-parallel Netty collective server", e);
        }
    }

    public URI uri() {
        InetSocketAddress bound = (InetSocketAddress) channel.localAddress();
        return URI.create("netty://" + address.getHostString() + ":" + bound.getPort());
    }

    @Override
    public void close() {
        LOGGER.info("Closing Netty tensor-parallel collective server uri={}", channel == null ? address : uri());
        if (channel != null) {
            channel.close().awaitUninterruptibly();
        }
        applicationExecutor.shutdownNow();
        bossGroup.shutdownGracefully();
        workerGroup.shutdownGracefully();
    }

    private final class Handler extends SimpleChannelInboundHandler<ByteBuf> {
        @Override
        protected void channelRead0(ChannelHandlerContext ctx, ByteBuf frame) {
            byte messageType = frame.readByte();
            if (messageType != MSG_ALL_REDUCE_SUM) {
                writeError(ctx, "Unsupported collective message type " + messageType);
                return;
            }
            byte[] requestBytes = new byte[frame.readableBytes()];
            frame.readBytes(requestBytes);
            applicationExecutor.execute(() -> handleAllReduceSum(ctx, requestBytes));
        }
    }

    private void handleAllReduceSum(ChannelHandlerContext ctx, byte[] requestBytes) {
        DecodedRequest decoded;
        try (Timer.Context ignored = InferenceProfiler.timer(METRICS, "collective.server.decode").time()) {
            decoded = decodeRequest(requestBytes);
        } catch (RuntimeException e) {
            writeError(ctx, e.getMessage());
            return;
        }
        try (AbstractTensor local = decoded.local();
             AbstractTensor reduced = group.allReduceSum(decoded.request().key(), decoded.request().rank(), decoded.request().size(), local)) {
            byte[] response;
            try (Timer.Context ignored = InferenceProfiler.timer(METRICS, "collective.server.encode").time()) {
                response = codec.encode(reduced);
            }
            ByteBuf out = ctx.alloc().buffer(1 + response.length);
            out.writeByte(MSG_TENSOR_RESPONSE);
            out.writeBytes(response);
            ctx.writeAndFlush(out);
        } catch (RuntimeException e) {
            writeError(ctx, e.getMessage());
        }
    }

    private DecodedRequest decodeRequest(byte[] requestBytes) {
        java.nio.ByteBuffer bodyBuffer = java.nio.ByteBuffer.wrap(requestBytes).order(java.nio.ByteOrder.LITTLE_ENDIAN);
        int headerLength = bodyBuffer.getInt();
        if (headerLength < 1 || headerLength > bodyBuffer.remaining()) {
            throw new IllegalArgumentException("Invalid allReduceSum header length " + headerLength);
        }
        byte[] header = new byte[headerLength];
        bodyBuffer.get(header);
        byte[] tensorBytes = new byte[bodyBuffer.remaining()];
        bodyBuffer.get(tensorBytes);
        try {
            AllReduceSumRequest request = JsonUtils.om.readValue(header, AllReduceSumRequest.class);
            return new DecodedRequest(request, codec.decode(tensorBytes));
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    private void writeError(ChannelHandlerContext ctx, String message) {
        byte[] bytes = String.valueOf(message).getBytes(StandardCharsets.UTF_8);
        ByteBuf out = ctx.alloc().buffer(1 + bytes.length);
        out.writeByte(MSG_ERROR);
        out.writeBytes(bytes);
        ctx.writeAndFlush(out);
    }

    private record DecodedRequest(AllReduceSumRequest request, AbstractTensor local) {
    }

    private static class Group {
        private final Duration timeout;
        private final Map<String, Round> rounds = new HashMap<>();

        private Group(Duration timeout) {
            this.timeout = timeout;
        }

        private synchronized AbstractTensor allReduceSum(String key, int rank, int size, AbstractTensor local) {
            Round round = rounds.computeIfAbsent(key, ignored -> new Round(size, local));
            round.add(rank, local);
            if (round.arrived == size) {
                try (Timer.Context ignored = InferenceProfiler.timer(METRICS, "collective.server.reduce").time()) {
                    round.reduce();
                }
                notifyAll();
            }
            try (Timer.Context ignored = InferenceProfiler.timer(METRICS, "collective.server.wait").time()) {
                waitForReduction(key, round);
            }
            AbstractTensor result = round.copyReduced();
            round.returned++;
            if (round.returned == size) {
                round.closeReduced();
                rounds.remove(key);
            }
            return result;
        }

        private void waitForReduction(String key, Round round) {
            long deadline = System.nanoTime() + timeout.toNanos();
            while (round.reduced == null) {
                long remainingNanos = deadline - System.nanoTime();
                if (remainingNanos <= 0) {
                    throw new IllegalStateException("Timed out waiting for allReduceSum key=" + key
                            + " expected=" + round.size + " arrived=" + round.arrived);
                }
                try {
                    wait(Math.max(1L, remainingNanos / 1_000_000L));
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    throw new IllegalStateException("Interrupted waiting for allReduceSum key=" + key, e);
                }
            }
        }
    }

    private static class Round {
        private final int size;
        private final int[] shape;
        private final DType dType;
        private final AbstractTensor[] contributions;
        private int arrived;
        private int returned;
        private AbstractTensor reduced;

        private Round(int size, AbstractTensor first) {
            this.size = size;
            this.shape = first.shape().shapeArray();
            this.dType = first.dType();
            this.contributions = new AbstractTensor[size];
        }

        private void add(int rank, AbstractTensor local) {
            if (contributions[rank] != null) {
                throw new IllegalStateException("rank " + rank + " already contributed to this allReduceSum");
            }
            if (local.dType() != dType || !Arrays.equals(local.shape().shapeArray(), shape)) {
                throw new IllegalArgumentException("allReduceSum contributions must have matching dtype and shape");
            }
            contributions[rank] = copy(local);
            arrived++;
        }

        private void reduce() {
            reduced = new FloatBufferTensor(shape);
            for (AbstractTensor contribution : contributions) {
                for (int row = 0; row < reduced.shape().first(); row++) {
                    for (int col = 0; col < reduced.shape().last(); col++) {
                        reduced.set(reduced.get(row, col) + contribution.get(row, col), row, col);
                    }
                }
                contribution.close();
            }
        }

        private AbstractTensor copyReduced() {
            return copy(reduced);
        }

        private void closeReduced() {
            if (reduced != null) {
                reduced.close();
                reduced = null;
            }
        }

        private static AbstractTensor copy(AbstractTensor source) {
            if (source.dType() != DType.F32) {
                throw new UnsupportedOperationException("Netty collectives currently support F32 tensors");
            }
            AbstractTensor copy = new FloatBufferTensor(source.shape());
            copy.copyFrom(source, 0, 0, (int) source.size());
            return copy;
        }
    }
}
