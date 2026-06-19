package io.teknek.deliverance.model.tensorparallel.transport;

import com.codahale.metrics.MetricRegistry;
import com.codahale.metrics.Timer;
import io.netty.bootstrap.Bootstrap;
import io.netty.buffer.ByteBuf;
import io.netty.channel.Channel;
import io.netty.channel.ChannelHandlerContext;
import io.netty.channel.ChannelInitializer;
import io.netty.channel.ChannelOption;
import io.netty.channel.SimpleChannelInboundHandler;
import io.netty.channel.nio.NioEventLoopGroup;
import io.netty.channel.socket.SocketChannel;
import io.netty.channel.socket.nio.NioSocketChannel;
import io.netty.handler.codec.LengthFieldBasedFrameDecoder;
import io.netty.handler.codec.LengthFieldPrepender;
import io.teknek.deliverance.JsonUtils;
import io.teknek.deliverance.model.InferenceProfiler;
import io.teknek.deliverance.model.tensorparallel.TensorParallelCollectives;
import io.teknek.deliverance.model.tensorparallel.TensorParallelContext;
import io.teknek.deliverance.tensor.AbstractTensor;

import java.io.IOException;
import java.net.URI;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Netty TCP client implementation of tensor-parallel collectives.
 */
public class NettyTensorParallelCollectives implements TensorParallelCollectives, AutoCloseable {
    private static final MetricRegistry METRICS = new MetricRegistry();
    private static final int MAX_FRAME_LENGTH = 256 * 1024 * 1024;
    private static final byte MSG_ALL_REDUCE_SUM = 1;
    private static final byte MSG_TENSOR_RESPONSE = 2;
    private static final byte MSG_ERROR = 3;

    private final TensorParallelContext context;
    private final URI baseUri;
    private final TensorPayloadCodec codec = new BinaryTensorPayloadCodec();
    private final AtomicLong collectiveSequence = new AtomicLong();
    private final NioEventLoopGroup group = new NioEventLoopGroup(1);
    private volatile Channel channel;
    private volatile CompletableFuture<byte[]> pendingResponse;

    public NettyTensorParallelCollectives(TensorParallelContext context, URI baseUri) {
        this.context = context;
        this.baseUri = baseUri;
    }

    @Override
    public synchronized AbstractTensor allReduceSum(String key, AbstractTensor local) {
        String wireKey = collectiveSequence.getAndIncrement() + ":" + key;
        byte[] body;
        try (Timer.Context ignored = InferenceProfiler.timer(METRICS, "collective.client.serialize").time()) {
            try {
                byte[] header = JsonUtils.om.writeValueAsBytes(new AllReduceSumRequest(wireKey, context.rank(), context.size()));
                byte[] tensor = codec.encode(local);
                body = ByteBuffer.allocate(1 + Integer.BYTES + header.length + tensor.length)
                        .order(ByteOrder.LITTLE_ENDIAN)
                        .put(MSG_ALL_REDUCE_SUM)
                        .putInt(header.length)
                        .put(header)
                        .put(tensor)
                        .array();
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }
        pendingResponse = new CompletableFuture<>();
        ByteBuf out = channel().alloc().buffer(body.length);
        out.writeBytes(body);
        try (Timer.Context ignored = InferenceProfiler.timer(METRICS, "collective.client.netty").time()) {
            channel().writeAndFlush(out).syncUninterruptibly();
            try {
                byte[] response = pendingResponse.get(30, TimeUnit.SECONDS);
                try (Timer.Context ignoredDecode = InferenceProfiler.timer(METRICS, "collective.client.deserialize").time()) {
                    return codec.decode(response);
                }
            } catch (Exception e) {
                throw new RuntimeException("Netty collective request failed", e);
            } finally {
                pendingResponse = null;
            }
        }
    }

    private Channel channel() {
        Channel current = channel;
        if (current != null && current.isActive()) {
            return current;
        }
        try {
            channel = new Bootstrap()
                    .group(group)
                    .channel(NioSocketChannel.class)
                    .option(ChannelOption.TCP_NODELAY, true)
                    .handler(new ChannelInitializer<SocketChannel>() {
                        @Override
                        protected void initChannel(SocketChannel ch) {
                            ch.pipeline().addLast(new LengthFieldBasedFrameDecoder(MAX_FRAME_LENGTH, 0, 4, 0, 4));
                            ch.pipeline().addLast(new LengthFieldPrepender(4));
                            ch.pipeline().addLast(new Handler());
                        }
                    })
                    .connect(baseUri.getHost(), baseUri.getPort())
                    .sync()
                    .channel();
            return channel;
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new RuntimeException("Unable to connect Netty collective client to " + baseUri, e);
        }
    }

    @Override
    public synchronized void close() {
        if (channel != null) {
            channel.close().awaitUninterruptibly();
            channel = null;
        }
        group.shutdownGracefully();
    }

    private final class Handler extends SimpleChannelInboundHandler<ByteBuf> {
        @Override
        protected void channelRead0(ChannelHandlerContext ctx, ByteBuf frame) {
            byte messageType = frame.readByte();
            byte[] bytes = new byte[frame.readableBytes()];
            frame.readBytes(bytes);
            CompletableFuture<byte[]> future = pendingResponse;
            if (future == null) {
                return;
            }
            if (messageType == MSG_TENSOR_RESPONSE) {
                future.complete(bytes);
            } else if (messageType == MSG_ERROR) {
                future.completeExceptionally(new IllegalStateException(new String(bytes, StandardCharsets.UTF_8)));
            } else {
                future.completeExceptionally(new IllegalStateException("Unsupported collective response type " + messageType));
            }
        }

        @Override
        public void exceptionCaught(ChannelHandlerContext ctx, Throwable cause) {
            CompletableFuture<byte[]> future = pendingResponse;
            if (future != null) {
                future.completeExceptionally(cause);
            }
            ctx.close();
        }
    }
}
