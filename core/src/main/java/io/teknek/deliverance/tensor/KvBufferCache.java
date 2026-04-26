package io.teknek.deliverance.tensor;

import com.google.common.base.Preconditions;
import io.teknek.deliverance.DType;

import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.safetensors.DistributedContext;
import io.teknek.deliverance.safetensors.Config;
import io.teknek.deliverance.tensor.impl.BFloat16BufferTensor;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.ShortBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Paths;
import java.util.*;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * A cache for key-value buffers used in the model.
 */
public class KvBufferCache implements Closeable {

    public static class PrefixEntry {
        public final KvBuffer buffer;
        public final int length;

        public PrefixEntry(KvBuffer buffer, int length) {
            this.buffer = buffer;
            this.length = length;
        }

    }
    private static final Logger logger = LoggerFactory.getLogger(KvBufferCache.class);

    private final AbstractModel model;
    private final KvBufferCacheSettings kvBufferCacheSettings;
    private final int blockSize;

    public final Map<List<Integer>, PrefixEntry> prefixCache = Collections.synchronizedMap(
            new LinkedHashMap<List<Integer>, PrefixEntry>(16, 0.75f, true) {
                public boolean removeEldestEntry(Map.Entry<List<Integer>, PrefixEntry> eldest) {
                    boolean evict= size()> kvBufferCacheSettings.getMaxEntries();
                    if (evict && eldest != null && eldest.getValue() != null){
                        model.getMetricRegistry().meter("kvbuffercache.evict").mark();
                        try { eldest.getValue().buffer.close(); }catch (RuntimeException e){
                            logger.warn("could not close tensor in cache",e);
                        }
                    }
                    return evict;
                }
            });

    public KvBufferCache(AbstractModel model, KvBufferCacheSettings kvBufferCacheSettings) {
        this.model = model;
        this.kvBufferCacheSettings = kvBufferCacheSettings;
        this.blockSize = kvBufferCacheSettings.getBlockSize();
    }

    public PrefixEntry lookupPrefix(int [] tokens){
        int limit = kvBufferCacheSettings.getMaxPrefixTokensPerPrompt();
        int len = Math.min(tokens.length, limit);
        List<Integer> soFar = new ArrayList<>();
        PrefixEntry best = null;
        int tokenLimit = Math.min(tokens.length, len);
        for (int i = 0; i < tokenLimit; i++) {
            soFar.add(tokens[i]);
            PrefixEntry e = prefixCache.get(soFar);
            if(e!= null){
                best = e;
            }
        }
        model.getMetricRegistry().meter("kvbuffercache.lookup").mark();
        if (best!= null && best.length >= blockSize && best.length % blockSize == 0){
            model.getMetricRegistry().meter("kvbuffercache.hits").mark();
            return best;
        }
        model.getMetricRegistry().meter("kvbuffercache.misses").mark();
        return null;
    }

    public void storePrefix(int [] tokens,KvBuffer buffer){
        int limit = kvBufferCacheSettings.getMaxPrefixTokensPerPrompt();
        int tokenLen = Math.min(tokens.length, limit);
        List<Integer> soFar = new ArrayList<>();
        for (int i =0; i< tokenLen; i++){
            soFar.add(tokens[i]);
            int prefixLen = i + 1;
            if (prefixLen < blockSize || prefixLen % blockSize != 0 ){
                continue;
            }
            KvBuffer snapshot = getEphemeralKvBuffer();
            copyPrefix(buffer, snapshot, prefixLen);
            prefixCache.putIfAbsent(new ArrayList<>(soFar), new PrefixEntry(snapshot, prefixLen));
        }
    }

    public void copyPrefix(KvBuffer src, KvBuffer dest, int length){
        Config c = model.getConfig();
        int layers = c.dctx().numberOfLayers;
        for( int layer=0; layer <  layers; layer++){
            for(int pos=0; pos<length; pos++){
                AbstractTensor srcK = src.getKeyTensorForPosition(layer, pos);
                AbstractTensor srcV = src.getValTensorForPosition(layer, pos);

                AbstractTensor dstK = dest.getKeyTensorForPosition(layer, pos);
                AbstractTensor dstV = dest.getValTensorForPosition(layer, pos);
                dstK.copyFrom(srcK,0,0, (int) srcK.size());
                dstV.copyFrom(srcV, 0,0, (int) srcV.size());
                srcK.close();
                srcV.close();
                dstK.close();
                dstV.close();
            }
        }
        dest.setCurrentContextPosition(length);
    }
    /*
    public KvBuffer getKvBuffer(String session) {
        return kvBufferCache.computeIfAbsent(session, s -> new KvBuffer(s, 1 << 23)); // 8MB per page
    }*/

    /**just orphans less memory :) */
    public KvBuffer getEphemeralKvBuffer() {
        return new KvBuffer(UUID.randomUUID().toString(), 1 << 20);
    }

    @Override
    public void close() {
        /*
        Iterator<Map.Entry<String, KvBuffer>> it = kvBufferCache.entrySet().iterator();
        while (it.hasNext()) {
            it.next().getValue().close();
            it.remove();
        }*/
    }

    class KvPageContext {
        public final int numberOfLayerPages;
        public final int numberOfContextPages;
        private final int layersPerPage;
        private final int contextLengthPerPage;
        private final String session;

        public final TensorShape pageShape;

        public KvPageContext(String session, int numberOfLayerPages, int numberOfContextPages, int layersPerPage, int contextLengthPerPage) {
            this.session = session;
            this.numberOfLayerPages = numberOfLayerPages;
            this.numberOfContextPages = numberOfContextPages;
            this.layersPerPage = layersPerPage;
            this.contextLengthPerPage = contextLengthPerPage;

            if (numberOfLayerPages < 1) {
                throw new IllegalArgumentException("totalPageCount must be >= 1");
            }
            if (numberOfContextPages < 1) {
                throw new IllegalArgumentException("numberOfContextPages must be >= 1");
            }
            if (layersPerPage < 1) {
                throw new IllegalArgumentException("layersPerPage must be >= 1");
            }
            if (contextLengthPerPage < 1) {
                throw new IllegalArgumentException("contextLengthPerPage must be >= 1");
            }

            TensorShape s;
            Config c = model.getConfig();
            DistributedContext dctx = c.dctx();
            int[] rawShape = new int[] { layersPerPage, 2, contextLengthPerPage, c.kvLength };

            // Adjust the shape to be relative to the kv cache size (in case of GQA)
            if (c.kvLength != dctx.kvSegmentLength) {
                SparseOffset<Integer> kvOffset = SparseOffset.of(dctx.kvSegmentStart, dctx.kvSegmentEnd);
                s = TensorShape.sparseColumn(rawShape, kvOffset);
            } else {
                s = TensorShape.of(rawShape);
            }
            this.pageShape = s;
        }
    }

    /**
     * A Page of a key-value buffer.
     * Rather than allocating one giant buffer for the entire key-value buffer, we allocate slices of the buffer
     * as needed. This allows us to keep the memory usage low, and also allows us to allocate very large contexts.
     */
    class KvBufferPage implements AutoCloseable {
        private final AbstractTensor tensor;

        private final AtomicBoolean closed = new AtomicBoolean(false);
        private final RandomAccessFile raf;

        KvBufferPage(KvPageContext pageCtx, String pageId) {
            //this looks more and more like two subclasses vs an if statement
            if (kvBufferCacheSettings.isEphemeral()) {
                this.raf = null;
                TensorAllocator tc = kvBufferCacheSettings.getDedicatedCache() == null ?
                        model.getTensorAllocator(): kvBufferCacheSettings.getDedicatedCache();
                this.tensor = tc.get(model.getWorkingDType(), pageCtx.pageShape);
            } else {
                try {
                    raf = new RandomAccessFile(
                            Paths.get(
                                    kvBufferCacheSettings.getWorkingDirectory().toString(),
                                    pageCtx.session.toString() + "-" + pageId + ".page"
                            ).toFile(),
                            "rw"
                    );
                    long bytes = pageCtx.pageShape.size() * model.getWorkingDType().size();
                    logger.debug("Allocating page {} with {} bytes {}", pageId, bytes, raf.length());
                    if (raf.length() != bytes) {
                        raf.setLength(bytes);
                    }

                    AbstractTensor<?,?> t;
                    if (model.getWorkingDType() == DType.F32) {
                        FloatBuffer fb = raf.getChannel()
                                .map(FileChannel.MapMode.READ_WRITE, 0, bytes)
                                .order(ByteOrder.LITTLE_ENDIAN)
                                .asFloatBuffer();

                        t = new FloatBufferTensor(fb, pageCtx.pageShape, true);
                    } else if (model.getWorkingDType() == DType.BF16) {
                        ShortBuffer sb = raf.getChannel()
                                .map(FileChannel.MapMode.READ_WRITE, 0, bytes)
                                .order(ByteOrder.LITTLE_ENDIAN)
                                .asShortBuffer();

                        t = new BFloat16BufferTensor("kvmem", sb, pageCtx.pageShape, true);
                    } else {
                        throw new UnsupportedOperationException("Only F32/BF16 is supported for now");
                    }
                    this.tensor = t;
                } catch (IOException e) {
                    throw new IOError(e);
                }
            }
        }

        public AbstractTensor getTensor() {
            assert !closed.get() : "Page is closed";
            return tensor;
        }

        public boolean isClosed() {
            return closed.get();
        }

        @Override
        public void close() throws IOException {
            if (closed.compareAndSet(false, true)) {
                if (raf != null) {
                    raf.close();
                }
                tensor.close();
            }
        }
    }

    public class KvBuffer implements AutoCloseable {
        private final String session;
        private final AtomicInteger currentContextPosition = new AtomicInteger(0);
        private final KvBufferPage[][] pages;

        private final KvPageContext pageContext;

            KvBuffer(String session, int maxPageSizeInBytes) {
            this.session = session;
            this.pageContext = computePageSize(maxPageSizeInBytes);
            this.pages = new KvBufferPage[pageContext.numberOfLayerPages][pageContext.numberOfContextPages];
        }

        public int getCurrentContextPosition() {
            return currentContextPosition.get();
        }

        public void setCurrentContextPosition(int position) {
            currentContextPosition.set(position);
        }

        /**
         * Currently this is called by AbstractModel inside the generation loop. The position starts at 0
         * interestingly the 0th token has special properties with respect to increment
         */
        public void incrementContextPosition() {
            currentContextPosition.incrementAndGet();
        }

        public KvPageContext computePageSize(long maxPageSizeInBytes) {
            Config c = model.getConfig();
            DType workingDType = model.getWorkingDType();
            long s = 2L * workingDType.size() * c.dctx().kvSegmentLength; // Size per layer per context

            Preconditions.checkArgument(maxPageSizeInBytes > s, "maxPageSizeInBytes must be greater than the size of a single layer");

            int N = c.dctx().numberOfLayers;
            int C = c.contextLength;

            int optimalLayersPerPage = 1;
            int optimalContextLengthPerPage = 1;
            long maxProduct = 0;

            // Try partitioning by layers
            for (int x = N; x >= 1; x--) {
                long y = maxPageSizeInBytes / (x * s);

                if (y >= 1 && y <= C) {
                    long product = x * y;

                    if (product > maxProduct) {
                        optimalLayersPerPage = x;
                        optimalContextLengthPerPage = (int) y;
                        maxProduct = product;
                    }
                    // Break if product starts decreasing
                    if (product < maxProduct) {
                        break;
                    }
                }
            }

            // Calculate the number of pages needed
            int numberOfLayerPages = (int) Math.ceil((double) N / optimalLayersPerPage);
            int numberOfContextPages = (int) Math.ceil((double) C / optimalContextLengthPerPage);

            // Calculate the size of each page
            long pageSize = optimalLayersPerPage * optimalContextLengthPerPage * s;

            if (pageSize > maxPageSizeInBytes) {
                throw new IllegalArgumentException(
                        "Calculation error: pageSize > maxPageSizeInBytes: " + pageSize + " > " + maxPageSizeInBytes
                );
            }

            logger.debug(
                    "Optimal page size: {} layers, {} context length, {} bytes, {} layer pages, {} length pages",
                    optimalLayersPerPage,
                    optimalContextLengthPerPage,
                    pageSize,
                    numberOfLayerPages,
                    numberOfContextPages
            );

            return new KvPageContext(session, numberOfLayerPages, numberOfContextPages, optimalLayersPerPage, optimalContextLengthPerPage);
        }

        @Override
        public void close() {
            for (KvBufferPage[] layerPages : pages) {
                if (layerPages != null) {
                    for (KvBufferPage page : layerPages) {
                        if (page != null) {
                            try {
                                page.close();
                            } catch (IOException e) {
                                logger.debug("Error closing page", e);
                            }
                        }
                    }
                }
            }
        }

        public AbstractTensor getKeyTensorForPosition(int layerIndex, int position) {
            return getTensorForPosition(layerIndex, position, 0);
        }

        public AbstractTensor getValTensorForPosition(int layerIndex, int position) {
            return getTensorForPosition(layerIndex, position, 1);
        }

        private AbstractTensor getTensorForPosition(int layerIndex, int position, int index) {
            // Calculate page indices and relative indices
            int layerPageIndex = layerIndex / pageContext.layersPerPage;
            int contextPageIndex = position / pageContext.contextLengthPerPage;
            int relativeLayerIndex = layerIndex % pageContext.layersPerPage;
            int relativeContextIndex = position % pageContext.contextLengthPerPage;

            KvBufferPage page = pages[layerPageIndex][contextPageIndex];
            if (page == null || page.isClosed()) {
                page = new KvBufferPage(pageContext, "L" + layerPageIndex + "C" + contextPageIndex);
                pages[layerPageIndex][contextPageIndex] = page;
            }

            return page.getTensor().slice(true, relativeLayerIndex, index, relativeContextIndex);
        }

        public AbstractTensor[] getKeyTensorsUptoPosition(int layerIndex, int upperBound) {
            return getTensorsUptoPosition(layerIndex, 0, upperBound);
        }

        public AbstractTensor[] getValTensorsUptoPosition(int layerIndex, int upperBound) {
            return getTensorsUptoPosition(layerIndex, 1, upperBound);
        }

        private AbstractTensor[] getTensorsUptoPosition(int layerIndex, int index, int upperBound) {
            int layerPageIndex = layerIndex / pageContext.layersPerPage;
            int contextPageIndex = upperBound / pageContext.contextLengthPerPage;
            int relativeLayerIndex = layerIndex % pageContext.layersPerPage;

            KvBufferPage[] layerPages = pages[layerPageIndex];

            AbstractTensor[] tensors = new AbstractTensor[contextPageIndex + 1];

            for (int i = 0; i <= contextPageIndex; i++) {
                KvBufferPage page = layerPages[i];

                if (page == null || page.isClosed()) {
                    page = new KvBufferPage(pageContext, "L" + layerPageIndex + "C" + contextPageIndex);
                    layerPages[i] = page;
                }

                tensors[i] = page.getTensor().slice(true, relativeLayerIndex, index);
            }

            return tensors;
        }

        @Override
        public String toString() {
            return "KvBuffer{" +
                    "session=" + session +
                    ", currentContextPosition=" + currentContextPosition +
                    ", pages=" + Arrays.toString(pages) +
                    ", pageContext=" + pageContext +
                    '}';
        }
    }


}