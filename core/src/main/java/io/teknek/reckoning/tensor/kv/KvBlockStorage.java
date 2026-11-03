package io.teknek.reckoning.tensor.kv;

import io.teknek.reckoning.DType;
import io.teknek.reckoning.tensor2.TensorRef;

interface KvBlockStorage extends AutoCloseable {
    KvBlockLayout layout();

    DType dtype();

    DType dtype(int keyOrValue);

    int layers();

    int tokenCount();

    int blockSize();

    int kvLength();

    int layerKvLength(int layer);

    long denseBytesEquivalent();

    long encodedBytes();

    TensorRef rowView(int layer, int blockRow, int keyOrValue);

    TensorRef pageView(int layer, int keyOrValue);

    void copyRow(int layer, int blockRow, int keyOrValue, TensorRef destination);

    void copyRows(int layer, int keyOrValue, int blockRowStart, int rowCount, TensorRef destination,
            int destinationRowStart);

    @Override
    void close();
}
