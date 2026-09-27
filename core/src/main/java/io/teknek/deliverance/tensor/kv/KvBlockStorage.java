package io.teknek.deliverance.tensor.kv;

import io.teknek.deliverance.DType;
import io.teknek.deliverance.tensor2.TensorRef;

interface KvBlockStorage extends AutoCloseable {
    KvBlockLayout layout();

    DType dtype();

    DType dtype(int keyOrValue);

    int layers();

    int tokenCount();

    int blockSize();

    int kvLength();

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
