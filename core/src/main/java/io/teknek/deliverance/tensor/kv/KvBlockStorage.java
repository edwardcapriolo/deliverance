package io.teknek.deliverance.tensor.kv;

import io.teknek.deliverance.DType;
import io.teknek.deliverance.tensor.AbstractTensor;

interface KvBlockStorage extends AutoCloseable {
    KvBlockLayout layout();

    DType dtype();

    int layers();

    int tokenCount();

    int blockSize();

    int kvLength();

    long denseBytesEquivalent();

    long encodedBytes();

    AbstractTensor rowView(int layer, int blockRow, int keyOrValue);

    void copyRow(int layer, int blockRow, int keyOrValue, AbstractTensor destination);

    void copyRows(int layer, int keyOrValue, int blockRowStart, int rowCount, AbstractTensor destination,
            int destinationRowStart);

    @Override
    void close();
}
