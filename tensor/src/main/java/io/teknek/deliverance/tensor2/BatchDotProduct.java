package io.teknek.deliverance.tensor2;

public class BatchDotProduct {
    private TensorRef result;
    private TensorRef a;
    private TensorRef b;
    private int aRowOffset;
    private int aColumnOffset;
    private int bColumnOffset;
    private int columnLength;
    private int resultRowOffset;
    private int bRowOffset;
    private int rowChunkSize;

    public BatchDotProduct result(TensorRef result) {
        this.result = result;
        return this;
    }

    public BatchDotProduct a(TensorRef a) {
        this.a = a;
        return this;
    }

    public BatchDotProduct b(TensorRef b) {
        this.b = b;
        return this;
    }

    public BatchDotProduct aRowOffset(int aRowOffset) {
        this.aRowOffset = aRowOffset;
        return this;
    }

    public BatchDotProduct aColumnOffset(int aColumnOffset) {
        this.aColumnOffset = aColumnOffset;
        return this;
    }

    public BatchDotProduct bColumnOffset(int bColumnOffset) {
        this.bColumnOffset = bColumnOffset;
        return this;
    }

    public BatchDotProduct columnLength(int columnLength) {
        this.columnLength = columnLength;
        return this;
    }

    public BatchDotProduct resultRowOffset(int resultRowOffset) {
        this.resultRowOffset = resultRowOffset;
        return this;
    }

    public BatchDotProduct bRowOffset(int bRowOffset) {
        this.bRowOffset = bRowOffset;
        return this;
    }

    public BatchDotProduct rowChunkSize(int rowChunkSize) {
        this.rowChunkSize = rowChunkSize;
        return this;
    }

    TensorRef result() {
        return result;
    }

    TensorRef a() {
        return a;
    }

    TensorRef b() {
        return b;
    }

    int aRowOffset() {
        return aRowOffset;
    }

    int aColumnOffset() {
        return aColumnOffset;
    }

    int bColumnOffset() {
        return bColumnOffset;
    }

    int columnLength() {
        return columnLength;
    }

    int resultRowOffset() {
        return resultRowOffset;
    }

    int bRowOffset() {
        return bRowOffset;
    }

    int rowChunkSize() {
        return rowChunkSize;
    }
}
