package io.teknek.deliverance.tensor.operations;
import com.google.common.base.Preconditions;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.AbstractTensorUtils;
import io.teknek.deliverance.tensor.TensorMutability;
import net.jafama.FastMath;

public class NaiveTensorOperations implements TensorOperations {
    @Override
    public String name() {
        return "Naive Java Operations";
    }

    @Override
    public int parallelSplitSize() {
        return 32; // was Integer.MAV_VALUE seems overkill
    }

    // a[0..n] += b[0..n]
    @Override
    public void accumulate(AbstractTensor a, AbstractTensor b, int offset, int length) {
        Preconditions.checkArgument(a.dims() == b.dims());

        boolean isBatch = b.shape().first() > 1;
        for (int ai = 0; ai < a.shape().first(); ai++) {
            AbstractTensor as = a.slice(ai);
            AbstractTensor bs = isBatch ? b.slice(ai) : b;
            for (int i = offset; i < offset + length; ++i) {
                as.set(as.get(0, i) + bs.get(0, i), 0, i);
            }
        }
    }

    // a[0..n] *= b[0..n]
    @Override
    public void maccumulate(AbstractTensor a, AbstractTensor b, int offset, int length) {
        Preconditions.checkArgument(a.dims() == b.dims());
        Preconditions.checkArgument(a.shape().last() == b.shape().last());
        Preconditions.checkArgument(b.shape().first() == 1 || a.shape().first() == b.shape().first());
        Preconditions.checkArgument(offset >= 0 && length >= 0 && offset + length <= a.shape().last());

        boolean isBatch = b.shape().first() > 1;
        for (int ai = 0; ai < a.shape().first(); ai++) {
            AbstractTensor as = a.slice(ai);
            AbstractTensor bs = isBatch ? b.slice(ai) : b;
            for (int i = offset; i < offset + length; ++i) {
                as.set(as.get(0, i) * bs.get(0, i), 0, i);
            }
        }
    }

    @Override
    public float dotProduct(AbstractTensor a, AbstractTensor b, int aoffset, int boffset, int limit) {
        Preconditions.checkArgument(a.dims() == b.dims() && a.shape().first() == 1);

        int alen = aoffset + limit;
        int blen = boffset + limit;

        float s = 0;
        for (; aoffset < alen && boffset < blen; aoffset++, boffset++) {
            s += a.get(0, aoffset) * b.get(0, boffset);
        }

        return s;
    }

    @Override
    public void batchDotProduct(
            AbstractTensor result,
            AbstractTensor a,
            AbstractTensor b,
            int aColumnOffset,
            int bColumnOffset,
            int columnLength,
            int rRowOffset,
            int bRowOffset,
            int rowChunkSize
    ) {
        TensorMutability.requireWritable(result, "batchDotProduct");
        a = TensorMutability.unwrapReadOnly(a);
        b = TensorMutability.unwrapReadOnly(b);
        Preconditions.checkArgument(a.dims() == 2 && b.dims() == 2 && result.dims() == 2);

        int bRowLimit = bRowOffset + rowChunkSize;

        for (int i = 0; i < a.shape().first(); i++) {
            for (int j = bRowOffset; j < bRowLimit; j++) {
                float d = dotProduct(a.slice(i), b.slice(j), aColumnOffset, bColumnOffset, columnLength);
                result.set(d, i, j + rRowOffset);
            }
        }
    }

    // Computes a constant times a vector plus a vector (single-precision).
    // On return, the contents of vector Y are replaced with the result. The value computed is (alpha * X[i]) + Y[i].
    @Override
    public void saxpy(float alpha, AbstractTensor x, AbstractTensor y, int xoffset, int yoffset, int limit) {
        Preconditions.checkArgument(x.shape().first() == 1 && y.shape().first() == 1);
        for (int xo = xoffset, yo = yoffset; xo < (xoffset + limit) && yo < (yoffset + limit); xo++, yo++) {
            float v = (alpha * x.get(0, xo)) + y.get(0, yo);
            y.set(v, 0, yo);
        }
    }

    @Override
    public void scale(float factor, AbstractTensor x, int offset, int length) {
        int limit = offset + length;

        for (int b = 0; b < x.shape().first(); b++)
            for (int i = offset; i < limit; ++i)
                x.set(x.get(b, i) * factor, b, i);
    }

    @Override
    public float max(AbstractTensor input, int row, int offset, int length) {
        float max = input.get(row, offset);
        for (int i = offset + 1; i < offset + length; i++) {
            max = Math.max(max, input.get(row, i));
        }
        return max;
    }

    @Override
    public void argMax(AbstractTensor input, AbstractTensor output, int offset, int length) {
        TensorOperations.super.argMax(input, output, offset, length);
    }

    @Override
    public float sum(AbstractTensor input, int row, int offset, int length) {
        return TensorOperations.super.sum(input, row, offset, length);
    }

    @Override
    public void exp(AbstractTensor input, AbstractTensor output, int offset, int length) {
        TensorMutability.requireWritable(output, "exp");
        if (!input.shape().equals(output.shape())) {
            throw new IllegalArgumentException("input and output must have same shape");
        }
        int limit = offset + length;
        for (int row = 0; row < input.shape().first(); row++) {
            for (int i = offset; i < limit; i++) {
                output.set((float) FastMath.exp(input.get(row, i)), row, i);
            }
        }
    }

    @Override
    public AbstractTensor quantize(AbstractTensor t, DType qtype, int offset, int length) {
        return AbstractTensorUtils.quantize(t, qtype, true);
        //return null;
    }
}
