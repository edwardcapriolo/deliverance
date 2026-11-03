package io.teknek.reckoning.model.tensorparallel.transport;

import io.teknek.reckoning.tensor.AbstractTensor;
import io.teknek.reckoning.tensor2.TensorRef;

public interface TensorPayloadCodec {
    String contentType();

    byte[] encode(TensorRef tensor);

    byte[] encode(AbstractTensor tensor);

    TensorRef decode(byte[] bytes);

    AbstractTensor decodeAbstract(byte[] bytes);
}
