package io.teknek.deliverance.model.tensorparallel.transport;

import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor2.TensorRef;

public interface TensorPayloadCodec {
    String contentType();

    byte[] encode(TensorRef tensor);

    byte[] encode(AbstractTensor tensor);

    TensorRef decode(byte[] bytes);

    AbstractTensor decodeAbstract(byte[] bytes);
}
