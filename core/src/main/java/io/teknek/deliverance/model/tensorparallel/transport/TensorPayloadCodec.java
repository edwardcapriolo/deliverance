package io.teknek.deliverance.model.tensorparallel.transport;

import io.teknek.deliverance.tensor.AbstractTensor;

public interface TensorPayloadCodec {
    String contentType();

    byte[] encode(AbstractTensor tensor);

    AbstractTensor decode(byte[] bytes);
}
