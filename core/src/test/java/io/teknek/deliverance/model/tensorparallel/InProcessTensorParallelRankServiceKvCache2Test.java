package io.teknek.deliverance.model.tensorparallel;

import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.KvBufferCache;
import io.teknek.deliverance.tensor.kv.KvCacheSession;
import org.junit.jupiter.api.Test;

import java.util.Optional;
import java.util.UUID;

import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.ArgumentMatchers.same;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertEquals;

class InProcessTensorParallelRankServiceKvCache2Test {

    @Test
    void kvCache2RankServiceUsesKvCacheSessionForPrefillAndDecode() {
        AbstractModel model = mock(AbstractModel.class);
        TensorParallelCollectives collectives = new SingleRankTensorParallelCollectives();
        KvCacheSession kvSession = mock(KvCacheSession.class);
        KvBufferCache.KvBuffer oldKvBuffer = mock(KvBufferCache.KvBuffer.class);
        AbstractTensor prefillOutput = mock(AbstractTensor.class);
        AbstractTensor decodeOutput = mock(AbstractTensor.class);
        UUID sessionId = UUID.randomUUID();
        int[] promptTokens = {1, 2, 3};

        when(model.usesKvCache2Generation()).thenReturn(true);
        when(model.getTensorParallelCollectives()).thenReturn(collectives);
        when(model.newKvCacheSession()).thenReturn(kvSession);
        when(model.newKvBuffer()).thenReturn(oldKvBuffer);
        when(model.batchForward(same(promptTokens), eq(0), same(kvSession))).thenReturn(prefillOutput);
        when(model.forward(eq(4), eq(promptTokens.length), same(kvSession), eq(Optional.empty())))
                .thenReturn(decodeOutput);

        InProcessTensorParallelRankService service = new InProcessTensorParallelRankService(model);

        service.batchForward(sessionId, promptTokens, 0);
        service.forward(sessionId, 4, promptTokens.length);

        verify(model).newKvCacheSession();
        verify(model, never()).newKvBuffer();
        verify(model).batchForward(same(promptTokens), eq(0), same(kvSession));
        verify(model).forward(eq(4), eq(promptTokens.length), same(kvSession), eq(Optional.empty()));
    }

    @Test
    void rankServiceRejectsNonKvCache2Model() {
        AbstractModel model = mock(AbstractModel.class);
        when(model.usesKvCache2Generation()).thenReturn(false);

        assertThrows(IllegalArgumentException.class, () -> new InProcessTensorParallelRankService(model));
    }
}
