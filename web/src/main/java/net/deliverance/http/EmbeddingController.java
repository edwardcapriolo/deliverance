package net.deliverance.http;

import io.teknek.deliverance.embedding.PoolingType;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.CreateEmbeddingRequest;
import io.teknek.deliverance.model.CreateEmbeddingResponse;
import io.teknek.deliverance.model.Embedding;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.RestController;

import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;

@RestController
public class EmbeddingController {

    @Autowired
    private Map<MultiModelConfig, AbstractModel> models;

    public EmbeddingController(Map<MultiModelConfig, AbstractModel> models){
        this.models = models;
    }

    private Optional<Map.Entry<MultiModelConfig, AbstractModel>> findModel(String name){
        return models.entrySet().stream()
                .filter(x-> x.getKey().getModelName()
                        .equalsIgnoreCase(name)).findFirst();
    }

    @RequestMapping(method = RequestMethod.POST, value="/embeddings", produces =  { "application/json" }, consumes = { "application/json" })
    public CreateEmbeddingResponse createEmbedding(@RequestBody CreateEmbeddingRequest request){
        Optional<Map.Entry<MultiModelConfig, AbstractModel>> z = findModel(request.getModel().getString());
        if (z.isEmpty()){
            throw new RuntimeException("model not found " + request.getModel());
        }
        float[] result = z.get().getValue().embed(request.getInput().getString(), PoolingType.AVG);
        List<BigDecimal> resultAsB = new ArrayList<>();
        for (float f: result){
            resultAsB.add(new BigDecimal(f));
        }
        CreateEmbeddingResponse resp = new CreateEmbeddingResponse();
        Embedding e = new Embedding().index(0).embedding(resultAsB);
        resp.addDataItem(e);
        return resp;
    }
}
