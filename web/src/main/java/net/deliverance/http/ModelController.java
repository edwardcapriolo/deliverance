package net.deliverance.http;

import io.teknek.deliverance.model.*;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.RestController;

import java.util.Map;

@RestController
public class ModelController {

    @Autowired
    private Map<MultiModelConfig, CausalLanguageModel> causalLanguageModels;

    private Map<MultiModelConfig, AbstractModel> embeddingModels;

    public ModelController(@Qualifier("causalLanguageModels") Map<MultiModelConfig, CausalLanguageModel> causalLanguageModels,
            @Qualifier("embeddingModels") Map<MultiModelConfig, AbstractModel> embeddingModels){
        this.causalLanguageModels = causalLanguageModels;
        this.embeddingModels = embeddingModels;
    }

    @RequestMapping(method = RequestMethod.GET, value="/models", produces =  { "application/json" },
            consumes = { "application/json" })
    public ListModelsResponse listModels(){
        ListModelsResponse response = new ListModelsResponse();
        response._object(ListModelsResponse.ObjectEnum.LIST);
        for (Map.Entry<MultiModelConfig, CausalLanguageModel> model : causalLanguageModels.entrySet()){
            response.addDataItem(new Model()
                    .ownedBy(model.getKey().getModelOwner())
                    .created(0)
                    ._object(Model.ObjectEnum.MODEL)
                    .id(model.getKey().getModelName()));
        }
        for (Map.Entry<MultiModelConfig, AbstractModel> model : embeddingModels.entrySet()){
            response.addDataItem(new Model()
                    .ownedBy(model.getKey().getModelOwner())
                    .created(0)
                    ._object(Model.ObjectEnum.MODEL)
                    .id(model.getKey().getModelName()));
        }
        return response;
    }
}
