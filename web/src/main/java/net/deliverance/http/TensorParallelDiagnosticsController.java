package net.deliverance.http;

import io.teknek.deliverance.model.CausalLanguageModel;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import java.util.LinkedHashMap;
import java.util.Map;

@RestController
public class TensorParallelDiagnosticsController {
    private final Map<MultiModelConfig, CausalLanguageModel> causalLanguageModels;

    public TensorParallelDiagnosticsController(
            @Qualifier("causalLanguageModels") Map<MultiModelConfig, CausalLanguageModel> causalLanguageModels) {
        this.causalLanguageModels = causalLanguageModels;
    }

    @RequestMapping(method = RequestMethod.GET, value = "/tp/status", produces = {"application/json"})
    public Map<String, Object> status() {
        return tpDiagnostics(TensorParallelSpringCausalLanguageModel::diagnostics);
    }

    @RequestMapping(method = RequestMethod.GET, value = "/tp/gossip", produces = {"application/json"})
    public Map<String, Object> gossip() {
        return tpDiagnostics(TensorParallelSpringCausalLanguageModel::gossipDiagnostics);
    }

    @RequestMapping(method = RequestMethod.GET, value = "/tp/endpoints", produces = {"application/json"})
    public Map<String, Object> endpoints() {
        return tpDiagnostics(TensorParallelSpringCausalLanguageModel::endpointDiagnostics);
    }

    @RequestMapping(method = RequestMethod.POST, value = "/tp/assign", produces = {"application/json"})
    public Map<String, Object> assign(@RequestParam("nodeId") String nodeId, @RequestParam("rank") int rank) {
        return tpDiagnostics(model -> model.assignRank(nodeId, rank));
    }

    private Map<String, Object> tpDiagnostics(DiagnosticExtractor extractor) {
        Map<String, Object> result = new LinkedHashMap<>();
        for (Map.Entry<MultiModelConfig, CausalLanguageModel> entry : causalLanguageModels.entrySet()) {
            if (entry.getValue() instanceof TensorParallelSpringCausalLanguageModel tpModel) {
                result.put(entry.getKey().getModelName(), extractor.extract(tpModel));
            }
        }
        result.put("tensorParallelModels", result.size());
        return result;
    }

    private interface DiagnosticExtractor {
        Map<String, Object> extract(TensorParallelSpringCausalLanguageModel model);
    }
}
