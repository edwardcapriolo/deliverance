package io.teknek.deliverance.guided;

import com.google.common.primitives.Ints;
import io.teknek.deliverance.generator.GeneratorParameters;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.sketches.guide.ChoiceGuide;
import io.teknek.sketches.guide.Guide;
import io.teknek.sketches.guide.Index;
import io.teknek.sketches.guide.IndexGuide;
import io.teknek.sketches.guide.Vocabulary;
import io.teknek.sketches.types.Choice;

import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

public class LogitsProcessorFactory {

    private LogitsProcessorFactory() {
    }

    public static Optional<LogitsProcessor> create(AbstractModel model, GeneratorParameters parameters) {
        if (parameters.guidedChoice.isPresent() && parameters.guidedRegex.isPresent()) {
            throw new IllegalArgumentException("guidedChoice and guidedRegex can not both be set");
        }
        if (parameters.guidedChoice.isPresent()) {
            Choice choice = new Choice(parameters.guidedChoice.get());
            Guide guide = new ChoiceGuide(encodeChoices(model, choice), model.getConfig().eosTokens);
            return Optional.of(new GuideLogitsProcessor(guide));
        }
        if (parameters.guidedRegex.isPresent()) {
            Vocabulary vocabulary = vocabularyFromModel(model);
            Guide guide = new IndexGuide(new Index(parameters.guidedRegex.get(), vocabulary));
            return Optional.of(new GuideLogitsProcessor(guide));
        }
        return Optional.empty();
    }

    private static Map<String, List<Integer>> encodeChoices(AbstractModel model, Choice choice) {
        Map<String, List<Integer>> encoded = new LinkedHashMap<>();
        for (String item : choice.items()) {
            List<Integer> ids = Arrays.stream(model.encodeForRuntime(item))
                    .mapToInt(Ints::checkedCast)
                    .boxed()
                    .toList();
            encoded.put(item, ids);
        }
        return encoded;
    }

    private static Vocabulary vocabularyFromModel(AbstractModel model) {
        Vocabulary vocabulary = new Vocabulary(model.getConfig().eosTokens, Map.of());
        for (int tokenId = 0; tokenId < model.getConfig().vocabularySize; tokenId++) {
            if (model.getConfig().eosTokens.contains(tokenId)) {
                continue;
            }
            vocabulary.insert(model.decodeToken(tokenId), tokenId);
        }
        return vocabulary;
    }
}
