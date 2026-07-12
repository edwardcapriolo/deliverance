package io.teknek.deliverance.guided;

import com.google.common.primitives.Ints;
import io.teknek.deliverance.generator.GeneratorParameters;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.sketches.guide.ChoiceGuide;
import io.teknek.sketches.guide.Guide;
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
        if (parameters.guidedChoice.isPresent()) {
            Choice choice = new Choice(parameters.guidedChoice.get());
            Guide guide = new ChoiceGuide(encodeChoices(model, choice), model.getConfig().eosTokens);
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
}
