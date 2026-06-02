package io.teknek.sketches;

import com.google.common.collect.BiMap;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.sketches.types.ContextFreeGrammar;
import io.teknek.sketches.types.JsonSchema;
import io.teknek.sketches.types.Regex;
import io.teknek.sketches.types.Term;

import java.util.Map;
import java.util.function.Function;

abstract class SteerableModel {

    abstract BiMap<String, Long> vocabulary();
    abstract Object tokenizer();
    abstract String getEosToken();
    abstract long getEosTokenId();
    abstract String convertTokenToString(long token);
}
class Generator {

    Generator(SteerableModel model, Class c){

    }
}
abstract class BaseBackend {
    public abstract Object getRegexLogitsPocessor(String regex);


    ///     def get_regex_logits_processor(self, regex: str) -> LogitsProcessorType:
}

class OutlinesCoreBackend extends BaseBackend {
    private final SteerableModel steerableModel;
    OutlinesCoreBackend(SteerableModel model){
        this.steerableModel = model;
    }

    @Override
    public Object getRegexLogitsPocessor(String regex) {
        Index i = new RegexIndex(regex, steerableModel.vocabulary());

        return new SketchesCoreLogitsProcessor(i);
    }
}




class SteerableGenerator {
    //logits_processor: Optional[LogitsProcessorType]

    SteerableGenerator(SteerableModel model, Class c, Term t){
        //term = python_types_to_terms(output_type)
        if (t instanceof JsonSchema){
            throw new IllegalArgumentException("not yet implemented");
        } else if (t instanceof ContextFreeGrammar){
            throw new IllegalArgumentException("not yet implemented");
        } else if (t instanceof Regex){
            //get_regex_logits_processor
        }
    }
}
public class Sketches {



    /* deliverance looks somewhat like transformers and with grace we will get much closer. This
    interface uses the current model and its grace tokenizer.
     */
    public SteerableModel frommDeliverance(AbstractModel model){
        return null;
    }
}
