package io.teknek.sketches;

import com.google.common.collect.BiMap;

import java.util.ArrayList;
import java.util.List;
interface Bitmask {}
interface TokenBitmaskAllocator {
    Bitmask allocateTokenBitmask(int vocabSize);
}
interface LogitsBiaser {

}

class Guide {
    protected Index index;
    public Guide(Index index) {
        this.index = index;
    }
}

interface Index {

}
class RegexIndex implements Index {
    private final String regex;
    private final BiMap vocab;

    public RegexIndex(String regex, BiMap vocab){
        this.regex = regex;
        this.vocab = vocab;
    }

    public String getRegex() {
        return regex;
    }

    public BiMap getVocab() {
        return vocab;
    }
}
public class SketchesCoreLogitsProcessor {

    private boolean isFirstToken;
    private Index index;

    //setup
    int batchSize;
    int vocabSize;
    List<Guide> guides = new ArrayList<>();
    //            self.allocate_token_bitmask = allocate_token_bitmask
    //            self.bias_logits = self._bias_logits_numpy

    SketchesCoreLogitsProcessor(Index index){
        this.isFirstToken = true;
        this.index = index;
    }
    public void reset(){
        isFirstToken = true;
    }

    public void setup(int batchSize, int vocabSize){
        this.batchSize = batchSize;
        this.vocabSize = vocabSize;

//        if self.tensor_library_name == "torch":
//        from outlines_core.kernels.torch import allocate_token_bitmask

//        self.allocate_token_bitmask = allocate_token_bitmask
//        self.bias_logits = self._bias_logits_torch

        for (int i=0;i< batchSize;i++){
            this.guides.add(new Guide(this.index));
        }

    }
}
