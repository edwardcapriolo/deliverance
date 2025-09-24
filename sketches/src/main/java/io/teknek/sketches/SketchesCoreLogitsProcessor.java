package io.teknek.sketches;

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
public class SketchesCoreLogitsProcessor {

    private boolean isFirstToken;
    private Index index;

    //setup
    int batchSize;
    int vocabSize;
    List<Guide> guides = new ArrayList<>();

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
        for (int i=0;i< batchSize;i++){
            this.guides.add(new Guide(this.index));
        }
    }
}
