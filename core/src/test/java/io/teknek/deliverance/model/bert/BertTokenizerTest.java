package io.teknek.deliverance.model.bert;

import io.teknek.deliverance.safetensors.fetch.ModelFetcher;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.io.File;

public class BertTokenizerTest {
    @Test
    void tokenize(){
        ModelFetcher fetch = new ModelFetcher("google-bert", "bert-base-cased");
        File f = fetch.maybeDownload();
        BertTokenizer bt = new BertTokenizer(f.toPath());
        //https://github.com/huggingface/transformers/blob/83fe012d58528a14ee0bb0146885f2d6fcb1ec3f/tests/tokenization/test_tokenization_utils.py#L129
        //from pretrained does not include leading space here
        Assertions.assertEquals(" Force", bt.decode(2300));
        Assertions.assertEquals(" [PAD]", bt.decode(0));
        int last = bt.getModel().vocabLookup.size() - 1;
        System.out.println(bt.getModel().vocabLookup);
        //Assertions.assertEquals("##:", bt.decode(last));
    }
}
