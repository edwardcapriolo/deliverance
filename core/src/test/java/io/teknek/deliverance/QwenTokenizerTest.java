package io.teknek.deliverance;

import io.teknek.deliverance.model.ModelSupport;
import io.teknek.deliverance.model.ModelType;
import io.teknek.deliverance.model.llama.LlamaTokenizer;
import io.teknek.deliverance.model.qwen2.Qwen2ModelType;
import io.teknek.deliverance.tokenizer.Tokenizer;
import io.teknek.deliverance.tokenizer.TokenizerModel;
import org.apache.commons.lang3.stream.Streams;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;

import java.io.File;
import java.lang.reflect.InvocationTargetException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

public class QwenTokenizerTest {

    /*
try:
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
except:
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")
    print(tokenizer.encode(" hello sir. How are you Edward?", add_special_tokens=False))
    [23811, 27048, 13, 2585, 525, 498, 21891, 30]
     */
    @Disabled
    void encodeTest() throws NoSuchMethodException, InvocationTargetException, InstantiationException, IllegalAccessException {
        File config = new File("src/test/resources/qwen2_model_dir/config.json");
        ModelSupport.addModel("QWEN2", new Qwen2ModelType());
        ModelType detect = ModelSupport.detectModel(config);
        Tokenizer tokenizer = detect.getTokenizerClass().getConstructor(Path.class).newInstance(Paths.get(config.getParent()));
        Assertions.assertNotNull(detect);
        Assertions.assertEquals(LlamaTokenizer.class, tokenizer.getClass());
        long[] result = tokenizer.encode(" hello sir. How are you Edward?");
        System.out.println(Arrays.toString(result));
        //[49122, 2301, 404, 66003, 546, 9330, 84501, 30]
        List<String> back = new ArrayList<>();
        for (long l: result){
            back.add(tokenizer.decode(l));
        }

        System.out.println(back);
        Assertions.assertEquals(8, back.size());
        Assertions.assertArrayEquals(new long []{ 23811, 27048, 13, 2585, 525, 498, 21891, 30}, result);
    }

    @Disabled
    void decodeTest() throws NoSuchMethodException, InvocationTargetException, InstantiationException, IllegalAccessException {
        long [] tk = new long []{ 23811, 27048, 13, 2585, 525, 498, 21891, 30};
        List<String> converted = new ArrayList<>();

        File config = new File("src/test/resources/qwen2_model_dir/config.json");
        ModelSupport.addModel("QWEN2", new Qwen2ModelType());
        ModelType detect = ModelSupport.detectModel(config);
        LlamaTokenizer tokenizer = (LlamaTokenizer) detect.getTokenizerClass().getConstructor(Path.class).newInstance(Paths.get(config.getParent()));
        for (long l : tk){
            converted.add(tokenizer.decode(l));
        }
        Assertions.assertEquals("bla", converted);
    }
}
