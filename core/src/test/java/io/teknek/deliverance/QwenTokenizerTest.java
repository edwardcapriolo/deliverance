package io.teknek.deliverance;

import io.teknek.deliverance.math.ActivationFunction;
import io.teknek.deliverance.model.ModelSupport;
import io.teknek.deliverance.model.ModelType;
import io.teknek.deliverance.model.llama.LlamaTokenizer;
import io.teknek.deliverance.model.qwen2.Qwen2Config;
import io.teknek.deliverance.model.qwen2.Qwen2ModelType;
import io.teknek.deliverance.model.qwen2.Qwen2Tokenizer;
import io.teknek.deliverance.safetensors.Config;
import io.teknek.deliverance.tokenizer.Tokenizer;
import io.teknek.deliverance.tokenizer.TokenizerModel;
import org.apache.commons.lang3.stream.Streams;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;

import java.io.File;
import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

import static io.teknek.deliverance.JsonUtils.om;
import static org.junit.jupiter.api.Assertions.assertEquals;

public class QwenTokenizerTest {

    @Test
    void configMappingTest() throws NoSuchMethodException, InvocationTargetException, InstantiationException, IllegalAccessException, IOException {
        File configFile = new File("src/test/resources/qwen2_model_dir/config.json");
        ModelSupport.addModel("QWEN2", new Qwen2ModelType());
        ModelType modelType = ModelSupport.detectModel(configFile);
        Qwen2Config  configC = (Qwen2Config) om.readValue(configFile, modelType.getConfigClass());
        Tokenizer tokenizer = modelType.getTokenizerClass().getConstructor(Path.class)
                .newInstance(Paths.get(configFile.getParent()));
        assertEquals(151643, configC.bosToken);
        assertEquals(152064, configC.vocabularySize);
        assertEquals(ActivationFunction.Type.SILU, configC.activationFunction);
        assertEquals(List.of("Qwen2ForCausalLM"), configC.getArchitectures());
    }
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
        assertEquals(Qwen2Tokenizer.class, tokenizer.getClass());
        long[] result = tokenizer.encode(" hello sir. How are you Edward?");
        System.out.println(Arrays.toString(result));
        assertEquals("?", tokenizer.decode(30));

        int [] hfTest = {23811, 27048, 13, 2585, 525, 498, 21891, 30};
        for (int i = 0; i < hfTest.length; i++) {
            System.out.println(i +" "+hfTest[i] +" "+ tokenizer.decode(hfTest[i]));
        }
        //[49122, 2301, 404, 66003, 546, 9330, 84501, 30]
        List<String> back = new ArrayList<>();
        for (long l: result){
            back.add(tokenizer.decode(l));
        }

        System.out.println(back);
        assertEquals(8, back.size());
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
        assertEquals("bla", converted);
    }

    /**
     * tokenizer("Hello world")["input_ids"]
     *     [9707, 1879]
     *
     *     >>> tokenizer(" Hello world")["input_ids"]
     */
}
