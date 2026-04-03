package io.teknek.deliverance.integration;

import com.codahale.metrics.MetricRegistry;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.JsonUtils;
import io.teknek.deliverance.generator.GeneratorParameters;
import io.teknek.deliverance.generator.Response;
import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.model.*;
import io.teknek.deliverance.safetensors.fetch.ModelFetcher;
import io.teknek.deliverance.safetensors.prompt.PromptSupport;
import io.teknek.deliverance.tensor.KvBufferCacheSettings;
import io.teknek.deliverance.tensor.TensorCache;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;
import io.teknek.deliverance.tensor.operations.NativeSimdTensorOperations;
import io.teknek.deliverance.toolcallparser.DefaultToolCallParser;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.UUID;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class GemmaPromptIT {


    @Disabled
    public void summarizeGemmaTest() throws IOException {
        ModelFetcher fetch = new ModelFetcher("tjake", "gemma-2-2b-it-JQ4");
        File f = fetch.maybeDownload();
        MetricRegistry mr = new MetricRegistry();
        TensorCache tensorCache = new TensorCache(mr);
        WrappedForkJoinPool pool = new WrappedForkJoinPool(WrappedForkJoinPool.autoSizeByCores());
        NativeSimdTensorOperations operation = new NativeSimdTensorOperations(new ConfigurableTensorProvider(tensorCache, pool).get());
        try (AbstractModel m = ModelSupport.loadModel(f, DType.F32, DType.I8, new ConfigurableTensorProvider(operation),
                mr, tensorCache, new KvBufferCacheSettings(true), fetch, new NoOpTokenizerRenderer(), new DefaultToolCallParser(), pool)) {
            String prompt = """
                    You are a software engineer.
                    
                    ### INSTRUCTIONS ###
                    *   Your task is to write a complete, correct, and production-ready Java code.
                    *   Do not include any explanations, comments, or surrounding text, only the code block.
                    *   You MUST use the "java" markdown code block format.
                    
                    ### CODE ###
                    Implement the method:
                    public static float cosineSimilarity(float[] a, float[] b)
                    """;
            PromptSupport.Builder g = m.promptSupport().get().builder()
                    .addUserMessage(prompt);

            var uuid = UUID.randomUUID();

            Response k = m.generate(uuid, g.build(), new GeneratorParameters().withNtokens(8192).withTemperature(0.0f),
                    new GenerateEvent() {
                        @Override
                        public void emit(int next, String nextRaw, String nextCleaned, float timing) {
                            System.out.println(nextCleaned);
                        }
                    });
            System.out.println(k.responseText);

        }

    }

    //    return np.full(
    //        (1, (vocab_size + 31) // 32),
    //        -1,
    //        dtype=np.int32,
    //    )
    @Test
    public void chat() {

        AbstractModel model = Gemma2Suite.getOrCreate();
        String prompt = """
                What does this python code do?
                ---------------------------
                def allocate_token_bitmask(vocab_size: int) -> np.ndarray:
                    return np.full(
                        (1, (vocab_size + 31) // 32),
                        -1,
                        dtype=np.int32,
                    )
                """;
        PromptSupport.Builder g = model.promptSupport().get().builder().addUserMessage(prompt);
        var uuid = UUID.randomUUID();

        Response k = model.generate(uuid, g.build(), new GeneratorParameters().withMaxTokens(100).withTemperature(0.0f),
               new DoNothingGenerateEvent());
        String expected = """
Let's break down this Python code snippet.

**Understanding the Code**

This code defines a function called `allocate_token_bitmask` that generates a bitmask for a vocabulary.  Here's a step-by-step explanation:

1. **Function Definition:**
   - `def allocate_token_bitmask(vocab_size: int) -> np.ndarray:`
     - This line defines a function named `allocate_token_bitmask`. """.trim();
        assertEquals(expected, k.responseText.trim());

    }

    @Test
    public void gemmaTest() throws IOException {
        AbstractModel m = Gemma2Suite.getOrCreate();
        MetricRegistry mr = Gemma2Suite.getBuilder().getMr();
        String prompt = "What is the capital of New York, USA?";
        PromptSupport.Builder g = m.promptSupport().get().builder()
                .addUserMessage(prompt);
        assertEquals("<start_of_turn>user\n" +
                "What is the capital of New York, USA?<end_of_turn>\n" +
                "<start_of_turn>model\n", g.build().getPrompt());
        var uuid = UUID.randomUUID();

        Response k = m.generate(uuid, g.build(), new GeneratorParameters().withTemperature(0.0f),
                new DoNothingGenerateEvent());
        System.out.println(k.responseText);
        assertTrue(k.responseText.contains("Albany"));

        System.out.println(Arrays.toString(mr.histogram("sample.fullsample").getSnapshot().getValues()));
        System.out.println(mr.histogram("sample.fullsample").getSnapshot().getMean());
        System.out.println(mr.histogram("sample.fullsample").getSnapshot().get99thPercentile());

        System.out.println(Arrays.toString(mr.histogram("sample.forward1").getSnapshot().getValues()));
        System.out.println(mr.histogram("sample.forward1").getSnapshot().getMean());
        System.out.println(mr.histogram("sample.forward1").getSnapshot().get99thPercentile());

        System.out.println(Arrays.toString(mr.histogram("sample.dotproduct2").getSnapshot().getValues()));
        System.out.println(mr.histogram("sample.dotproduct2").getSnapshot().getMean());
        System.out.println(mr.histogram("sample.dotproduct2").getSnapshot().get99thPercentile());

    }

    @Test
    public void gemmaGuidedTest() {
        AbstractModel m = Gemma2Suite.getOrCreate();
        String prompt = "Who is the better NFL football team?";
        PromptSupport.Builder g = m.promptSupport().get().builder()
                .addUserMessage(prompt);
        assertEquals("<start_of_turn>user\n" +
                "Who is the better NFL football team?<end_of_turn>\n" +
                "<start_of_turn>model\n", g.build().getPrompt());
        var uuid = UUID.randomUUID();
        Response k = m.generate(uuid, g.build(), new GeneratorParameters()
                        .withTemperature(0.0f)
                        .withGuidedChoice(List.of("Giants", "Jets")),
                new DoNothingGenerateEvent());
        System.out.println(k.responseText);
        assertTrue(k.responseText.contains("Giants"));
    }

    @Test
    public void gemmaGuidedTestNeg() throws IOException {
        AbstractModel m = Gemma2Suite.getOrCreate();
        String prompt = "Which NFL franchise does not play in New York?";
        PromptSupport.Builder g = m.promptSupport().get().builder()
                .addUserMessage(prompt);
        var uuid = UUID.randomUUID();

        Response k = m.generate(uuid, g.build(), new GeneratorParameters()
                        .withTemperature(0.0f)
                        .withGuidedChoice(List.of("Giants", "Jets", "Seahawks")),
               new DoNothingGenerateEvent());
        assertTrue(k.responseText.contains("Seahawks"));
    }

    /*
    [SamplerReturn{token=235285, topNLogProbs=Optional[[IndexValueToken{index=235324, value=15.803696, token='7', logProb=-3.894785}, IndexValueToken{index=235322, value=15.863689, token='<', logProb=-3.8347912}, IndexValueToken{index=235284, value=16.127287, token='2', logProb=-3.5711937}, IndexValueToken{index=235310, value=16.025269, token='4', logProb=-3.673212}, IndexValueToken{index=235274, value=15.911542, token='1', logProb=-3.7869387}, IndexValueToken{index=235308, value=16.46085, token='5', logProb=-3.2376308}, IndexValueToken{index=235285, value=18.862904, token='I', logProb=-0.835577}, IndexValueToken{index=235304, value=16.25747, token='3', logProb=-3.4410114}, IndexValueToken{index=2926, value=17.083061, token='My', logProb=-2.6154194}, IndexValueToken{index=14692, value=17.353512, token='Okay', logProb=-2.3449688}]]}, SamplerReturn{token=15532, topNLogProbs=Optional[[IndexValueToken{index=877, value=14.774166, token=' will', logProb=-5.113823}, IndexValueToken{index=1317, value=14.787314, token=' just', logProb=-5.1006746}, IndexValueToken{index=1144, value=15.152139, token=' am', logProb=-4.7358503}, IndexValueToken{index=36011, value=17.24628, token=' randomly', logProb=-2.6417084}, IndexValueToken{index=1742, value=14.848688, token=' think', logProb=-5.039301}, IndexValueToken{index=6475, value=15.349413, token=' choose', logProb=-4.538576}, IndexValueToken{index=791, value=18.36088, token=' have', logProb=-1.5271091}, IndexValueToken{index=235303, value=18.40707, token=''', logProb=-1.4809189}, IndexValueToken{index=15532, value=18.942417, token=' picked', logProb=-0.9455719}, IndexValueToken{index=13670, value=16.427366, token=' chose', logProb=-3.4606228}]]}, SamplerReturn{token=573, topNLogProbs=Optional[[IndexValueToken{index=736, value=12.627438, token=' this', logProb=-8.475925}, IndexValueToken{index=6253, value=13.067088, token=' random', logProb=-8.036275}, IndexValueToken{index=1758, value=14.603782, token=' number', logProb=-6.4995813}, IndexValueToken{index=955, value=14.314642, token='...', logProb=-6.788721}, IndexValueToken{index=235292, value=14.791648, token=':', logProb=-6.311715}, IndexValueToken{index=573, value=21.075577, token=' the', logProb=-0.027786255}, IndexValueToken{index=235248, value=14.945629, token=' ', logProb=-6.157734}, IndexValueToken{index=5231, value=15.401367, token=' **', logProb=-5.701996}, IndexValueToken{index=139, value=14.51151, token='  ', logProb=-6.591853}, IndexValueToken{index=476, value=16.901316, token=' a', logProb=-4.2020473}]]}, SamplerReturn{token=1758, topNLogProbs=Optional[[IndexValueToken{index=36011, value=10.948022, token=' randomly', logProb=-10.386384}, IndexValueToken{index=1700, value=10.997557, token=' #', logProb=-10.336849}, IndexValueToken{index=5968, value=11.159005, token=' numbers', logProb=-10.175401}, IndexValueToken{index=2412, value=11.354157, token=' following', logProb=-9.980248}, IndexValueToken{index=235248, value=11.102899, token=' ', logProb=-10.231507}, IndexValueToken{index=5231, value=12.701079, token=' **', logProb=-8.633327}, IndexValueToken{index=1758, value=21.320602, token=' number', logProb=-0.013803482}, IndexValueToken{index=139, value=12.433427, token='  ', logProb=-8.900979}, IndexValueToken{index=6165, value=11.356123, token=' Number', logProb=-9.978283}, IndexValueToken{index=6253, value=16.999119, token=' random', logProb=-4.335287}]]}, SamplerReturn{token=5231, topNLogProbs=Optional[[IndexValueToken{index=649, value=10.28941, token=' *', logProb=-9.200556}, IndexValueToken{index=19190, value=10.892544, token=' ***', logProb=-8.597422}, IndexValueToken{index=968, value=11.089134, token=' <', logProb=-8.400831}, IndexValueToken{index=2804, value=11.197407, token=' ...', logProb=-8.292559}, IndexValueToken{index=235269, value=11.580621, token=',', logProb=-7.9093447}, IndexValueToken{index=955, value=13.206521, token='...', logProb=-6.2834444}, IndexValueToken{index=235292, value=16.540133, token=':', logProb=-2.949833}, IndexValueToken{index=5231, value=18.991808, token=' **', logProb=-0.4981575}, IndexValueToken{index=235248, value=18.357092, token=' ', logProb=-1.1328735}, IndexValueToken{index=139, value=15.249001, token='  ', logProb=-4.240965}]]}, SamplerReturn{token=235308, topNLogProbs=Optional[[IndexValueToken{index=9295, value=12.151476, token='random', logProb=-9.054129}, IndexValueToken{index=235321, value=15.351205, token='8', logProb=-5.8543997}, IndexValueToken{index=235274, value=12.902421, token='1', logProb=-8.303184}, IndexValueToken{index=235284, value=17.47865, token='2', logProb=-3.7269554}, IndexValueToken{index=235324, value=19.661781, token='7', logProb=-1.5438232}, IndexValueToken{index=235248, value=14.019234, token=' ', logProb=-7.186371}, IndexValueToken{index=235304, value=19.329502, token='3', logProb=-1.8761024}, IndexValueToken{index=235318, value=18.46304, token='6', logProb=-2.7425652}, IndexValueToken{index=235310, value=19.255108, token='4', logProb=-1.9504967}, IndexValueToken{index=235308, value=20.285082, token='5', logProb=-0.9205227}]]}, SamplerReturn{token=688, topNLogProbs=Optional[[IndexValueToken{index=235265, value=7.7079096, token='.', logProb=-12.574187}, IndexValueToken{index=235318, value=7.8388195, token='6', logProb=-12.443277}, IndexValueToken{index=5231, value=10.221854, token=' **', logProb=-10.060243}, IndexValueToken{index=775, value=9.157878, token='/**', logProb=-11.124219}, IndexValueToken{index=235341, value=13.947628, token='!', logProb=-6.334469}, IndexValueToken{index=168428, value=19.526667, token='**.', logProb=-0.7554302}, IndexValueToken{index=116742, value=12.780639, token='.**', logProb=-7.501458}, IndexValueToken{index=190213, value=10.587126, token='**,', logProb=-9.694971}, IndexValueToken{index=95573, value=11.255058, token='**:', logProb=-9.027039}, IndexValueToken{index=688, value=19.642668, token='**', logProb=-0.6394291}]]}, SamplerReturn{token=235341, topNLogProbs=Optional[[IndexValueToken{index=7221, value=14.427841, token='.</', logProb=-5.274721}, IndexValueToken{index=123781, value=14.508775, token=' 😄', logProb=-5.1937876}, IndexValueToken{index=51984, value=14.561884, token=' 😉', logProb=-5.1406784}, IndexValueToken{index=160588, value=14.652362, token=' 🎉', logProb=-5.0502005}, IndexValueToken{index=139, value=16.78816, token='  ', logProb=-2.914402}, IndexValueToken{index=109, value=15.901237, token='

', logProb=-3.8013258}, IndexValueToken{index=235341, value=18.97498, token='!', logProb=-0.72758293}, IndexValueToken{index=954, value=16.350082, token=' .', logProb=-3.35248}, IndexValueToken{index=44416, value=15.720037, token=' 😊', logProb=-3.9825249}, IndexValueToken{index=235248, value=18.566132, token=' ', logProb=-1.1364307}]]}, SamplerReturn{token=235248, topNLogProbs=Optional[[IndexValueToken{index=33359, value=15.344488, token=' 🙂', logProb=-5.3785305}, IndexValueToken{index=160588, value=15.954437, token=' 🎉', logProb=-4.7685814}, IndexValueToken{index=109, value=15.856832, token='

', logProb=-4.866187}, IndexValueToken{index=968, value=16.162943, token=' <', logProb=-4.5600758}, IndexValueToken{index=139, value=19.451315, token='  ', logProb=-1.2717037}, IndexValueToken{index=70636, value=16.348415, token=' 😁', logProb=-4.3746033}, IndexValueToken{index=44416, value=18.417412, token=' 😊', logProb=-2.3056068}, IndexValueToken{index=123781, value=17.210155, token=' 😄', logProb=-3.5128632}, IndexValueToken{index=51984, value=16.371353, token=' 😉', logProb=-4.3516655}, IndexValueToken{index=235248, value=20.020939, token=' ', logProb=-0.7020798}]]}, SamplerReturn{token=245539, topNLogProbs=Optional[[IndexValueToken{index=250786, value=14.366234, token='🧮', logProb=-6.698715}, IndexValueToken{index=240802, value=14.546617, token='💫', logProb=-6.5183325}, IndexValueToken{index=109, value=18.884811, token='

', logProb=-2.1801376}, IndexValueToken{index=247166, value=15.42326, token='🃏', logProb=-5.6416893}, IndexValueToken{index=243308, value=14.716934, token='🎤', logProb=-6.348015}, IndexValueToken{index=108, value=19.826292, token='
', logProb=-1.238657}, IndexValueToken{index=245539, value=20.418488, token='🎲', logProb=-0.6464615}, IndexValueToken{index=110, value=17.436266, token='


', logProb=-3.628683}, IndexValueToken{index=241177, value=16.586096, token='😜', logProb=-4.478853}, IndexValueToken{index=111, value=16.41035, token='



', logProb=-4.654598}]]}, SamplerReturn{token=235248, topNLogProbs=Optional[[IndexValueToken{index=2692, value=15.415576, token=' </', logProb=-5.8365984}, IndexValueToken{index=123781, value=15.537116, token=' 😄', logProb=-5.7150583}, IndexValueToken{index=110, value=15.508643, token='


', logProb=-5.743531}, IndexValueToken{index=107, value=16.236305, token='<end_of_turn>', logProb=-5.015869}, IndexValueToken{index=237698, value=16.355968, token='😊', logProb=-4.896206}, IndexValueToken{index=44416, value=17.080235, token=' 😊', logProb=-4.17194}, IndexValueToken{index=139, value=19.721338, token='  ', logProb=-1.5308361}, IndexValueToken{index=109, value=16.541315, token='

', logProb=-4.7108593}, IndexValueToken{index=108, value=17.81848, token='
', logProb=-3.4336948}, IndexValueToken{index=235248, value=20.860172, token=' ', logProb=-0.3920021}]]}, SamplerReturn{token=108, topNLogProbs=Optional[[IndexValueToken{index=247556, value=11.961187, token='🔢', logProb=-9.149311}, IndexValueToken{index=241177, value=12.344558, token='😜', logProb=-8.765941}, IndexValueToken{index=107, value=12.186972, token='<end_of_turn>', logProb=-8.923527}, IndexValueToken{index=112, value=13.147385, token='




', logProb=-7.963114}, IndexValueToken{index=240802, value=13.157052, token='💫', logProb=-7.9534464}, IndexValueToken{index=110, value=17.927673, token='


', logProb=-3.182825}, IndexValueToken{index=111, value=16.293896, token='



', logProb=-4.8166027}, IndexValueToken{index=235248, value=13.537128, token=' ', logProb=-7.57337}, IndexValueToken{index=109, value=19.188066, token='

', logProb=-1.922432}, IndexValueToken{index=108, value=20.88846, token='
', logProb=-0.22203827}]]}, SamplerReturn{token=235322, topNLogProbs=Optional[[IndexValueToken{index=2692, value=13.924648, token=' </', logProb=-6.3978863}, IndexValueToken{index=235286, value=14.049848, token='\', logProb=-6.272687}, IndexValueToken{index=235248, value=16.654255, token=' ', logProb=-3.6682796}, IndexValueToken{index=1841, value=14.488721, token='What', logProb=-5.8338137}, IndexValueToken{index=18925, value=14.213005, token='Would', logProb=-6.1095295}, IndexValueToken{index=727, value=17.733097, token='</', logProb=-2.5894375}, IndexValueToken{index=235322, value=20.09787, token='<', logProb=-0.22466469}, IndexValueToken{index=968, value=15.083164, token=' <', logProb=-5.2393703}, IndexValueToken{index=107, value=17.699715, token='<end_of_turn>', logProb=-2.62282}, IndexValueToken{index=139, value=15.6108055, token='  ', logProb=-4.711729}]]}, SamplerReturn{token=615, topNLogProbs=Optional[[IndexValueToken{index=3589, value=6.6558695, token='current', logProb=-11.3235655}, IndexValueToken{index=1432, value=7.039555, token='span', logProb=-10.93988}, IndexValueToken{index=1598, value=7.0783772, token='br', logProb=-10.901058}, IndexValueToken{index=108, value=7.126996, token='
', logProb=-10.852439}, IndexValueToken{index=3310, value=7.316609, token='next', logProb=-10.662827}, IndexValueToken{index=615, value=17.948559, token='end', logProb=-0.03087616}, IndexValueToken{index=1387, value=8.87362, token='center', logProb=-9.105815}, IndexValueToken{index=1580, value=7.460183, token=' end', logProb=-10.519252}, IndexValueToken{index=2997, value=14.46263, token='start', logProb=-3.5168047}, IndexValueToken{index=13072, value=7.8935595, token='answer', logProb=-10.0858755}]]}, SamplerReturn{token=235298, topNLogProbs=Optional[[IndexValueToken{index=235293, value=3.8237474, token='=', logProb=-18.562899}, IndexValueToken{index=107635, value=4.0355287, token='_<', logProb=-18.351118}, IndexValueToken{index=75076, value=3.8872836, token='_*', logProb=-18.499363}, IndexValueToken{index=235290, value=5.256213, token='-', logProb=-17.130432}, IndexValueToken{index=235313, value=5.541416, token='>', logProb=-16.84523}, IndexValueToken{index=576, value=10.11526, token=' of', logProb=-12.271386}, IndexValueToken{index=1762, value=7.3867855, token=' _', logProb=-14.999861}, IndexValueToken{index=559, value=5.7151093, token='of', logProb=-16.671537}, IndexValueToken{index=5801, value=10.811387, token='\_', logProb=-11.575259}, IndexValueToken{index=235298, value=22.386631, token='_', logProb=-1.5258789E-5}]]}, SamplerReturn{token=559, topNLogProbs=Optional[[IndexValueToken{index=15508, value=10.369719, token='turn', logProb=-15.680775}, IndexValueToken{index=2997, value=10.716382, token='start', logProb=-15.334111}, IndexValueToken{index=3179, value=11.302419, token='Of', logProb=-14.748075}, IndexValueToken{index=10728, value=11.116491, token='OF', logProb=-14.934002}, IndexValueToken{index=79795, value=10.747114, token='ofthe', logProb=-15.303379}, IndexValueToken{index=12457, value=13.233422, token='your', logProb=-12.817071}, IndexValueToken{index=559, value=26.050415, token='of', logProb=-7.8201294E-5}, IndexValueToken{index=2672, value=11.893731, token='off', logProb=-14.156762}, IndexValueToken{index=235253, value=12.393969, token='o', logProb=-13.656525}, IndexValueToken{index=576, value=16.505135, token=' of', logProb=-9.545359}]]}, SamplerReturn{token=235298, topNLogProbs=Optional[[IndexValueToken{index=235293, value=5.1134872, token='=', logProb=-16.041588}, IndexValueToken{index=235313, value=6.167048, token='>', logProb=-14.988028}, IndexValueToken{index=107635, value=5.385942, token='_<', logProb=-15.769133}, IndexValueToken{index=15508, value=6.5135894, token='turn', logProb=-14.641485}, IndexValueToken{index=235290, value=7.378484, token='-', logProb=-13.776591}, IndexValueToken{index=5801, value=10.262447, token='\_', logProb=-10.892628}, IndexValueToken{index=905, value=6.4358773, token='__', logProb=-14.719198}, IndexValueToken{index=2894, value=9.978697, token=' turn', logProb=-11.176378}, IndexValueToken{index=1762, value=6.992061, token=' _', logProb=-14.163013}, IndexValueToken{index=235298, value=21.155039, token='_', logProb=-3.6239624E-5}]]}, SamplerReturn{token=15508, topNLogProbs=Optional[[IndexValueToken{index=9399, value=9.644975, token='take', logProb=-13.196729}, IndexValueToken{index=773, value=9.894059, token='return', logProb=-12.947644}, IndexValueToken{index=12457, value=11.047995, token='your', logProb=-11.793709}, IndexValueToken{index=19052, value=10.500204, token='tune', logProb=-12.341499}, IndexValueToken{index=66787, value=10.491578, token=' turno', logProb=-12.350125}, IndexValueToken{index=1201, value=15.757963, token='round', logProb=-7.08374}, IndexValueToken{index=15508, value=22.840807, token='turn', logProb=-8.9645386E-4}, IndexValueToken{index=7617, value=10.876396, token='term', logProb=-11.965307}, IndexValueToken{index=690, value=10.661129, token='urn', logProb=-12.180574}, IndexValueToken{index=26059, value=11.29998, token='tour', logProb=-11.541723}]]}, SamplerReturn{token=235313, topNLogProbs=Optional[[IndexValueToken{index=16259, value=5.961116, token='>&', logProb=-14.826608}, IndexValueToken{index=235248, value=6.161161, token=' ', logProb=-14.626562}, IndexValueToken{index=32229, value=6.2910533, token='>.', logProb=-14.49667}, IndexValueToken{index=235298, value=7.2955647, token='_', logProb=-13.492159}, IndexValueToken{index=2577, value=10.25299, token='><', logProb=-10.534734}, IndexValueToken{index=30561, value=6.753545, token='>'', logProb=-14.034179}, IndexValueToken{index=14648, value=9.287034, token='>\', logProb=-11.5006895}, IndexValueToken{index=82305, value=7.4454193, token='>*', logProb=-13.342304}, IndexValueToken{index=65529, value=7.399906, token='>`', logProb=-13.387817}, IndexValueToken{index=235313, value=20.787672, token='>', logProb=-5.1498413E-5}]]}, SamplerReturn{token=235248, topNLogProbs=Optional[[IndexValueToken{index=140, value=11.570394, token='   ', logProb=-9.242478}, IndexValueToken{index=1, value=13.599644, token='<eos>', logProb=-7.213228}, IndexValueToken{index=112, value=12.868686, token='




', logProb=-7.944186}, IndexValueToken{index=111, value=15.253961, token='



', logProb=-5.5589113}, IndexValueToken{index=139, value=14.779063, token='  ', logProb=-6.0338087}, IndexValueToken{index=110, value=16.241657, token='


', logProb=-4.5712147}, IndexValueToken{index=109, value=16.666002, token='

', logProb=-4.1468697}, IndexValueToken{index=235248, value=20.225313, token=' ', logProb=-0.58755875}, IndexValueToken{index=108, value=18.124666, token='
', logProb=-2.6882057}, IndexValueToken{index=107, value=19.74028, token='<end_of_turn>', logProb=-1.0725918}]]}, SamplerReturn{token=108, topNLogProbs=Optional[[IndexValueToken{index=115, value=9.885976, token='







', logProb=-11.252725}, IndexValueToken{index=114, value=10.298255, token='






', logProb=-10.8404455}, IndexValueToken{index=112, value=12.081442, token='', logProb=-9.057259}, IndexValueToken{index=1, value=12.321919, token='<eos>', logProb=-8.816781}, IndexValueToken{index=113, value=10.7750845, token='', logProb=-10.363616}, IndexValueToken{index=108, value=21.101353, token='', logProb=-0.037347794}, IndexValueToken{index=109, value=17.269321, token='', logProb=-3.869379}, IndexValueToken{index=110, value=16.778517, token='', logProb=-4.3601837}, IndexValueToken{index=107, value=14.806606, token='<end_of_turn>', logProb=-6.332094}, IndexValueToken{index=111, value=14.089223, token='



', logProb=-7.0494776}]]}, SamplerReturn{token=107, topNLogProbs=Optional[[IndexValueToken{index=18925, value=12.618357, token='Would', logProb=-9.445505}, IndexValueToken{index=235287, value=12.772391, token='*', logProb=-9.291471}, IndexValueToken{index=688, value=12.754199, token='**', logProb=-9.309663}, IndexValueToken{index=1, value=13.899034, token='<eos>', logProb=-8.164828}, IndexValueToken{index=235322, value=16.065037, token='<', logProb=-5.998825}, IndexValueToken{index=1841, value=13.548776, token='What', logProb=-8.515086}, IndexValueToken{index=139, value=13.680857, token='  ', logProb=-8.383005}, IndexValueToken{index=5331, value=14.243997, token='Let', logProb=-7.819865}, IndexValueToken{index=107, value=22.053408, token='<end_of_turn>', logProb=-0.010454178}, IndexValueToken{index=235248, value=16.924097, token=' ', logProb=-5.139765}]]}]
     */


    /*
    [
    IndexValueToken{index=9295, value=12.151476, token='random', logProb=-9.054129},
    IndexValueToken{index=235321, value=15.351205, token='8', logProb=-5.8543997},
    IndexValueToken{index=235274, value=12.902421, token='1', logProb=-8.303184},
    IndexValueToken{index=235284, value=17.47865, token='2', logProb=-3.7269554},
    IndexValueToken{index=235324, value=19.661781, token='7', logProb=-1.5438232},
    IndexValueToken{index=235248, value=14.019234, token=' ', logProb=-7.186371},
    IndexValueToken{index=235304, value=19.329502, token='3', logProb=-1.8761024},
    IndexValueToken{index=235318, value=18.46304, token='6', logProb=-2.7425652},
    IndexValueToken{index=235310, value=19.255108, token='4', logProb=-1.9504967},
    IndexValueToken{index=235308, value=20.285082, token='5', logProb=-0.9205227}
    ]
5
     */
    @Test
    public void logProbs() throws JsonProcessingException {
        AbstractModel m = Gemma2Suite.getOrCreate();
        String prompt = "Pick a random number between 1 and 9. Replay with only the pick.";
        PromptSupport.Builder g = m.promptSupport().get().builder()
                .addUserMessage(prompt);
        var uuid = UUID.randomUUID();
        Response k = m.generate(uuid, g.build(), new GeneratorParameters()
                        .withTemperature(0.0f)
                        .withMaxTokens(300)
                        .withLogProbs(true)
                        .withTopLogProbs(10)
                , new GenerateEvent() {
                    @Override
                    public void emit(int next, String nextRaw, String nextCleaned, float timing) {
                        System.out.println(nextCleaned);
                    }
                });

        String expected = """
         [{"token":235285,"topNLogProbs":[{"index":235324,"value":15.803696,"token":"7","logProb":-3.894785},{"index":235322,"value":15.863689,"token":"<","logProb":-3.8347912},{"index":235284,"value":16.127287,"token":"2","logProb":-3.5711937},{"index":235310,"value":16.025269,"token":"4","logProb":-3.673212},{"index":235274,"value":15.911542,"token":"1","logProb":-3.7869387},{"index":235308,"value":16.46085,"token":"5","logProb":-3.2376308},{"index":235285,"value":18.862904,"token":"I","logProb":-0.835577},{"index":235304,"value":16.25747,"token":"3","logProb":-3.4410114},{"index":2926,"value":17.083061,"token":"My","logProb":-2.6154194},{"index":14692,"value":17.353512,"token":"Okay","logProb":-2.3449688}]},{"token":15532,"topNLogProbs":[{"index":877,"value":14.774166,"token":" will","logProb":-5.113823},{"index":1317,"value":14.787314,"token":" just","logProb":-5.1006746},{"index":1144,"value":15.152139,"token":" am","logProb":-4.7358503},{"index":36011,"value":17.24628,"token":" randomly","logProb":-2.6417084},{"index":1742,"value":14.848688,"token":" think","logProb":-5.039301},{"index":6475,"value":15.349413,"token":" choose","logProb":-4.538576},{"index":791,"value":18.36088,"token":" have","logProb":-1.5271091},{"index":235303,"value":18.40707,"token":"'","logProb":-1.4809189},{"index":15532,"value":18.942417,"token":" picked","logProb":-0.9455719},{"index":13670,"value":16.427366,"token":" chose","logProb":-3.4606228}]},{"token":573,"topNLogProbs":[{"index":736,"value":12.627438,"token":" this","logProb":-8.475925},{"index":6253,"value":13.067088,"token":" random","logProb":-8.036275},{"index":1758,"value":14.603782,"token":" number","logProb":-6.4995813},{"index":955,"value":14.314642,"token":"...","logProb":-6.788721},{"index":235292,"value":14.791648,"token":":","logProb":-6.311715},{"index":573,"value":21.075577,"token":" the","logProb":-0.027786255},{"index":235248,"value":14.945629,"token":" ","logProb":-6.157734},{"index":5231,"value":15.401367,"token":" **","logProb":-5.701996},{"index":139,"value":14.51151,"token":"  ","logProb":-6.591853},{"index":476,"value":16.901316,"token":" a","logProb":-4.2020473}]},{"token":1758,"topNLogProbs":[{"index":36011,"value":10.948022,"token":" randomly","logProb":-10.386384},{"index":1700,"value":10.997557,"token":" #","logProb":-10.336849},{"index":5968,"value":11.159005,"token":" numbers","logProb":-10.175401},{"index":2412,"value":11.354157,"token":" following","logProb":-9.980248},{"index":235248,"value":11.102899,"token":" ","logProb":-10.231507},{"index":5231,"value":12.701079,"token":" **","logProb":-8.633327},{"index":1758,"value":21.320602,"token":" number","logProb":-0.013803482},{"index":139,"value":12.433427,"token":"  ","logProb":-8.900979},{"index":6165,"value":11.356123,"token":" Number","logProb":-9.978283},{"index":6253,"value":16.999119,"token":" random","logProb":-4.335287}]},{"token":5231,"topNLogProbs":[{"index":649,"value":10.28941,"token":" *","logProb":-9.200556},{"index":19190,"value":10.892544,"token":" ***","logProb":-8.597422},{"index":968,"value":11.089134,"token":" <","logProb":-8.400831},{"index":2804,"value":11.197407,"token":" ...","logProb":-8.292559},{"index":235269,"value":11.580621,"token":",","logProb":-7.9093447},{"index":955,"value":13.206521,"token":"...","logProb":-6.2834444},{"index":235292,"value":16.540133,"token":":","logProb":-2.949833},{"index":5231,"value":18.991808,"token":" **","logProb":-0.4981575},{"index":235248,"value":18.357092,"token":" ","logProb":-1.1328735},{"index":139,"value":15.249001,"token":"  ","logProb":-4.240965}]},{"token":235308,"topNLogProbs":[{"index":9295,"value":12.151476,"token":"random","logProb":-9.054129},{"index":235321,"value":15.351205,"token":"8","logProb":-5.8543997},{"index":235274,"value":12.902421,"token":"1","logProb":-8.303184},{"index":235284,"value":17.47865,"token":"2","logProb":-3.7269554},{"index":235324,"value":19.661781,"token":"7","logProb":-1.5438232},{"index":235248,"value":14.019234,"token":" ","logProb":-7.186371},{"index":235304,"value":19.329502,"token":"3","logProb":-1.8761024},{"index":235318,"value":18.46304,"token":"6","logProb":-2.7425652},{"index":235310,"value":19.255108,"token":"4","logProb":-1.9504967},{"index":235308,"value":20.285082,"token":"5","logProb":-0.9205227}]},{"token":688,"topNLogProbs":[{"index":235265,"value":7.7079096,"token":".","logProb":-12.574187},{"index":235318,"value":7.8388195,"token":"6","logProb":-12.443277},{"index":5231,"value":10.221854,"token":" **","logProb":-10.060243},{"index":775,"value":9.157878,"token":"/**","logProb":-11.124219},{"index":235341,"value":13.947628,"token":"!","logProb":-6.334469},{"index":168428,"value":19.526667,"token":"**.","logProb":-0.7554302},{"index":116742,"value":12.780639,"token":".**","logProb":-7.501458},{"index":190213,"value":10.587126,"token":"**,","logProb":-9.694971},{"index":95573,"value":11.255058,"token":"**:","logProb":-9.027039},{"index":688,"value":19.642668,"token":"**","logProb":-0.6394291}]},{"token":235341,"topNLogProbs":[{"index":7221,"value":14.427841,"token":".</","logProb":-5.274721},{"index":123781,"value":14.508775,"token":" 😄","logProb":-5.1937876},{"index":51984,"value":14.561884,"token":" 😉","logProb":-5.1406784},{"index":160588,"value":14.652362,"token":" 🎉","logProb":-5.0502005},{"index":139,"value":16.78816,"token":"  ","logProb":-2.914402},{"index":109,"value":15.901237,"token":"\\n\\n","logProb":-3.8013258},{"index":235341,"value":18.97498,"token":"!","logProb":-0.72758293},{"index":954,"value":16.350082,"token":" .","logProb":-3.35248},{"index":44416,"value":15.720037,"token":" 😊","logProb":-3.9825249},{"index":235248,"value":18.566132,"token":" ","logProb":-1.1364307}]},{"token":235248,"topNLogProbs":[{"index":33359,"value":15.344488,"token":" 🙂","logProb":-5.3785305},{"index":160588,"value":15.954437,"token":" 🎉","logProb":-4.7685814},{"index":109,"value":15.856832,"token":"\\n\\n","logProb":-4.866187},{"index":968,"value":16.162943,"token":" <","logProb":-4.5600758},{"index":139,"value":19.451315,"token":"  ","logProb":-1.2717037},{"index":70636,"value":16.348415,"token":" 😁","logProb":-4.3746033},{"index":44416,"value":18.417412,"token":" 😊","logProb":-2.3056068},{"index":123781,"value":17.210155,"token":" 😄","logProb":-3.5128632},{"index":51984,"value":16.371353,"token":" 😉","logProb":-4.3516655},{"index":235248,"value":20.020939,"token":" ","logProb":-0.7020798}]},{"token":245539,"topNLogProbs":[{"index":250786,"value":14.366234,"token":"🧮","logProb":-6.698715},{"index":240802,"value":14.546617,"token":"💫","logProb":-6.5183325},{"index":109,"value":18.884811,"token":"\\n\\n","logProb":-2.1801376},{"index":247166,"value":15.42326,"token":"🃏","logProb":-5.6416893},{"index":243308,"value":14.716934,"token":"🎤","logProb":-6.348015},{"index":108,"value":19.826292,"token":"\\n","logProb":-1.238657},{"index":245539,"value":20.418488,"token":"🎲","logProb":-0.6464615},{"index":110,"value":17.436266,"token":"\\n\\n\\n","logProb":-3.628683},{"index":241177,"value":16.586096,"token":"😜","logProb":-4.478853},{"index":111,"value":16.41035,"token":"\\n\\n\\n\\n","logProb":-4.654598}]},{"token":235248,"topNLogProbs":[{"index":2692,"value":15.415576,"token":" </","logProb":-5.8365984},{"index":123781,"value":15.537116,"token":" 😄","logProb":-5.7150583},{"index":110,"value":15.508643,"token":"\\n\\n\\n","logProb":-5.743531},{"index":107,"value":16.236305,"token":"<end_of_turn>","logProb":-5.015869},{"index":237698,"value":16.355968,"token":"😊","logProb":-4.896206},{"index":44416,"value":17.080235,"token":" 😊","logProb":-4.17194},{"index":139,"value":19.721338,"token":"  ","logProb":-1.5308361},{"index":109,"value":16.541315,"token":"\\n\\n","logProb":-4.7108593},{"index":108,"value":17.81848,"token":"\\n","logProb":-3.4336948},{"index":235248,"value":20.860172,"token":" ","logProb":-0.3920021}]},{"token":108,"topNLogProbs":[{"index":247556,"value":11.961187,"token":"🔢","logProb":-9.149311},{"index":241177,"value":12.344558,"token":"😜","logProb":-8.765941},{"index":107,"value":12.186972,"token":"<end_of_turn>","logProb":-8.923527},{"index":112,"value":13.147385,"token":"\\n\\n\\n\\n\\n","logProb":-7.963114},{"index":240802,"value":13.157052,"token":"💫","logProb":-7.9534464},{"index":110,"value":17.927673,"token":"\\n\\n\\n","logProb":-3.182825},{"index":111,"value":16.293896,"token":"\\n\\n\\n\\n","logProb":-4.8166027},{"index":235248,"value":13.537128,"token":" ","logProb":-7.57337},{"index":109,"value":19.188066,"token":"\\n\\n","logProb":-1.922432},{"index":108,"value":20.88846,"token":"\\n","logProb":-0.22203827}]},{"token":235322,"topNLogProbs":[{"index":2692,"value":13.924648,"token":" </","logProb":-6.3978863},{"index":235286,"value":14.049848,"token":"\\\\","logProb":-6.272687},{"index":235248,"value":16.654255,"token":" ","logProb":-3.6682796},{"index":1841,"value":14.488721,"token":"What","logProb":-5.8338137},{"index":18925,"value":14.213005,"token":"Would","logProb":-6.1095295},{"index":727,"value":17.733097,"token":"</","logProb":-2.5894375},{"index":235322,"value":20.09787,"token":"<","logProb":-0.22466469},{"index":968,"value":15.083164,"token":" <","logProb":-5.2393703},{"index":107,"value":17.699715,"token":"<end_of_turn>","logProb":-2.62282},{"index":139,"value":15.6108055,"token":"  ","logProb":-4.711729}]},{"token":615,"topNLogProbs":[{"index":3589,"value":6.6558695,"token":"current","logProb":-11.3235655},{"index":1432,"value":7.039555,"token":"span","logProb":-10.93988},{"index":1598,"value":7.0783772,"token":"br","logProb":-10.901058},{"index":108,"value":7.126996,"token":"\\n","logProb":-10.852439},{"index":3310,"value":7.316609,"token":"next","logProb":-10.662827},{"index":615,"value":17.948559,"token":"end","logProb":-0.03087616},{"index":1387,"value":8.87362,"token":"center","logProb":-9.105815},{"index":1580,"value":7.460183,"token":" end","logProb":-10.519252},{"index":2997,"value":14.46263,"token":"start","logProb":-3.5168047},{"index":13072,"value":7.8935595,"token":"answer","logProb":-10.0858755}]},{"token":235298,"topNLogProbs":[{"index":235293,"value":3.8237474,"token":"=","logProb":-18.562899},{"index":107635,"value":4.0355287,"token":"_<","logProb":-18.351118},{"index":75076,"value":3.8872836,"token":"_*","logProb":-18.499363},{"index":235290,"value":5.256213,"token":"-","logProb":-17.130432},{"index":235313,"value":5.541416,"token":">","logProb":-16.84523},{"index":576,"value":10.11526,"token":" of","logProb":-12.271386},{"index":1762,"value":7.3867855,"token":" _","logProb":-14.999861},{"index":559,"value":5.7151093,"token":"of","logProb":-16.671537},{"index":5801,"value":10.811387,"token":"\\\\_","logProb":-11.575259},{"index":235298,"value":22.386631,"token":"_","logProb":-1.5258789E-5}]},{"token":559,"topNLogProbs":[{"index":15508,"value":10.369719,"token":"turn","logProb":-15.680775},{"index":2997,"value":10.716382,"token":"start","logProb":-15.334111},{"index":3179,"value":11.302419,"token":"Of","logProb":-14.748075},{"index":10728,"value":11.116491,"token":"OF","logProb":-14.934002},{"index":79795,"value":10.747114,"token":"ofthe","logProb":-15.303379},{"index":12457,"value":13.233422,"token":"your","logProb":-12.817071},{"index":559,"value":26.050415,"token":"of","logProb":-7.8201294E-5},{"index":2672,"value":11.893731,"token":"off","logProb":-14.156762},{"index":235253,"value":12.393969,"token":"o","logProb":-13.656525},{"index":576,"value":16.505135,"token":" of","logProb":-9.545359}]},{"token":235298,"topNLogProbs":[{"index":235293,"value":5.1134872,"token":"=","logProb":-16.041588},{"index":235313,"value":6.167048,"token":">","logProb":-14.988028},{"index":107635,"value":5.385942,"token":"_<","logProb":-15.769133},{"index":15508,"value":6.5135894,"token":"turn","logProb":-14.641485},{"index":235290,"value":7.378484,"token":"-","logProb":-13.776591},{"index":5801,"value":10.262447,"token":"\\\\_","logProb":-10.892628},{"index":905,"value":6.4358773,"token":"__","logProb":-14.719198},{"index":2894,"value":9.978697,"token":" turn","logProb":-11.176378},{"index":1762,"value":6.992061,"token":" _","logProb":-14.163013},{"index":235298,"value":21.155039,"token":"_","logProb":-3.6239624E-5}]},{"token":15508,"topNLogProbs":[{"index":9399,"value":9.644975,"token":"take","logProb":-13.196729},{"index":773,"value":9.894059,"token":"return","logProb":-12.947644},{"index":12457,"value":11.047995,"token":"your","logProb":-11.793709},{"index":19052,"value":10.500204,"token":"tune","logProb":-12.341499},{"index":66787,"value":10.491578,"token":" turno","logProb":-12.350125},{"index":1201,"value":15.757963,"token":"round","logProb":-7.08374},{"index":15508,"value":22.840807,"token":"turn","logProb":-8.9645386E-4},{"index":7617,"value":10.876396,"token":"term","logProb":-11.965307},{"index":690,"value":10.661129,"token":"urn","logProb":-12.180574},{"index":26059,"value":11.29998,"token":"tour","logProb":-11.541723}]},{"token":235313,"topNLogProbs":[{"index":16259,"value":5.961116,"token":">&","logProb":-14.826608},{"index":235248,"value":6.161161,"token":" ","logProb":-14.626562},{"index":32229,"value":6.2910533,"token":">.","logProb":-14.49667},{"index":235298,"value":7.2955647,"token":"_","logProb":-13.492159},{"index":2577,"value":10.25299,"token":"><","logProb":-10.534734},{"index":30561,"value":6.753545,"token":">'","logProb":-14.034179},{"index":14648,"value":9.287034,"token":">\\\\","logProb":-11.5006895},{"index":82305,"value":7.4454193,"token":">*","logProb":-13.342304},{"index":65529,"value":7.399906,"token":">`","logProb":-13.387817},{"index":235313,"value":20.787672,"token":">","logProb":-5.1498413E-5}]},{"token":235248,"topNLogProbs":[{"index":140,"value":11.570394,"token":"   ","logProb":-9.242478},{"index":1,"value":13.599644,"token":"<eos>","logProb":-7.213228},{"index":112,"value":12.868686,"token":"\\n\\n\\n\\n\\n","logProb":-7.944186},{"index":111,"value":15.253961,"token":"\\n\\n\\n\\n","logProb":-5.5589113},{"index":139,"value":14.779063,"token":"  ","logProb":-6.0338087},{"index":110,"value":16.241657,"token":"\\n\\n\\n","logProb":-4.5712147},{"index":109,"value":16.666002,"token":"\\n\\n","logProb":-4.1468697},{"index":235248,"value":20.225313,"token":" ","logProb":-0.58755875},{"index":108,"value":18.124666,"token":"\\n","logProb":-2.6882057},{"index":107,"value":19.74028,"token":"<end_of_turn>","logProb":-1.0725918}]},{"token":108,"topNLogProbs":[{"index":115,"value":9.885976,"token":"\\n\\n\\n\\n\\n\\n\\n\\n","logProb":-11.252725},{"index":114,"value":10.298255,"token":"\\n\\n\\n\\n\\n\\n\\n","logProb":-10.8404455},{"index":112,"value":12.081442,"token":"\\n\\n\\n\\n\\n","logProb":-9.057259},{"index":1,"value":12.321919,"token":"<eos>","logProb":-8.816781},{"index":113,"value":10.7750845,"token":"\\n\\n\\n\\n\\n\\n","logProb":-10.363616},{"index":108,"value":21.101353,"token":"\\n","logProb":-0.037347794},{"index":109,"value":17.269321,"token":"\\n\\n","logProb":-3.869379},{"index":110,"value":16.778517,"token":"\\n\\n\\n","logProb":-4.3601837},{"index":107,"value":14.806606,"token":"<end_of_turn>","logProb":-6.332094},{"index":111,"value":14.089223,"token":"\\n\\n\\n\\n","logProb":-7.0494776}]},{"token":107,"topNLogProbs":[{"index":18925,"value":12.618357,"token":"Would","logProb":-9.445505},{"index":235287,"value":12.772391,"token":"*","logProb":-9.291471},{"index":688,"value":12.754199,"token":"**","logProb":-9.309663},{"index":1,"value":13.899034,"token":"<eos>","logProb":-8.164828},{"index":235322,"value":16.065037,"token":"<","logProb":-5.998825},{"index":1841,"value":13.548776,"token":"What","logProb":-8.515086},{"index":139,"value":13.680857,"token":"  ","logProb":-8.383005},{"index":5331,"value":14.243997,"token":"Let","logProb":-7.819865},{"index":107,"value":22.053408,"token":"<end_of_turn>","logProb":-0.010454178},{"index":235248,"value":16.924097,"token":" ","logProb":-5.139765}]}]""";


        /*
        {"index":9295,"value":12.151476,"token":"random","logProb":-9.054129},
        {"index":235321,"value":15.351205,"token":"8","logProb":-5.8543997},
        {"index":235274,"value":12.902421,"token":"1","logProb":-8.303184},
        {"index":235284,"value":17.47865,"token":"2","logProb":-3.7269554},
        {"index":235324,"value":19.661781,"token":"7","logProb":-1.5438232},
        {"index":235248,"value":14.019234,"token":" ","logProb":-7.186371},
        {"index":235304,"value":19.329502,"token":"3","logProb":-1.8761024},
        {"index":235318,"value":18.46304,"token":"6","logProb":-2.7425652},
        {"index":235310,"value":19.255108,"token":"4","logProb":-1.9504967},
        {"index":235308,"value":20.285082,"token":"5","logProb":-0.9205227}
         */

        /*
        I picked the number **5** !
         */
        assertEquals(expected, JsonUtils.om.writeValueAsString(k.samplerReturns));
    }

}
