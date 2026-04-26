package io.teknek.deliverance.integration;

import com.codahale.metrics.MetricRegistry;
import com.fasterxml.jackson.core.JsonProcessingException;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.JsonUtils;
import io.teknek.deliverance.generator.GeneratorParameters;
import io.teknek.deliverance.generator.Response;
import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.model.*;
import io.teknek.deliverance.safetensors.fetch.ModelFetcher;
import io.teknek.deliverance.safetensors.prompt.PromptSupport;
import io.teknek.deliverance.tensor.ArrayQueueTensorAllocator;
import io.teknek.deliverance.tensor.KvBufferCacheSettings;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;
import io.teknek.deliverance.tensor.operations.NativeSimdTensorOperations;
import io.teknek.deliverance.toolcallparser.DefaultToolCallParser;
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
        ArrayQueueTensorAllocator arrayQueueTensorAllocator = new ArrayQueueTensorAllocator(mr);
        WrappedForkJoinPool pool = new WrappedForkJoinPool(WrappedForkJoinPool.autoSizeByCores());
        NativeSimdTensorOperations operation = new NativeSimdTensorOperations(new ConfigurableTensorProvider(arrayQueueTensorAllocator, pool).get());
        try (AbstractModel m = ModelSupport.loadModel(f, DType.F32, DType.I8, new ConfigurableTensorProvider(operation),
                mr, arrayQueueTensorAllocator, new KvBufferCacheSettings(true), fetch, new NoOpTokenizerRenderer(), new DefaultToolCallParser(), pool)) {
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
    public void xtc() throws JsonProcessingException {
        AbstractModel m = Gemma2Suite.getOrCreate();
        String prompt = "Pick a random number between 1 and 9.";
        PromptSupport.Builder g = m.promptSupport().get().builder()
                .addUserMessage(prompt);
        var uuid = UUID.randomUUID();

        Response k = m.generate(uuid, g.build(), new GeneratorParameters()
                        .withTemperature(0.0f)
                        .withMaxTokens(300)
                        .withLogProbs(true)
                        .withTopLogProbs(10)
                        .withXtcThreshold(0.1f)
                        .withXtcProbability(0.5f)
                , new DoNothingGenerateEvent());

        String expected = """
        [{"token":235285,"topNLogProbs":[{"index":590,"value":15.83738,"token":" I","logProb":-5.8863173,"probability":0.0027771855},{"index":688,"value":15.85426,"token":"**","logProb":-5.869437,"probability":0.0028244625},{"index":235248,"value":16.02266,"token":" ","logProb":-5.7010384,"probability":0.003342493},{"index":4858,"value":16.949995,"token":"Here","logProb":-4.7737026,"probability":0.0084490385},{"index":235308,"value":15.931621,"token":"5","logProb":-5.792077,"probability":0.0030516372},{"index":235304,"value":16.770586,"token":"3","logProb":-4.9531116,"probability":0.0070614023},{"index":2926,"value":17.128735,"token":"My","logProb":-4.594963,"probability":0.010102594},{"index":651,"value":17.120945,"token":"The","logProb":-4.6027527,"probability":0.010024204},{"index":14692,"value":19.175196,"token":"Okay","logProb":-2.548502,"probability":0.07819872},{"index":235285,"value":21.553528,"token":"I","logProb":-0.17016983,"probability":0.84352154}]},{"token":15532,"topNLogProbs":[{"index":1144,"value":14.256382,"token":" am","logProb":-5.928089,"probability":0.0026635667},{"index":26030,"value":14.881573,"token":" rolled","logProb":-5.3028984,"probability":0.004977147},{"index":877,"value":14.32738,"token":" will","logProb":-5.857091,"probability":0.00285955},{"index":1317,"value":15.749116,"token":" just","logProb":-4.435355,"probability":0.011850856},{"index":13670,"value":16.57198,"token":" chose","logProb":-3.6124916,"probability":0.026984528},{"index":15532,"value":17.966393,"token":" picked","logProb":-2.2180786,"probability":0.10881799},{"index":6475,"value":15.464391,"token":" choose","logProb":-4.7200804,"probability":0.008914462},{"index":36011,"value":18.034044,"token":" randomly","logProb":-2.1504269,"probability":0.11643445},{"index":791,"value":18.997715,"token":" have","logProb":-1.1867561,"probability":0.30520973},{"index":235303,"value":19.270552,"token":"'","logProb":-0.91391945,"probability":0.40094963}]},{"token":573,"topNLogProbs":[{"index":921,"value":12.9872465,"token":" out","logProb":-8.707353,"probability":1.6536546E-4},{"index":235292,"value":13.403232,"token":":","logProb":-8.291368,"probability":2.5067144E-4},{"index":955,"value":13.274069,"token":"...","logProb":-8.42053,"probability":2.2029779E-4},{"index":235248,"value":14.162514,"token":" ","logProb":-7.5320854,"probability":5.356201E-4},{"index":6253,"value":13.7007265,"token":" random","logProb":-7.9938726,"probability":3.3752443E-4},{"index":5231,"value":15.688207,"token":" **","logProb":-6.0063925,"probability":0.0024629573},{"index":736,"value":14.059255,"token":" this","logProb":-7.6353445,"probability":4.8307216E-4},{"index":139,"value":14.200627,"token":"  ","logProb":-7.493972,"probability":5.564285E-4},{"index":476,"value":19.640924,"token":" a","logProb":-2.0536747,"probability":0.12826271},{"index":573,"value":21.55116,"token":" the","logProb":-0.14343834,"probability":0.8663742}]},{"token":1758,"topNLogProbs":[{"index":6165,"value":10.260561,"token":" Number","logProb":-11.487459,"probability":1.0257933E-5},{"index":235248,"value":10.733228,"token":" ","logProb":-11.014792,"probability":1.645646E-5},{"index":5968,"value":10.804284,"token":" numbers","logProb":-10.943736,"probability":1.7668342E-5},{"index":2412,"value":11.822517,"token":" following","logProb":-9.925503,"probability":4.8911268E-5},{"index":19068,"value":11.220753,"token":" Random","logProb":-10.527267,"probability":2.6795746E-5},{"index":139,"value":12.317754,"token":"  ","logProb":-9.430266,"probability":8.025782E-5},{"index":36011,"value":11.749056,"token":" randomly","logProb":-9.998964,"probability":4.5446974E-5},{"index":1758,"value":21.680271,"token":" number","logProb":-0.06774902,"probability":0.934495},{"index":6253,"value":19.014359,"token":" random","logProb":-2.7336617,"probability":0.06498092},{"index":5231,"value":13.415579,"token":" **","logProb":-8.332441,"probability":2.4058398E-4}]},{"token":5231,"topNLogProbs":[{"index":2804,"value":10.06843,"token":" ...","logProb":-10.483812,"probability":2.7985827E-5},{"index":235269,"value":10.945792,"token":",","logProb":-9.60645,"probability":6.729329E-5},{"index":968,"value":10.404237,"token":" <","logProb":-10.1480055,"probability":3.9154096E-5},{"index":19190,"value":11.430997,"token":" ***","logProb":-9.121245,"probability":1.0931845E-4},{"index":139,"value":15.879427,"token":"  ","logProb":-4.6728153,"probability":0.009345921},{"index":649,"value":10.74961,"token":" *","logProb":-9.802632,"probability":5.5305823E-5},{"index":955,"value":12.200102,"token":"...","logProb":-8.35214,"probability":2.3589107E-4},{"index":235248,"value":17.94088,"token":" ","logProb":-2.6113625,"probability":0.07343443},{"index":5231,"value":20.44819,"token":" **","logProb":-0.10405159,"probability":0.9011788},{"index":235292,"value":16.379269,"token":":","logProb":-4.1729736,"probability":0.0154063795}]},{"token":235310,"topNLogProbs":[{"index":235274,"value":11.549464,"token":"1","logProb":-10.14655,"probability":3.921112E-5},{"index":9295,"value":12.43191,"token":"random","logProb":-9.264105,"probability":9.476553E-5},{"index":235304,"value":19.100357,"token":"3","logProb":-2.5956573,"probability":0.07459682},{"index":235248,"value":14.117612,"token":" ","logProb":-7.5784025,"probability":5.1137747E-4},{"index":235321,"value":14.991996,"token":"8","logProb":-6.7040186,"probability":0.0012259753},{"index":235318,"value":19.273897,"token":"6","logProb":-2.4221172,"probability":0.08873355},{"index":235324,"value":20.172935,"token":"7","logProb":-1.5230789,"probability":0.21803953},{"index":235308,"value":21.02075,"token":"5","logProb":-0.67526436,"probability":0.5090218},{"index":235284,"value":16.400229,"token":"2","logProb":-5.295786,"probability":0.0050126733},{"index":235310,"value":19.41926,"token":"4","logProb":-2.2767544,"probability":0.10261672}]},{"token":168428,"topNLogProbs":[{"index":235284,"value":8.352479,"token":"2","logProb":-12.630736,"probability":3.2699484E-6},{"index":235324,"value":8.459996,"token":"7","logProb":-12.523219,"probability":3.6411204E-6},{"index":5231,"value":10.050638,"token":" **","logProb":-10.932577,"probability":1.7866607E-5},{"index":775,"value":9.539424,"token":"/**","logProb":-11.443791,"probability":1.0715798E-5},{"index":235341,"value":13.760819,"token":"!","logProb":-7.222396,"probability":7.300512E-4},{"index":168428,"value":19.597937,"token":"**.","logProb":-1.3852787,"probability":0.25025403},{"index":190213,"value":10.44975,"token":"**,","logProb":-10.533465,"probability":2.663018E-5},{"index":95573,"value":11.5700865,"token":"**:","logProb":-9.413129,"probability":8.164509E-5},{"index":116742,"value":10.572018,"token":".**","logProb":-10.411198,"probability":3.0093612E-5},{"index":688,"value":20.693966,"token":"**","logProb":-0.28924942,"probability":0.74882543}]},{"token":968,"topNLogProbs":[{"index":108,"value":14.377342,"token":"\\n","logProb":-6.602808,"probability":0.0013565535},{"index":123781,"value":15.091771,"token":" 😄","logProb":-5.888379,"probability":0.0027714653},{"index":109,"value":16.018827,"token":"\\n\\n","logProb":-4.961323,"probability":0.0070036575},{"index":140,"value":15.097087,"token":"   ","logProb":-5.8830633,"probability":0.002786237},{"index":235248,"value":19.590944,"token":" ","logProb":-1.3892059,"probability":0.24927317},{"index":968,"value":20.058218,"token":" <","logProb":-0.9219322,"probability":0.39774975},{"index":44416,"value":17.340939,"token":" 😊","logProb":-3.6392117,"probability":0.026273048},{"index":2692,"value":15.298018,"token":" </","logProb":-5.682132,"probability":0.0034062893},{"index":51984,"value":15.551226,"token":" 😉","logProb":-5.4289246,"probability":0.004387812},{"index":139,"value":19.767843,"token":"  ","logProb":-1.212307,"probability":0.29751015}]},{"token":615,"topNLogProbs":[{"index":3310,"value":6.350381,"token":"next","logProb":-13.217238,"probability":1.8189731E-6},{"index":13072,"value":7.381625,"token":"answer","logProb":-12.185994,"probability":5.101407E-6},{"index":479,"value":6.632854,"token":"en","logProb":-12.934765,"probability":2.4126982E-6},{"index":1598,"value":7.9216433,"token":"br","logProb":-11.645976,"probability":8.754208E-6},{"index":3589,"value":7.670664,"token":"current","logProb":-11.8969555,"probability":6.81111E-6},{"index":1432,"value":8.221684,"token":"span","logProb":-11.345935,"probability":1.1817431E-5},{"index":2997,"value":14.142801,"token":"start","logProb":-5.424818,"probability":0.004405868},{"index":1387,"value":9.18991,"token":"center","logProb":-10.377709,"probability":3.111846E-5},{"index":1580,"value":8.155562,"token":" end","logProb":-11.412057,"probability":1.1061312E-5},{"index":615,"value":19.563087,"token":"end","logProb":-0.0045318604,"probability":0.9954784}]},{"token":235298,"topNLogProbs":[{"index":75076,"value":4.346281,"token":"_*","logProb":-19.421875,"probability":3.6744037E-9},{"index":17824,"value":4.4393835,"token":"_\\\\","logProb":-19.328773,"probability":4.032927E-9},{"index":559,"value":5.6585526,"token":"of","logProb":-18.109604,"probability":1.36489415E-8},{"index":235290,"value":6.911684,"token":"-","logProb":-16.856472,"probability":4.778892E-8},{"index":235293,"value":4.8647165,"token":"=","logProb":-18.903439,"probability":6.1707923E-9},{"index":235313,"value":5.6678224,"token":">","logProb":-18.100334,"probability":1.3776051E-8},{"index":5801,"value":12.237074,"token":"\\\\_","logProb":-11.531082,"probability":9.820072E-6},{"index":1762,"value":6.9286823,"token":" _","logProb":-16.839474,"probability":4.8608193E-8},{"index":235298,"value":23.768145,"token":"_","logProb":-1.1444092E-5,"probability":0.99998856},{"index":576,"value":9.573241,"token":" of","logProb":-14.194915,"probability":6.842689E-7}]},{"token":559,"topNLogProbs":[{"index":12457,"value":10.098518,"token":"your","logProb":-15.765472,"probability":1.422795E-7},{"index":224934,"value":10.54262,"token":"ofType","logProb":-15.321371,"probability":2.2182633E-7},{"index":10728,"value":10.712591,"token":"OF","logProb":-15.1514,"probability":2.6292432E-7},{"index":2997,"value":10.895124,"token":"start","logProb":-14.968866,"probability":3.15576E-7},{"index":235253,"value":13.3046,"token":"o","logProb":-12.559391,"probability":3.5117678E-6},{"index":3179,"value":11.172547,"token":"Of","logProb":-14.691443,"probability":4.164733E-7},{"index":79795,"value":11.352855,"token":"ofthe","logProb":-14.511136,"probability":4.987624E-7},{"index":2672,"value":11.150505,"token":"off","logProb":-14.713486,"probability":4.0739374E-7},{"index":576,"value":15.137317,"token":" of","logProb":-10.726674,"probability":2.1951522E-5},{"index":559,"value":25.863962,"token":"of","logProb":-2.861023E-5,"probability":0.9999714}]},{"token":235298,"topNLogProbs":[{"index":235293,"value":5.5766525,"token":"=","logProb":-16.215149,"probability":9.075077E-8},{"index":235313,"value":6.54971,"token":">","logProb":-15.242092,"probability":2.4012837E-7},{"index":107635,"value":5.737829,"token":"_<","logProb":-16.053972,"probability":1.06622394E-7},{"index":905,"value":6.9218426,"token":"__","logProb":-14.869959,"probability":3.4838456E-7},{"index":1762,"value":7.5903673,"token":" _","logProb":-14.201434,"probability":6.798225E-7},{"index":15508,"value":5.998664,"token":"turn","logProb":-15.793138,"probability":1.3839727E-7},{"index":235298,"value":21.791758,"token":"_","logProb":-4.386902E-5,"probability":0.99995613},{"index":2894,"value":10.907417,"token":" turn","logProb":-10.884384,"probability":1.8748737E-5},{"index":5801,"value":11.003063,"token":"\\\\_","logProb":-10.788738,"probability":2.0630536E-5},{"index":235290,"value":9.037113,"token":"-","logProb":-12.754688,"probability":2.8887455E-6}]},{"token":15508,"topNLogProbs":[{"index":66787,"value":9.826621,"token":" turno","logProb":-13.822907,"probability":9.926304E-7},{"index":19052,"value":10.033852,"token":"tune","logProb":-13.615677,"probability":1.2211997E-6},{"index":2997,"value":9.976598,"token":"start","logProb":-13.672931,"probability":1.1532452E-6},{"index":7617,"value":11.151775,"token":"term","logProb":-12.497753,"probability":3.7350358E-6},{"index":2894,"value":10.813786,"token":" turn","logProb":-12.835743,"probability":2.6638372E-6},{"index":26059,"value":11.356966,"token":"tour","logProb":-12.2925625,"probability":4.585724E-6},{"index":690,"value":11.309041,"token":"urn","logProb":-12.3404875,"probability":4.3711366E-6},{"index":1201,"value":12.063595,"token":"round","logProb":-11.585934,"probability":9.295932E-6},{"index":15508,"value":23.649483,"token":"turn","logProb":-4.5776367E-5,"probability":0.9999542},{"index":12457,"value":12.09305,"token":"your","logProb":-11.5564785,"probability":9.573818E-6}]},{"token":235313,"topNLogProbs":[{"index":16259,"value":5.584489,"token":">&","logProb":-15.683471,"probability":1.5443837E-7},{"index":32403,"value":5.836206,"token":">[","logProb":-15.431753,"probability":1.9864368E-7},{"index":32229,"value":6.379075,"token":">.","logProb":-14.888885,"probability":3.4185317E-7},{"index":30561,"value":6.0564284,"token":">'","logProb":-15.211531,"probability":2.4758035E-7},{"index":82305,"value":6.4104366,"token":">*","logProb":-14.857523,"probability":3.527441E-7},{"index":65529,"value":7.5334353,"token":">`","logProb":-13.734524,"probability":1.0843565E-6},{"index":14648,"value":9.443291,"token":">\\\\","logProb":-11.824669,"probability":7.321694E-6},{"index":235313,"value":21.267939,"token":">","logProb":-2.0980835E-5,"probability":0.999979},{"index":1562,"value":6.295532,"token":" >","logProb":-14.972427,"probability":3.144542E-7},{"index":2577,"value":9.709478,"token":"><","logProb":-11.558481,"probability":9.554663E-6}]},{"token":108,"topNLogProbs":[{"index":140,"value":12.310919,"token":"   ","logProb":-9.389467,"probability":8.359998E-5},{"index":44416,"value":12.333406,"token":" 😊","logProb":-9.36698,"probability":8.550125E-5},{"index":110,"value":16.006866,"token":"\\n\\n\\n","logProb":-5.6935196,"probability":0.003367719},{"index":112,"value":12.937077,"token":"\\n\\n\\n\\n\\n","logProb":-8.7633095,"probability":1.5636627E-4},{"index":139,"value":15.796337,"token":"  ","logProb":-5.904049,"probability":0.0027283754},{"index":108,"value":19.511087,"token":"\\n","logProb":-2.1892986,"probability":0.11199527},{"index":109,"value":17.242485,"token":"\\n\\n","logProb":-4.457901,"probability":0.011586658},{"index":111,"value":14.996491,"token":"\\n\\n\\n\\n","logProb":-6.7038946,"probability":0.0012261273},{"index":107,"value":18.91832,"token":"<end_of_turn>","logProb":-2.7820663,"probability":0.061910447},{"index":235248,"value":21.48463,"token":" ","logProb":-0.21575546,"probability":0.80593234}]},{"token":107,"topNLogProbs":[{"index":235287,"value":11.537923,"token":"*","logProb":-9.29639,"probability":9.175491E-5},{"index":18925,"value":11.593517,"token":"Would","logProb":-9.240795,"probability":9.700043E-5},{"index":1841,"value":12.365625,"token":"What","logProb":-8.468687,"probability":2.0994036E-4},{"index":688,"value":12.206853,"token":"**","logProb":-8.62746,"probability":1.7911912E-4},{"index":235322,"value":16.742298,"token":"<","logProb":-4.0920143,"probability":0.01670555},{"index":5331,"value":12.707554,"token":"Let","logProb":-8.126759,"probability":2.9552457E-4},{"index":235248,"value":16.152103,"token":" ","logProb":-4.682209,"probability":0.009258539},{"index":1,"value":13.710598,"token":"<eos>","logProb":-7.1237144,"probability":8.057682E-4},{"index":139,"value":12.498736,"token":"  ","logProb":-8.335576,"probability":2.39831E-4},{"index":107,"value":20.805346,"token":"<end_of_turn>","logProb":-0.028966904,"probability":0.9714486}]}]""";

        //assertEquals("I picked the number **4**. <end_of_turn>".trim(), k.responseText.trim());
        //assertEquals("[235285, 15532, 573, 1758, 5231, 235310, 168428, 968, 615, 235298, 559, 235298, 15508, 235313, 108, 107]", k.generatedTokens.toString());
        //assertEquals(expected, JsonUtils.om.writeValueAsString(k.samplerReturns));
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

    }

    @Test
    public void logProbs() throws JsonProcessingException {
        AbstractModel m = Gemma2Suite.getOrCreate();
        String prompt = "Pick a random number between 1 and 9. Replay with only the pick.";
        PromptSupport.Builder g = m.promptSupport().get().builder()
                .addUserMessage(prompt);
        var uuid = UUID.randomUUID();

        Response response = m.generate(uuid, g.build(), new GeneratorParameters()
                        .withTemperature(0.0f)
                        .withMaxTokens(300)
                        .withLogProbs(true)
                        .withTopLogProbs(10)
                , new DoNothingGenerateEvent());
        assertEquals("I picked the number **5**! \uD83C\uDFB2 \n" +
                "<end_of_turn>".trim(), response.responseText.trim());

    }

    @Test
    public void topK() throws JsonProcessingException {
        AbstractModel m = Gemma2Suite.getOrCreate();
        String prompt = "Pick a random number between 1 and 9. Reply with only the pick.";
        PromptSupport.Builder g = m.promptSupport().get().builder()
                .addUserMessage(prompt);
        var uuid = UUID.randomUUID();

        Response response = m.generate(uuid, g.build(), new GeneratorParameters()
                        .withTemperature(0.3f)
                        .withMaxTokens(20)
                        .withLogProbs(false)
                        .withTopK(0.9999f)
                        //.withLogProbs(true)
                        //.withTopLogProbs(10)
                , new GenerateEvent() {
                    @Override
                    public void emit(int next, String nextRaw, String nextCleaned, float timing) {
                        System.out.println(nextCleaned);
                    }
                });
        System.out.println(response.responseText);
        //assertEquals("I picked the number **5**! \uD83C\uDFB2 \n" +
        //        "<end_of_turn>".trim(), response.responseText.trim());

    }

    @Test
    public void temperatureAndXtc() throws JsonProcessingException {
        AbstractModel m = Gemma2Suite.getOrCreate();
        String prompt = "Pick a random number between 1 and 9. ";
        PromptSupport.Builder g = m.promptSupport().get().builder()
                .addUserMessage(prompt);
        var uuid = UUID.randomUUID();

        Response response = m.generate(uuid, g.build(), new GeneratorParameters()
                        .withTemperature(0.3f)
                        .withMaxTokens(20)
                        .withLogProbs(true)
                        .withTopLogProbs(10)
                        .withXtcThreshold(0.1f)
                        .withXtcProbability(0.5f)

                , new GenerateEvent() {
                    @Override
                    public void emit(int next, String nextRaw, String nextCleaned, float timing) {
                        System.out.println(nextCleaned);
                    }
                });
        System.out.println(response.responseText);
        //assertEquals("I picked the number **5**! \uD83C\uDFB2 \n" +
        //        "<end_of_turn>".trim(), response.responseText.trim());

    }

    @Test
    public void temperatureAndTopp() throws JsonProcessingException {
        AbstractModel m = Gemma2Suite.getOrCreate();
        String prompt = "Pick a random number between 1 and 9. Reply with only the pick. ";
        PromptSupport.Builder g = m.promptSupport().get().builder()
                .addUserMessage(prompt);
        var uuid = UUID.randomUUID();

        Response response = m.generate(uuid, g.build(), new GeneratorParameters()
                        .withTemperature(0.9f)
                        .withMaxTokens(20)
                        .withLogProbs(true)
                        .withTopLogProbs(10)
                        .withTopP(0.2f)

                , new GenerateEvent() {
                    @Override
                    public void emit(int next, String nextRaw, String nextCleaned, float timing) {
                        System.out.println(nextCleaned);
                    }
                });
        System.out.println(response.responseText);
        //assertEquals("I picked the number **5**! \uD83C\uDFB2 \n" +
        //        "<end_of_turn>".trim(), response.responseText.trim());

    }

}
