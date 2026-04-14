## Generator Sampling
The Generative part of generative AI uses what we all know of as "Math Magic" to generate text/images/sound. 
Let's go through some of the basics: 

## Temperature 0.0
The default for temperature is 0.0f. The model's (gemma2, llama) has
given it's "best answer". The highest scoring "logit" or token.

```java
String prompt = "Pick a random number between 1 and 9. Reply with only the pick. ";
PromptSupport.Builder g = m.promptSupport().get().builder()
        .addUserMessage(prompt);
Response response = m.generate(UUID.randomUUID(), g.build(), new GeneratorParameters());
```
Looking at the prompt, you or I may understand it means. It seems self-explanatory, however a lower
powered model 1B params may struggle with the question. When the models struggle "random things" come out and tuning every
parameter likely wont help. The model might answer like "I'm going to pick a number becaue I like picking my poison. Poisons are listed on the perioic table next to the milk in your grocers freezer."

## Assuming "understanding"

Let's assume the model is powerful enough that it does "understand". Internally it is going to come up with a bunch 
of choices based on how it was trained.  

| logit/token | decoded | logprob    |
|-------------|---------|------------|
| 343         | 5       | -0.9205227 | 
| 235         | 3       | -1.1000000 |

For logprob smaller negative numbers are better. We will show you how to convert to easier to understand values later.
For reference the entire list looks like this:

```json
        [{"index":9295,"value":12.151476,"token":"random","logProb":-9.054129},
        {"index":235321,"value":15.351205,"token":"8","logProb":-5.8543997},
        {"index":235274,"value":12.902421,"token":"1","logProb":-8.303184},
        {"index":235284,"value":17.47865,"token":"2","logProb":-3.7269554},
        {"index":235324,"value":19.661781,"token":"7","logProb":-1.5438232},
        {"index":235248,"value":14.019234,"token":" ","logProb":-7.186371},
        {"index":235304,"value":19.329502,"token":"3","logProb":-1.8761024},
        {"index":235318,"value":18.46304,"token":"6","logProb":-2.7425652},
        {"index":235310,"value":19.255108,"token":"4","logProb":-1.9504967},
        {"index":235308,"value":20.285082,"token":"5","logProb":-0.9205227}]
```

## Understanding ?

You may have noticed a few things in the JSON "toplogprobs" output: The different numbers 1-9 have significantly different values.
Logits for " " and "random" have snuck into the top list. The space could be the models instinct to pad the value, 
and the "random" could be an attempt to start trying to "answer" the question with "prose", even though we told it to only 
give us a single pick. 

## temperature change

Lets change the prompt to get some more diverse output. Notice I removed the "Reply with only the pick". This gives the 
model some room to get "chatty".
```json
   String prompt = "Pick a random number between 1 and 9.";
   Response response = m.generate(uuid, g.build(), new GeneratorParameters().withTemperature(0.3f)
```

Notice the probability column, that is calculated from the logprob. You can see here the distribution is wider then the first example.
The model wants to start the sentence with "My" (0.010102594). That is a roughly 10% chance.

```json
{"index":590,"value":15.83738,"token":" I","logProb":-5.8863173,"probability":0.0027771855},
{"index":688,"value":15.85426,"token":"**","logProb":-5.869437,"probability":0.0028244625},
{"index":235248,"value":16.02266,"token":" ","logProb":-5.7010384,"probability":0.003342493},
{"index":4858,"value":16.949995,"token":"Here","logProb":-4.7737026,"probability":0.0084490385},
{"index":235308,"value":15.931621,"token":"5","logProb":-5.792077,"probability":0.0030516372},
{"index":235304,"value":16.770586,"token":"3","logProb":-4.9531116,"probability":0.0070614023},
{"index":2926,"value":17.128735,"token":"My","logProb":-4.594963,"probability":0.010102594}
```

The temperature transformation function applied to every logit.
```java
  try (AbstractTensor scaledLogits = model.getTensorCache().getDirty(logits.dType(), logits.shape())) {
        for (int i = 0; i < model.config.vocabularySize; i++) {
            float v = logits.get(0, i) / temperature;
            scaledLogits.set(v, 0, i);
        }
   ...
```
The average model has approximately 200,000-400,000 logits. All these
methods have a lot of bits to shift.

If we divide all the logits by the same number we are going to either cause "clumping" ".99" or "stratification" "0.2". 
Nothing stops us at 1, we could divide by 1.2! Either way, reshaping does not do anything if we are only picking 
the largest one. This is where the sampling is important.

## Nucleus sampling default (top_p 0.1  10%)

https://en.wikipedia.org/wiki/Top-p_sampling

To understand why top_p woks so well lets look at the problem. Some models have  
200,000 to 300,000 tokens. Setting a percentile cutoff is very hard. 

200_000 * .9999 = 20

For our problem there are only 9 good answers 1-9. If the model picks the "20th" answer it won't be good.

If we asked the question, "Pick a city in the United States". There are thousands of good answers
and the 20th would likely be fine.

For top_p we sort the probabilities highest first for each logit 

| 5  | 3  | ... |   shoe |
|:---|:--:|----:|-------:|
| 8% | 4% | ... | .0001% |

Top-p defaults to 10%. We cut-off the top 10% and rescale

| 5   |  3  |
|:----|:---:|
| .66 | .33 |

The inference engine draws a psuedo-random number like 0.09, 9%. That would fall-in the 5 bucket as 0.05 < .63. 
Another draw might generate 0.88 or 88%. That lands in the second bucket.



