## Prefix Cache

Prefix cache is incredibly useful when you have large **system/root prompts**. Here is an example where we have a set
of directives that we will give to every prompt. 


```
PromptContext ctx = m.promptSupport().get().builder()
    .addSystemMessage("You are an assistant that produces concise, production-grade software.")
    .addSystemMessage("Output java code.")
    .addSystemMessage("Refrain from editorializing your reply.")
    .addSystemMessage("Generate java code into the package 'io.teknek.shape' .")
    .addSystemMessage("Do not import java.awt")
```

Obviously if we can "reuse" the prompt we are winning. This is what the
prefix cache does.

### KV Cache settings

The settings are applied when the model is initialized. (not per query)
```
KvBufferCacheSettings settings = new KvBufferCacheSettings(true)
                .withMaxPrefixTokensPerPrompt(512)
                .withMaxEntries(10_000)
                .withBlockSize(16);
try (AbstractModel m = AutoModelForCausaLm.newBuilder(fetch).withWorkingQuantType(DType.I8)
        .withKvBufferCacheSettings(settings)
        .withTokenTokenRenderer(new TokenizerRenderer()).build()){
```

* maxPrefixTokensPerPrompt - Regardless of how long a prompt is sent only consider this many tokens for the cache
* maxEntries - the size of the cache. IF this is a memory cache eviction happens passed this limit
* withBlockSize - the number of tokens in a block (explained below)

#### Block size
It is possible to store the KVs at each token, but that is a bad idea with limited value. Some words can be multiple 
tokens and the KV information is large (MBs). We get the most value from a long complete match, so the machines
only occur at the block boundaries. 

### How do you know it is working

A number of metrics were added to the feature, but we are so proud of it we even log information every prompt

```
[main] INFO io.teknek.deliverance.model.AbstractModel - time_to_first_token=811.371745 prefix_length=24
```

The higher the prefix_length the larger KV match you found. time_to_first_token should be a lower number when 
there is any matching.
