#### deliverance

The name `Deliverance` 

https://www.merriam-webster.com/dictionary/deliverance

: the act of delivering someone or something : the state of being delivered
especially : liberation, `rescue`

Have you ever spent 15 minutes building VLLM to end up with a disk full from its 20GB image and 60GB of docker layers?
Deliverance compiles < 1 minute. Int a 33MB boot application.

: something delivered
especially : an `opinion` or `decision` (such as the verdict of a jury) expressed publicly.

Could just go with j-inference open-jay-i that would be boring. We aren't `inferencing`, we are
`delivering`

### Lightning quick start 

```shell
 export JAVA_HOME=/usr/lib/jvm/java-24-temurin-jdk
 # dont skip the tets all the time they are fun, but just this time
 mvn package -Dmaven.test.skip=true
 cd web
edward@fedora:~/deliverence/web$ sh run.sh 
WARNING: Using incubator modules: jdk.incubator.vector
Standard Commons Logging discovery in action with spring-jcl: please remove commons-logging.jar from classpath in order to avoid potential conflicts

  .   ____          _            __ _ _
 /\\ / ___'_ __ _ _(_)_ __  __ _ \ \ \ \
( ( )\___ | '_ | '_| | '_ \/ _` | \ \ \ \
 \\/  ___)| |_)| | | | | || (_| |  ) ) ) )
  '  |____| .__|_| |_|_| |_\__, | / / / /
 =========|_|==============|___/=/_/_/_/

 :: Spring Boot ::                (v3.5.5)

2025-10-30T14:37:10.247-04:00  INFO 218011 --- [           main] n.d.http.DeliveranceApplication          : Starting DeliveranceApplication using Java 24.0.2 with PID 218011 (/home/edward/deliverence/web/target/web-0.0.1-SNAPSHOT.jar started by edward in /home/edward/deliverence/web)
2025-10-30T14:37:13.932-04:00  WARN 218011 --- [           main] i.t.deliverance.model.AbstractModel      : embedding TensorShape{tshape=[57, 2048], capacity=116736, sparseRange=Optional.empty} 116736
2025-10-30T14:37:32.561-04:00  WARN 218011 --- [           main] i.t.deliverance.model.AbstractModel      : After batch forward size: 116736 shape: TensorShape{tshape=[57, 2048], capacity=116736, sparseRange=Optional.empty}
Response{responseText='10000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000', responseTextWithSpecialTokens='10000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000', finishReason=MAX_TOKENS, promptTokens=56, generatedTokens=199, promptTimeMs=19012, generateTimeMs=77902}
2025-10-30T14:38:52.134-04:00  INFO 218011 --- [           main] o.s.b.a.w.s.WelcomePageHandlerMapping    : Adding welcome page: class path resource [public/index.html]
2025-10-30T14:38:53.002-04:00  INFO 218011 --- [           main] n.d.http.DeliveranceApplication          : Started DeliveranceApplication in 103.909 seconds (process running for 105.417)

```
We run n query on startup to warm the cache sorry :) otherwise it would start faster. Also first time has to 
download a model!

Open your browser to http://localhost:8080

<p align="center">
  <img src="deliv.png"  alt="Deliver me">
</p>

### Using as a library
Odds are you don't want to use the amazing UI depicted above. Deliverance is made in components and 
very easy to use as a library as you might use transformers/

```java
ModelFetcher fetch = new ModelFetcher(modelOwner, modelName);
File f = fetch.maybeDownload();
MetricRegistry mr = new MetricRegistry();
TensorCache tensorCache = new TensorCache(mr);
try (AbstractModel m = ModelSupport.loadModel(f, DType.F32, DType.I8, new ConfigurableTensorProvider(tensorCache),
        new MetricRegistry(), tensorCache, new KvBufferCacheSettings(true))) {
    String prompt = "What is the best season to plant avocados?";
    PromptContext ctx;
    PromptSupport ps = m.promptSupport().get();
    Response r = m.generate(UUID.randomUUID(), ctx, new GeneratorParameters().withSeed(42), (s1, f1) -> {
    });
    System.out.println(r);
}
```

### Performance


#### CPU/GPU
In case you have been hiding under a rock I will let you in on the secret that GPUs are magic. LLM are very
compute bound. There are a few specific performance optimizations you should understand.

- NaiveTensorOperations does matrix operations using loops and arrays
- PanamaTensorOperations uses the "vector" aka project panama support now in java SIMD native to java
- NativeSimdTensorOperations uses native code "C" through JNI. SIMD from C runs well on optimized x86_64 hardware
- NativeGPUTensorOperations uses native code "C" and "shaders" through JNI. Requires an actual GPU

Not everything is fully optimized and some of the Operations classes delegate some methods to 
each other. The class *ConfigurableTensorProvider* will auto pick, but you can use an explicit list.

#### DISK/Ram

For larger models (even quanitized ones) the disk footprint is large 4GB - 100GB. Deliverance memory maps those files however
fast disk and ample RAM are needed as the disk access is very heavy (load from disk , load from disk , multiply). If you
do not have enough RAM disk cache and IOWait will be a big bottleneck

#### KVBuffer Cache
KvBufferCache can be sized in bytes. It can also be persisted to disk, but it does not clean up itself so feature is off by default.



#### Small/Quantized models
If you are running on a device without GPU your best mileage comes from going with the quantized models. 
Effectively this we are working with big arrays of floating point numbers, and quantizing (fancy rounding) 
down to Q4 helps the SIMD (Single Instruction Multiple Data) improves performance significantly. It does't 
make "blazing speed" and the small models just sometimes make nonsense, but it is nice for prototyping. 


#### Slowness with spring-boot run

After troubleshooting all the wrong things for hours I found not to use:
``` mvn:spring-boot run ```

The debug mode seems to remove lots of optimizations causing very slow runtime. *web/run.sh* should be a good stand in.