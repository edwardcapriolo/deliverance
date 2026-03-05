package io.teknek.deliverance.vibrant;

import com.codahale.metrics.MetricRegistry;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.model.embedding.EmbeddingModel;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.model.output.Response;
import dev.langchain4j.store.embedding.EmbeddingMatch;
import dev.langchain4j.store.embedding.EmbeddingSearchRequest;
import dev.langchain4j.store.embedding.EmbeddingSearchResult;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.generator.GeneratorParameters;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.ModelSupport;
import io.teknek.deliverance.safetensors.fetch.ModelFetcher;
import io.teknek.deliverance.safetensors.prompt.PromptSupport;
import io.teknek.deliverance.tensor.KvBufferCacheSettings;
import io.teknek.deliverance.tensor.TensorCache;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;
import io.teknek.deliverance.tensor.operations.NativeSimdTensorOperations;
import org.apache.maven.plugin.AbstractMojo;
import org.apache.maven.plugin.MojoExecutionException;
import org.apache.maven.plugins.annotations.LifecyclePhase;
import org.apache.maven.plugins.annotations.Mojo;
import org.apache.maven.plugins.annotations.Parameter;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.file.FileSystems;
import java.nio.file.Path;
import java.nio.file.PathMatcher;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.UUID;

@Mojo(name = "ragchat", defaultPhase = LifecyclePhase.PACKAGE)
public class RagChatMojo  extends AbstractMojo {
    @Parameter(name="modelConfig")
    private ModelConfig modelConfig;

    EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();
    InMemoryEmbeddingStore<TextSegment> vectorStore = new InMemoryEmbeddingStore<>();
    String embeddingQuery;
    StringBuilder embeddingBuffer;

    public void addPathToVectorStore(String path, String pattern){
        Path directoryPath = Paths.get(path);
        DocumentSplitter splitter = DocumentSplitters.recursive(1000, 100);
        PathMatcher pathMatcher = FileSystems.getDefault().getPathMatcher(pattern);
        List<Document> documents = FileSystemDocumentLoader.loadDocumentsRecursively(
                directoryPath, pathMatcher
        );
        List<Embedding> embeddings = new ArrayList<>();
        List<TextSegment> chunks = new ArrayList<>();

        for (Document d: documents){
            List<TextSegment> splits = splitter.split(d);
            for (TextSegment seg: splits){
                Response<Embedding> embed = embeddingModel.embed(seg);
                Embedding embeddingVector = embed.content();
                chunks.add(seg);
                embeddings.add(embeddingVector);
            }
        }
        vectorStore.addAll(embeddings, chunks);
    }

    private void indexLine(String inputLine){
        String path = inputLine.split("\\s+")[1];
        addPathToVectorStore(path, "glob:**/*.java");
        System.out.println("Vectorized " + path);
        System.out.print(">> ");
    }

    private void embeddingLine(String inputLine){
        this.embeddingQuery = inputLine.substring(inputLine.indexOf("embedding") + 9 );
        System.out.println("Embedding query has been set to: " + this.embeddingQuery);
        Embedding queryEmbedding = embeddingModel.embed(this.embeddingQuery).content();
        EmbeddingSearchRequest searchRequest = EmbeddingSearchRequest.builder()
                .queryEmbedding(queryEmbedding)
                .maxResults(3)
                .build();
        EmbeddingSearchResult<TextSegment> matches = vectorStore.search(searchRequest);

        this.embeddingBuffer = new StringBuilder();
        for (EmbeddingMatch<TextSegment> match : matches.matches()) {
            TextSegment segment = match.embedded();
            System.out.println("Embed id " + match.embeddingId());
            System.out.println(match.embedded().metadata());
            System.out.println("Relevance Score: " + match.score());
            System.out.println(segment.text().substring(0, Math.min(100, segment.text().length())) + "...\n");
            embeddingBuffer.append(match.embedded().metadata() + "\n");
            embeddingBuffer.append(segment.text());
        }
        System.out.print(">> ");
    }

    public void execute() throws MojoExecutionException {
        if (modelConfig == null) {
            modelConfig = new ModelConfig();
        }

        ModelFetcher fetch = new ModelFetcher(modelConfig.getOwner(), modelConfig.getModelName());
        File f = fetch.maybeDownload();
        MetricRegistry mr = new MetricRegistry();
        TensorCache tensorCache = new TensorCache(mr);
        DType working = DType.valueOf(modelConfig.getWorkingMemType());
        DType quantized = DType.valueOf(modelConfig.getQuantizedMemType());

        NativeSimdTensorOperations operation = new NativeSimdTensorOperations(new ConfigurableTensorProvider(tensorCache).get());
        try (AbstractModel model = ModelSupport.loadModel(f, working, quantized, new ConfigurableTensorProvider(operation),
                new MetricRegistry(), tensorCache, new KvBufferCacheSettings(true), fetch)) {
            System.out.println("Chat with deliverance! Type 'undeliver' to quit.");
            System.out.println("Type 'indexjava /path/to/source' to add a path to the vector store ");
            System.out.print(">> ");
            BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
            try {
                String inputLine;
                while (!(inputLine = br.readLine()).equalsIgnoreCase("undeliver")) {
                    if (inputLine.startsWith("indexjava")) {
                        indexLine(inputLine);
                        continue;
                    }
                    if (inputLine.startsWith("embedding")){
                        embeddingLine(inputLine);
                        continue;
                    }
                    if (inputLine.startsWith("template")){
                        String rest =  inputLine.substring(inputLine.indexOf(" "));
                        String promptTemplate = "Answer the following question based on the provided context.\n\n"
                                + "Context:\n%s\n\n"
                                + "Question: %s";

                        String prompt = String.format(promptTemplate, this.embeddingBuffer, rest);
                        PromptSupport.Builder b = model.promptSupport().get().builder();
                        b.addUserMessage(prompt);
                        var uuid = UUID.randomUUID();
                        io.teknek.deliverance.generator.Response k = model.generate(uuid, b.build(), new GeneratorParameters()
                                .withNtokens(2048)
                                .withIncludeStopStrInOutput(false)
                                .withStopWords(List.of("<|eot_id|>"))
                                .withTemperature(0.2f)
                                .withSeed(99998), (int next, String tok, String s1, float f1) -> {
                            System.out.print( model.getTokenRenderer().tokenizerToRendered(s1));
                        });
                        System.out.println(">> ");
                        continue;
                    }
                    PromptSupport.Builder b = model.promptSupport().get().builder();
                    b.addUserMessage(inputLine);
                    var uuid = UUID.randomUUID();
                    io.teknek.deliverance.generator.Response k = model.generate(uuid, b.build(), new GeneratorParameters()
                            .withNtokens(512)
                            .withIncludeStopStrInOutput(false)
                            .withStopWords(List.of("<|eot_id|>"))
                            .withTemperature(0.2f)
                            .withSeed(99998), (int next, String tok, String s1, float f1) -> {
                        System.out.print(model.getTokenRenderer().tokenizerToRendered(s1));
                    });
                    System.out.println();
                    System.out.print(">> ");
                }
            } catch (IOException e) {
                throw new MojoExecutionException(e);
            }
        }
    }

    public static void main(String [] args){
        InMemoryEmbeddingStore<TextSegment> vectorStore = new InMemoryEmbeddingStore<>();
        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();
        Path directoryPath = Paths.get("/home/edward/deliverence/vibrant-maven-plugin/");
        DocumentSplitter splitter = DocumentSplitters.recursive(1000, 100);
        PathMatcher pathMatcher = FileSystems.getDefault().getPathMatcher("glob:**/*.java");
        List<Document> documents = FileSystemDocumentLoader.loadDocumentsRecursively(
                directoryPath, pathMatcher

        );
        List<Embedding> embeddings = new ArrayList<>();
        List<TextSegment> chunks = new ArrayList<>();

        for (Document d: documents){
            List<TextSegment> splits = splitter.split(d);
            for (TextSegment seg: splits){
                Response<Embedding> embed = embeddingModel.embed(seg);
                Embedding embeddingVector = embed.content();
                chunks.add(seg);
                embeddings.add(embeddingVector);
            }
        }
        vectorStore.addAll(embeddings, chunks);
        String query = "What classes Extend Mojo?";

        Embedding queryEmbedding = embeddingModel.embed(query).content();
        EmbeddingSearchRequest searchRequest = EmbeddingSearchRequest.builder()
                .queryEmbedding(queryEmbedding)
                .maxResults(3)
                .build();
        EmbeddingSearchResult<TextSegment> matches = vectorStore.search(searchRequest);
        StringBuilder sb = new StringBuilder();
        for (EmbeddingMatch<TextSegment> match : matches.matches()) {
            TextSegment segment = match.embedded();
            System.out.println("Embed id " + match.embeddingId());
            System.out.println(match.embedded().metadata());
            System.out.println("Relevance Score: " + match.score());
            System.out.println(segment.text().substring(0, Math.min(100, segment.text().length())) + "...\n");
            sb.append(match.embedded().metadata() + "\n");
            sb.append(segment.text());
        }

        // Create a prompt template for RAG
        String promptTemplate = "Answer the following question based on the provided context.\n\n"
                + "Context:\n%s\n\n"
                + "Question: %s";

        String prompt = String.format(promptTemplate, sb, query);
    }


}
