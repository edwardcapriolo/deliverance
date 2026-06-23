# Semantic Search & Embeddings

Deliverance supports embedding models for semantic search, information retrieval, and code understanding. The [LEAF model](https://huggingface.co/MongoDB/mdbr-leaf-ir) is a compact, efficient embedding model optimized for information retrieval tasks, including semantic code search, RAG applications, and understanding codebases semantically.

## Use Cases

- **Semantic Code Search**: Find code by meaning, not just keywords, such as "find all database connection methods".
- **Code Understanding**: Understand relationships between classes, methods, and concepts in large codebases.
- **RAG Applications**: Build retrieval-augmented generation systems for code documentation and knowledge bases.
- **Information Retrieval**: Search semantically across documentation, code comments, and technical content.

```java
import com.codahale.metrics.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.embedding.PoolingType;
import io.teknek.deliverance.math.VectorMathUtils;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.ModelSupport;
import io.teknek.deliverance.safetensors.fetch.ModelFetcher;
import io.teknek.deliverance.tensor.KvBufferCacheSettings;
import io.teknek.deliverance.tensor.ArrayQueueTensorAllocator;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;
import java.io.File;

public void semanticCodeSearch() {
    String modelOwner = "MongoDB";
    String modelName = "mdbr-leaf-ir";

    // Download and load the LEAF embedding model
    ModelFetcher fetch = new ModelFetcher(modelOwner, modelName);
    File localModelPath = fetch.maybeDownload();
    MetricRegistry mr = new MetricRegistry();
    TensorCache arrayQueueTensorAllocator = new TensorCache(mr);
    AbstractModel embeddingModel = ModelSupport.loadEmbeddingModel(localModelPath, DType.F32, DType.F32,
            new ConfigurableTensorProvider(arrayQueueTensorAllocator), mr, arrayQueueTensorAllocator, new KvBufferCacheSettings(true));

    // Embed code snippets or documentation
    String query = "database connection initialization";
    String[] codeSnippets = {
        "public class DatabaseConnection { private Connection conn; ... }",
        "public void connectToDatabase(String url) { ... }",
        "public class UserService { public void authenticate() { ... } }",
        "Connection conn = DriverManager.getConnection(url, user, pass);"
    };

    // Generate embeddings
    float[] queryEmbedding = embeddingModel.embed(query, PoolingType.AVG);

    // Find most similar code snippet
    float maxSimilarity = -1.0f;
    String bestMatch = "";
    for (String snippet : codeSnippets) {
        float[] snippetEmbedding = embeddingModel.embed(snippet, PoolingType.AVG);
        float similarity = VectorMathUtils.cosineSimilarity(queryEmbedding, snippetEmbedding);
        if (similarity > maxSimilarity) {
            maxSimilarity = similarity;
            bestMatch = snippet;
        }
    }

    System.out.println("Best match: " + bestMatch + " (similarity: " + maxSimilarity + ")");
    embeddingModel.close();
}
```

## Building A Semantic Code Index

For tools that need to understand code semantically, you can use LEAF embeddings to:

1. **Index codebase**: Generate embeddings for classes, methods, and documentation.
2. **Semantic search**: Find relevant code by meaning, not just text matching.
3. **Context retrieval**: Retrieve semantically similar code for LLM context.
4. **Code understanding**: Understand relationships and patterns across large codebases.

The LEAF model's compact size, 23M parameters and 384 dimensions, makes it useful for production IDE and code-analysis workflows where low latency and memory efficiency matter.

See `core/src/main/java/io/teknek/deliverance/examples/LeafModelExample.java` for a complete example with CLI flags for normalization, batch size, and parallel processing.
