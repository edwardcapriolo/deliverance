package io.teknek.deliverance.grace;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import io.teknek.deliverance.model.qwen2.Qwen2Tokenizer;
import io.teknek.deliverance.safetensors.fetch.ModelFetcher;

import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

public class AutoTokenizer {

    private static ConcurrentHashMap<String, Class<PreTrainedTokenizerBase>> registry = new ConcurrentHashMap<>();
    static {
        //registry.put("BERT", null);
        //registry.put("QWEN2", null);
    }
    public static PreTrainedTokenizer fromPretrained(OwnerNameOrPath ownerNameOrPath,
                                       java.util.Optional<String> tokenizerType,
                                       java.util.Optional<java.util.List<Object>> inputs,
                                       Map<String, ?> tokenizerInitArgs){
        File path = null;
        if (ownerNameOrPath.ownerName != null){
            TokenizerModelFetcher mf = new TokenizerModelFetcher(ownerNameOrPath.ownerName.owner, ownerNameOrPath.ownerName.name);
            path = mf.maybeDownload();
        }
        ObjectMapper om = new ObjectMapper();



        String version = null;
        try {
            Map<String, Integer> map = om.readValue( new File(path, "vocab.json"), new TypeReference<Map<String, Integer>>() {});


            JsonNode document = om.readTree(new File(path,"tokenizer.json"));
            JsonNode versionNode = document.get("version");
            if (!versionNode.isNull()){
                version = versionNode.asText();
            }
            JsonNode addedTokens = document.get("added_tokens");
            SortedMap<Integer, AddedToken> addedTokenMap = new TreeMap<>();
            if (!addedTokens.isNull()){
                if (addedTokens.isArray()){
                    for (JsonNode addedToken : addedTokens){

                        int id = addedToken.get("id").asInt();
                        String content = addedToken.get("content").asText();
                        boolean singleWord = addedToken.get("single_word").asBoolean();
                        boolean lstrip = addedToken.get("lstrip").asBoolean();
                        boolean rstrip = addedToken.get("rstrip").asBoolean();
                        boolean special = addedToken.get("special").asBoolean();
                        JsonNode normalized =addedTokens.get("normalized");
                        AddedToken a = new AddedToken(content, singleWord, lstrip, rstrip, special,
                                normalized == null || normalized.isNull()? null: normalized.asBoolean());
                        addedTokenMap.put(id, a);
                    }
                }
            }
            Quen2Tokenizer q = new Quen2Tokenizer(new HashMap<>(), Optional.empty(),Optional.empty(),Optional.empty(),
                    Optional.empty(),Optional.empty(),Optional.empty(),Optional.empty(),
                    map, addedTokenMap
                    );
            return q;

        } catch (JsonProcessingException e) {
            throw new RuntimeException(e);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        /*
        Class<PreTrainedTokenizerBase> clazz =  null;
        if (tokenizerType.isPresent()){
            clazz = registry.get(tokenizerType.get());
            if (clazz == null){
                throw new RuntimeException("The specified tokenizer_type is not found: " + tokenizerType.get()
                        + " in registry: " + registry);
            }
        }*/
        //return null;

    }

    public static class OwnerName{
        String owner;
        String name;
        public OwnerName(String owner, String name){
            this.owner = owner;
            this.name = name;
        }
    }
    static class ModelPath {
        Path path;
    }
    public static class OwnerNameOrPath{
        private final OwnerName ownerName;
        private final ModelPath modelPath;
        public OwnerNameOrPath(OwnerName ownerName){
            this.ownerName = ownerName;
            this.modelPath = null;
        }

        public OwnerNameOrPath(ModelPath modelPath){
            this.modelPath = modelPath;
            this.ownerName = null;
        }
    }
}
