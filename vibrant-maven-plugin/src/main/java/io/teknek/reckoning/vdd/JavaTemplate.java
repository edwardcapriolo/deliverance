package io.teknek.reckoning.vdd;

import io.teknek.reckoning.model.AbstractModel;
import io.teknek.reckoning.safetensors.prompt.PromptSupport;

import java.util.ArrayList;
import java.util.List;

public class JavaTemplate {
    private String modelName;
    private String modelOwner;
    private List<String> basePrompts;
    private String packageName;
    private List<String> requests;

    public JavaTemplate(String packageName, List<String> vddRequests){
        List<String> rootPrompts = List.of(
          "You are an assistant that produces concise, production-grade software.",
          "Generate Java code.",
          "Place and text that is not java code in comments.",
          "Structure responses as follows:\n-----start {file}\n{content}\n-----end {file} " +
          "where {file} is the name of the file and {content} is the generated code."
        );
        basePrompts = new ArrayList<>();
        basePrompts.addAll(rootPrompts);
        basePrompts.add("Generate code in the package "+ packageName);
        this.requests = vddRequests;
    }

    private void makeRequests(AbstractModel model){
        PromptSupport.Builder pc = model.promptSupport().get().builder();
        basePrompts.forEach(pc::addSystemMessage);
        requests.forEach(pc::addUserMessage);
    }

}
