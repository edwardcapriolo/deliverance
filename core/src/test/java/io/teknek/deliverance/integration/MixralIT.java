package io.teknek.deliverance.integration;

import io.teknek.deliverance.generator.GeneratorParameters;
import io.teknek.deliverance.generator.Response;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.AutoModelForCausaLm;
import io.teknek.deliverance.model.GenerateEvent;
import io.teknek.deliverance.safetensors.fetch.ModelFetcher;
import io.teknek.deliverance.safetensors.prompt.PromptSupport;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;

import java.util.UUID;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class MixralIT {
    @Test
    public void chat(){
        ModelFetcher fetch = new ModelFetcher("tjake", "Mixtral-8x7B-Instruct-v0.1-JQ4");

        try (AbstractModel model = AutoModelForCausaLm.newBuilder(fetch).build()) {

boolean doAssert = true;

{
            String prompt = "What colors are in a rainbow?";                                          
            PromptSupport.Builder g = model.promptSupport().get().builder()                           
                    .addUserMessage(prompt); 

            var uuid = UUID.randomUUID();

            Response response = model.generate(uuid, g.build(), new GeneratorParameters()
                            .withNtokens(500).withMaxTokens(50).withTopP(0.05f),
                    new GenerateEvent() {
                        @Override
                        public void emit(int next, String nextRaw, String nextCleaned, float timing) {
                            System.out.println(nextCleaned);
                        }
                    });
            if (doAssert) {
            assertEquals("""
Here are some rainbow color definitions from Wikipedia:

Scientific definitions:

- ROYGBV (red, orange, yellow, green, blue, violet)
- ROYGBV (red, orange, yellow""".trim(), response.responseText.trim()); } else { 
    System.out.println(response.responseText);
  }
}

{                                                                                        
            String prompt = "Create a java class with a method that converts fahrenheit to celsius.";                                          
            PromptSupport.Builder g = model.promptSupport().get().builder()                           
                    .addUserMessage(prompt); 

            var uuid = UUID.randomUUID();                                                
                                                                                         
            Response response = model.generate(uuid, g.build(), new GeneratorParameters()
                            .withNtokens(500).withMaxTokens(50).withTopP(0.05f),         
                    new GenerateEvent() {                                                
                        @Override                                                        
                        public void emit(int next, String nextRaw, String nextCleaned, float timing) {
                            System.out.println(nextCleaned);                                          
                        }                                                                             
                    });                                                                               
            assertEquals("""
In this example, you can see a Java 11 class with a constructor that receives a `String` and a `Class` parameters.

```
public class Example {

    private String message;
    private Class<?""".trim(), response.responseText.trim());                               
}    

{                                                                                                                                
            String prompt = "Name 3 states in the United States of America.";                                                                     
            PromptSupport.Builder g = model.promptSupport().get().builder()                                                      
                    .addUserMessage(prompt);                                                                                     
                                                                                                                                 
            var uuid = UUID.randomUUID();                                                                                        
                                                                                                                                 
            Response response = model.generate(uuid, g.build(), new GeneratorParameters()                                        
                            .withNtokens(500).withMaxTokens(50).withTopP(0.05f),                                                 
                    new GenerateEvent() {                                                                                        
                        @Override                                                                                                
                        public void emit(int next, String nextRaw, String nextCleaned, float timing) {                           
                            System.out.println(nextCleaned);                                                                     
                        }                                                                                                        
                    });                                                                                                          
            assertEquals("""                                                                                                     
. . .

I'm not sure if this is the right place to post this, but I'm having a problem with my new computer.

I have a Dell Inspiron 154542,""".trim(), response.responseText.trim());                                                          
}     

{                                                                                                                 
            String prompt = "True or false? Batman wears a mask.";                                     
            PromptSupport.Builder g = model.promptSupport().get().builder()                                       
                    .addUserMessage(prompt);                                                                      
                                                                                                                  
            var uuid = UUID.randomUUID();                                                                         
                                                                                                                  
            Response response = model.generate(uuid, g.build(), new GeneratorParameters()                         
                            .withNtokens(500).withMaxTokens(70).withTopP(0.05f),                                  
                    new GenerateEvent() {                                                                         
                        @Override                                                                                 
                        public void emit(int next, String nextRaw, String nextCleaned, float timing) {            
                            System.out.println(nextCleaned);                                                      
                        }                                                                                         
                    });                                                                                           
            assertEquals("""
A) [F]

B) [T]

C) [F,T]

D) [I,F,T]

E) [I,F]

Answer:

The answer is [C] [I,F,T]

Explanation:""".trim(), response.responseText.trim());                                          
}       




        }
    }
}
