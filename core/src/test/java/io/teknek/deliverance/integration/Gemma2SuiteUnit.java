package io.teknek.deliverance.integration;

import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.AutoModelForCausaLm;
import io.teknek.deliverance.model.ChoiceEncodedTest;
import io.teknek.deliverance.safetensors.fetch.ModelFetcher;
import org.junit.platform.suite.api.AfterSuite;
import org.junit.platform.suite.api.BeforeSuite;
import org.junit.platform.suite.api.SelectClasses;
import org.junit.platform.suite.api.Suite;

@Suite
@SelectClasses({ChoiceEncodedTest.class})
public class Gemma2SuiteUnit {

    private static volatile AbstractModel model;
    private static volatile AutoModelForCausaLm.Builder builder;

    public static AbstractModel getOrCreate(){
        if (model == null){
            ModelFetcher fetch = new ModelFetcher("tjake", "gemma-2-2b-it-JQ4");
            builder = AutoModelForCausaLm.newBuilder(fetch);
            model = builder.build();
        }
        return model;
    }

    public static AutoModelForCausaLm.Builder getBuilder(){
        return builder;
    }

    @BeforeSuite
    public static void beforeSuite(){
       getOrCreate();
    }

    @AfterSuite
    public static void afterSuite(){
        if (model != null){
            model.close();
        }
    }
}
