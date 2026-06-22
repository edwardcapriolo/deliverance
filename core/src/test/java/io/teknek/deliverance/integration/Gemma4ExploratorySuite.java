package io.teknek.deliverance.integration;

import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.AutoModelForCausaLm;
import io.teknek.deliverance.safetensors.fetch.ModelFetcher;
import org.junit.platform.suite.api.AfterSuite;
import org.junit.platform.suite.api.SelectClasses;
import org.junit.platform.suite.api.Suite;

@Suite
@SelectClasses({Gemma4ExploratoryIT.class})
public class Gemma4ExploratorySuite {
    private static volatile AbstractModel model;
    private static volatile AutoModelForCausaLm.Builder builder;

    public static AbstractModel getOrCreate() {
        if (model == null) {
            ModelFetcher fetch = new ModelFetcher("edward", "gemma-4-E2B-it-JQ4");
            builder = AutoModelForCausaLm.newBuilder(fetch);
            model = builder.buildLocalTransformerModel();
        }
        return model;
    }

    public static AutoModelForCausaLm.Builder getBuilder() {
        return builder;
    }

    @AfterSuite
    public static void afterSuite() {
        if (model != null) {
            model.close();
            model = null;
        }
    }
}
