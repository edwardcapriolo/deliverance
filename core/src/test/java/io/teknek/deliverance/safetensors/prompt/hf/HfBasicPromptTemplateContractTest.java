package io.teknek.deliverance.safetensors.prompt.hf;

/**
 * Port of the dummy-template checks from Hugging Face TokenizerTesterMixin.test_chat_template.
 */
public class HfBasicPromptTemplateContractTest implements HfPromptTemplateContract {

    @Override
    public String hfDummyTemplate() {
        return "{% for message in messages %}{{message['role'] + message['content']}}{% endfor %}";
    }

    @Override
    public String hfDummyExpectedOutput() {
        return "systemsystem messageuseruser messageassistantassistant message";
    }
}
