package io.teknek.deliverance.generator;


public enum FinishReason {
    /** if the maximum number of tokens specified in the request was reached */
    MAX_TOKENS("length"),
    /** if the model hit a natural stop point or a provided stop sequence */
    STOP_TOKEN("stop"),
    /** the model wants to call a tool */
    TOOL_CALLS("tool_calls"),
    //deprecated
    FUNCTION_CALL("function_call"),
    //unused
    CONTENT_FILTER("content_filter");

    private final String externalMapping;
    private FinishReason(String s){
        externalMapping = s;
    }

    public String getExternalMapping(){
        return externalMapping;
    }
}