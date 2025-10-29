package io.teknek.deliverance;

import com.fasterxml.jackson.core.JsonGenerator;
import com.fasterxml.jackson.core.util.DefaultPrettyPrinter;
import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.MapperFeature;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.IOException;

public class JsonUtils {

    public static final ObjectMapper om = new ObjectMapper().configure(DeserializationFeature.FAIL_ON_IGNORED_PROPERTIES, false)
            .configure(DeserializationFeature.FAIL_ON_TRAILING_TOKENS, false)
            .configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false)
            .configure(DeserializationFeature.FAIL_ON_MISSING_CREATOR_PROPERTIES, false)
            .enable(MapperFeature.ACCEPT_CASE_INSENSITIVE_ENUMS);

    public static class JlamaPrettyPrinter extends DefaultPrettyPrinter {

        public static final JlamaPrettyPrinter INSTANCE = new JlamaPrettyPrinter();

        @Override
        public DefaultPrettyPrinter createInstance() {
            return INSTANCE;
        }

        private JlamaPrettyPrinter() {
            _objectIndenter = FixedSpaceIndenter.instance;
            _spacesInObjectEntries = false;
        }

        @Override
        public void beforeArrayValues(JsonGenerator jg) {}

        @Override
        public void writeEndArray(JsonGenerator jg, int nrOfValues) throws IOException {
            if (!this._arrayIndenter.isInline()) {
                --this._nesting;
            }
            jg.writeRaw(']');
        }

        @Override
        public void writeObjectFieldValueSeparator(JsonGenerator jg) throws IOException {
            jg.writeRaw(": ");
        }

        @Override
        public void beforeObjectEntries(JsonGenerator jg) {}

        @Override
        public void writeEndObject(JsonGenerator jg, int nrOfEntries) throws IOException {
            if (!this._objectIndenter.isInline()) {
                --this._nesting;
            }
            jg.writeRaw("}");
        }
    }
}
