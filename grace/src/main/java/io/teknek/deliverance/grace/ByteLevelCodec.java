package io.teknek.deliverance.grace;

import java.io.ByteArrayOutputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

final class ByteLevelCodec {
    private static final Map<Integer, Integer> BYTE_TO_CODEPOINT = buildByteToCodepoint();
    private static final Map<Integer, Integer> CODEPOINT_TO_BYTE = buildCodepointToByte();

    private ByteLevelCodec() {
    }

    static String decode(String value) {
        ByteArrayOutputStream output = new ByteArrayOutputStream();
        value.codePoints().forEach(codePoint -> {
            Integer decoded = CODEPOINT_TO_BYTE.get(codePoint);
            if (decoded != null) {
                output.write(decoded);
                return;
            }

            byte[] bytes = new String(Character.toChars(codePoint)).getBytes(StandardCharsets.UTF_8);
            output.writeBytes(bytes);
        });
        return output.toString(StandardCharsets.UTF_8);
    }

    static String encode(String value) {
        StringBuilder encoded = new StringBuilder();
        byte[] bytes = value.getBytes(StandardCharsets.UTF_8);
        for (byte current : bytes) {
            int unsigned = Byte.toUnsignedInt(current);
            encoded.appendCodePoint(BYTE_TO_CODEPOINT.get(unsigned));
        }
        return encoded.toString();
    }

    private static Map<Integer, Integer> buildByteToCodepoint() {
        List<Integer> bytes = new ArrayList<>();
        for (int value = '!'; value <= '~'; value++) {
            bytes.add(value);
        }
        for (int value = 161; value <= 172; value++) {
            bytes.add(value);
        }
        for (int value = 174; value <= 255; value++) {
            bytes.add(value);
        }

        List<Integer> codePoints = new ArrayList<>(bytes);
        int offset = 0;
        for (int value = 0; value < 256; value++) {
            if (!bytes.contains(value)) {
                bytes.add(value);
                codePoints.add(256 + offset);
                offset++;
            }
        }

        Map<Integer, Integer> mapping = new HashMap<>(bytes.size());
        for (int index = 0; index < bytes.size(); index++) {
            mapping.put(bytes.get(index), codePoints.get(index));
        }
        return mapping;
    }

    private static Map<Integer, Integer> buildCodepointToByte() {
        Map<Integer, Integer> mapping = new HashMap<>(BYTE_TO_CODEPOINT.size());
        for (Map.Entry<Integer, Integer> entry : BYTE_TO_CODEPOINT.entrySet()) {
            mapping.put(entry.getValue(), entry.getKey());
        }
        return mapping;
    }
}
