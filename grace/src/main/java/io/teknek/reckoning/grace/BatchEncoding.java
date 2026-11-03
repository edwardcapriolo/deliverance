package io.teknek.reckoning.grace;

import java.util.List;

public record BatchEncoding(List<Encoding> encodings) {
    public BatchEncoding {
        encodings = List.copyOf(encodings);
    }
}
