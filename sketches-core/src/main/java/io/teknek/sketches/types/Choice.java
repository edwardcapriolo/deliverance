package io.teknek.sketches.types;

import java.util.List;
import java.util.Objects;

public final class Choice extends Term {
    private final List<String> items;

    public Choice(List<String> items) {
        Objects.requireNonNull(items, "items");
        if (items.isEmpty()) {
            throw new IllegalArgumentException("Choice items must not be empty");
        }
        this.items = List.copyOf(items);
    }

    public List<String> items() {
        return items;
    }
}
