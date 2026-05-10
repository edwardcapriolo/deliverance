package io.teknek.deliverance.model;

import java.util.*;

class ChoiceEncoded {
    private Map<String, List<Long>> encoded = new HashMap<>();

    public ChoiceEncoded(List<String> choices, AbstractModel model) {
        choices.forEach(choice -> {
            long[] ids = model.encodeText(choice);
            ArrayList<Long> result = new ArrayList<>(ids.length);
            for (long id : ids) {
                result.add(id);
            }
            encoded.put(choice, result);
        });
    }

    public Map<String, List<Long>> getEncoded() {
        return encoded;
    }

    public boolean anyStartsWith(List<Integer> prefix) {
        for (List<Long> choice : encoded.values()) {
            if (choice.size() < prefix.size()) {
                continue;
            }
            boolean matches = true;
            for (int i = 0; i < prefix.size(); i++) {
                if (choice.get(i).intValue() != prefix.get(i)) {
                    matches = false;
                    break;
                }
            }
            if (matches) {
                return true;
            }
        }
        return false;
    }

    @Override
    public String toString() {
        return "ChoiceEncoded{" +
                "encoded=" + encoded +
                '}';
    }
}
