package io.teknek.deliverance.model;

import io.teknek.deliverance.tokenizer.Tokenizer;

import java.util.*;
import java.util.stream.Collectors;

class ChoiceEncoded {
    private Map<String, List<Long>> encoded = new HashMap<>();

    public ChoiceEncoded(List<String> choices, Tokenizer tokenizer) {
        choices.forEach(choice -> {
            /*
            long[] ids = tokenizer.encode(choice);
            List<Long> longList = Arrays.stream(ids)
                    .boxed()
                    .collect(Collectors.toList());
            encoded.put(choice, longList);*/
            List<String> toknenized = tokenizer.tokenize(choice);
            ArrayList<Long> result = new ArrayList<>();
            for (String s: toknenized) {
                long[] z = tokenizer.encode(s);
                for (int i = 0; i < z.length; i++) {
                    result.add(z[i]);
                }
            }
            encoded.put(choice, result);

        });
    }

    public Map<String, List<Long>> getEncoded() {
        return encoded;
    }

    @Override
    public String toString() {
        return "ChoiceEncoded{" +
                "encoded=" + encoded +
                '}';
    }
}
