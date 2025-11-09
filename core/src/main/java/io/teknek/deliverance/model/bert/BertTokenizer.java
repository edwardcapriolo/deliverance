package io.teknek.deliverance.model.bert;

import io.teknek.deliverance.tokenizer.WordPieceTokenizer;

import java.nio.file.Path;

public class BertTokenizer extends WordPieceTokenizer {

        public BertTokenizer(Path modelRoot) {
            super(modelRoot);
        }

}
