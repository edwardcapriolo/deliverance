package io.teknek.deliverance;

import io.teknek.deliverance.tokenizer.TokenizerModel;
import org.junit.jupiter.api.Test;

import java.util.regex.Pattern;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

public class TokenizerModelTest {

    @Test
    void letsSeeWhatWeGet(){
        String [] x = TokenizerModel.split(Pattern.compile(","),
                """
                        I would like to order 7,000,000 doughnuts, to feed my
                        """.stripTrailing(), 5, true);
        assertArrayEquals(new String[]{
                "I would like to order 7",
                ",",
                "000",
                ",",
                "000 doughnuts",
                ",",
                " to feed my"}, x );
    }

    @Test
    void includeTokens(){
        String [] x = TokenizerModel.split(Pattern.compile(","),
                """
                        I would like to order 7,000,000 doughnuts, to feed my
                        """.stripTrailing(), 5, false);
        assertArrayEquals(new String[]{
                "I would like to order 7",
                "000",
                "000 doughnuts",
                " to feed my"}, x);
    }

}
