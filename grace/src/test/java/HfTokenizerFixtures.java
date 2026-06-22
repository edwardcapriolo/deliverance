final class HfTokenizerFixtures {
    static final String HF_COMMON_INPUT = "This is a test 😊\n"
            + "I was born in 92000, and this is falsé.\n"
            + "生活的真谛是\n"
            + "Hi  Hello\n"
            + "Hi   Hello\n"
            + "\n"
            + " \n"
            + "  \n"
            + " Hello\n"
            + "<s>\n"
            + "hi<s>there\n"
            + "The following string should be properly encoded: Hello.\n"
            + "But ird and ปี   ird   ด\n"
            + "Hey how are you doing";

    private HfTokenizerFixtures() {
    }

}
