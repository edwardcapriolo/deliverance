# grace

`grace` is Deliverance's tokenizer and prompt-processing module.

It is intended to be a fuller-featured alternative to the older lightweight tokenizer path that lives in `core`, especially for modern Hugging Face tokenizers and prompt templates.

## Why grace exists

Modern tokenizers need more than "text in, ids out".

Examples of behavior that matter in practice:

- added token vocabularies
- special token ids
- chat templates
- skip-special-token decoding
- decode cleanup
- left/right trim behavior on added tokens
- family-specific BPE behavior

The original tokenizer path in `core` is intentionally simple, but that simplicity becomes a liability for newer model families.

`grace` exists to model tokenizer behavior more closely to Hugging Face while still feeling natural in Java.

## What grace supports

Today `grace` includes support for:

- tokenizer config parsing
- added vocab and special token lookup
- chat template loading
- local and remote tokenizer fetches
- family-aware tokenizer dispatch
- Gemma-style prompt tokenization
- byte-level BPE tokenization for GPT/Qwen/Llama-style fast tokenizers

## Basic usage

### Load by owner and model name

```java
PreTrainedTokenizer tokenizer = AutoTokenizer.fromPretrained(
        new AutoTokenizer.OwnerNameOrPath(
                new AutoTokenizer.OwnerName("google", "gemma-4-E2B-it")
        )
);
```

### Encode and decode

```java
Encoding encoding = tokenizer.encode("Hello world");
int[] ids = encoding.inputIds();

String decoded = tokenizer.decode(
        new TokenIds(ids),
        false,
        false,
        false,
        false
);
```

### Inspect special tokens

```java
tokenizer.specialTokensMap();
tokenizer.allSpecialTokens();
tokenizer.allSpecialIds();
tokenizer.bosTokenId();
tokenizer.eosTokenId();
```

## Fetching behavior

`TokenizerModelFetcher` uses the same cache root as model fetching:

- `~/.deliverance/<owner>_<model>/`

but downloads only tokenizer-related assets such as:

- `tokenizer.json`
- `tokenizer_config.json`
- `vocab.json`
- `merges.txt`
- `chat_template.jinja`
- `config.json`

This means:
- no duplicate tokenizer cache tree
- tokenizer-only fetches avoid model weights
- local directories can be reused offline if the required files are already present

## Family dispatch

One of the recent cleanups in `grace` was to stop forcing multiple tokenizer families through one implementation.

The factory now distinguishes between families such as:

- Gemma-style tokenizers
- byte-level BPE tokenizers
- Qwen tokenizer path
- Bert tokenizer path

This matters because the same raw `tokenizer_class` string is often not enough to decide how tokenization should behave.

## Tests

`grace` now has focused tests for:

- Hugging Face Gemma prompt golden token ids
- merge-array parsing
- regex pretokenizer gap preservation
- Gemma turn-token atomicity
- Gemma decode cleanup

If you are extending `grace`, add small focused tests for each tokenizer-family-specific bug you fix.

## When to use grace

Use `grace` when you need:

- Hugging Face-like tokenizer behavior
- chat template parity
- modern added/special token handling
- a stronger tokenizer oracle than the lightweight `core` tokenizer path

This is especially important for model families like Gemma 4, where prompt-control tokenization has a major effect on generation quality.
