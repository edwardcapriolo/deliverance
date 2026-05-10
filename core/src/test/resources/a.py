from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("google/gemma-4-E2B-it")

messages = [
    {"role": "system", "content": "You are a concise assistant."},
    {"role": "user", "content": "What is the capital of New York?"},
]

rendered = tok.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False
)

print("HF_RENDERED_START")
print(rendered)
print("HF_RENDERED_END")
