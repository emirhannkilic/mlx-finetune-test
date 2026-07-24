from mlx_lm import load, generate

model, tokenizer = load("mlx-community/Josiefied-Qwen3-4B-abliterated-v1-4bit")

prompt = "hello"
messages = [{"role": "user", "content": prompt}]
prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

response = generate(model, tokenizer, prompt=prompt, verbose=True)