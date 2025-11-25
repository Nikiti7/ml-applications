from transformers import AutoModelForCausalLM, AutoTokenizer

# Берем GPT-2
model_name = "gpt2"

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Входной промпт
prompt = "In the future, artificial intelligence will"

# Токенизация
inputs = tokenizer(prompt, return_tensors="pt")

# Генерация текста
output = model.generate(
    **inputs,
    max_length=80,
    do_sample=True,  # случайная генерация
    top_p=0.95,  # nucleus sampling
    top_k=50,  # ограничение словаря
)

print("\n--- Generated text ---\n")
print(tokenizer.decode(output[0], skip_special_tokens=True))
