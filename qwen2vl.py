import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
import requests

# Load Qwen2-VL-2B model
model_name = "Qwen/Qwen2-VL-2B-Instruct"

model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_name)

# Load image
url = "https://github.com/sergiovillanueva/prompt_to_mask/raw/master/assets/image.jpg"
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

# Your question
question = "How many gray cars are there?"

print(f"Question: {question}")

# Create messages format
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": question}
        ]
    }
]

# Apply chat template
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# Process inputs
inputs = processor(
    text=[text],
    images=[image],
    return_tensors="pt"
).to("cuda")

# Generate answer
with torch.no_grad():
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False
    )

# Decode answer
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
answer = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)[0]

print(f"Answer: {answer}") 