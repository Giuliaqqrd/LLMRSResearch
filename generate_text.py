from huggingface_hub import InferenceClient

client = InferenceClient(
    base_url="http://localhost:40900/v1/",
)

output = client.chat.completions.create(
    model="tgi",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Describe a cat"},
    ],
    stream=True,
    max_tokens=1024,
)

response_text = ""

for chunk in output:
    response_text += chunk.choices[0].delta.content

print(response_text)



