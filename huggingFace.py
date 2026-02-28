# runs the model Qwen on an image and produces a description
import os
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()

client = InferenceClient(
    api_key=os.getenv("HF_TOKEN"),
)

completion = client.chat.completions.create(
    model="Qwen/Qwen3.5-397B-A17B:novita",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Describe this image in one sentence."
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
                    }
                }
            ]
        }
    ],
)

print(completion.choices[0].message)

# lists all models
from huggingface_hub import HfApi

api = HfApi()

# List models available for inference
models = api.list_models(
    inference_provider="all",   # IMPORTANT
    direction=-1,
    limit=1000
)

# filter by task type: text generation
from huggingface_hub import HfApi

api = HfApi()

# List models available for inference
models = api.list_models(
    inference_provider="hf-inference",   # IMPORTANT
    pipeline_tag="text-generation",      # task type
    direction=-1,
    limit=20
)

for model in models:
    print(model.modelId)

# lists all tasks that we can filter by
import requests

url = "https://huggingface.co/api/tasks"
tasks = requests.get(url).json()

for t in tasks:
    print(t)

