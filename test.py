# # runs the model Qwen on an image and produces a description
import os
from huggingface_hub import InferenceClient, HfApi
from dotenv import load_dotenv

load_dotenv()

client = InferenceClient(
    api_key=os.getenv("HF_TOKEN"),
)
api = HfApi(token=os.getenv("HF_TOKEN"))

# List models available for inference
models = api.list_models(
    inference_provider="all",   # IMPORTANT
    pipeline_tag="text-generation",      # task type
    direction=-1,
    limit=10
)

list_model = list(models)
for model in list_model:
    print(model.modelId)

try:
    completion = client.chat.completions.create(
        model=f"{list_model[9].modelId}",
        messages=[
            {
                "role": "user",
                "content": "what is the capital of France?"
            }
        ],
    )
except Exception as e:
    print("trying featherless-ai")
    completion = client.chat.completions.create(
        model=f"{list_model[1].modelId}:featherless-ai",
        messages=[
            {
                "role": "user",
                "content": "what is the capital of France?"
            }
        ],
    )

print(completion.choices[0].message)


