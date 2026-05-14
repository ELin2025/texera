/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.texera.amber.operator.huggingFace

import com.fasterxml.jackson.annotation.{JsonIgnore, JsonProperty, JsonPropertyDescription}
import com.kjetland.jackson.jsonSchema.annotations.JsonSchemaTitle
import org.apache.texera.amber.core.tuple.{AttributeType, Schema}
import org.apache.texera.amber.core.workflow.{InputPort, OutputPort, PortIdentity}
import org.apache.texera.amber.operator.PythonOperatorDescriptor
import org.apache.texera.amber.operator.metadata.annotations.AutofillAttributeName
import org.apache.texera.amber.operator.metadata.{OperatorGroupConstants, OperatorInfo}

class HuggingFaceInferenceOpDesc extends PythonOperatorDescriptor {

  private val imageOnlyTasks = Set(
    "image-classification",
    "object-detection",
    "image-segmentation",
    "image-to-text"
  )

  private val imagePromptTasks = Set(
    "visual-question-answering",
    "document-question-answering",
    "zero-shot-image-classification",
    "image-text-to-text",
    "image-to-image"
  )

  private val audioOnlyTasks = Set(
    "automatic-speech-recognition",
    "audio-classification"
  )

  @JsonProperty(value = "hfApiToken", required = true)
  @JsonSchemaTitle("HF API Token")
  @JsonPropertyDescription("Your Hugging Face API token (from https://huggingface.co/settings/tokens)")
  var hfApiToken: String = ""

  @JsonProperty(value = "task", required = true, defaultValue = "text-generation")
  @JsonSchemaTitle("Task")
  @JsonPropertyDescription("The Hugging Face pipeline task type")
  var task: String = "text-generation"

  @JsonProperty(
    value = "modelId",
    required = true,
    defaultValue = "Qwen/Qwen2.5-72B-Instruct"
  )
  @JsonSchemaTitle("Tasks")
  @JsonPropertyDescription(
    "Select a Hugging Face model"
  )
  var modelId: String = "Qwen/Qwen2.5-72B-Instruct"

  @JsonProperty(value = "promptColumn", required = false)
  @JsonSchemaTitle("Prompt Column")
  @JsonPropertyDescription("Column in the input table to use as the user prompt")
  @AutofillAttributeName
  var promptColumn: String = ""

  @JsonProperty(value = "imageInput", required = false)
  @JsonSchemaTitle("Image Upload")
  @JsonPropertyDescription("Upload an image for Hugging Face image tasks")
  var imageInput: String = ""

  @JsonProperty(value = "inputImageColumn", required = false)
  @JsonSchemaTitle("Input Image Column")
  @JsonPropertyDescription("Column containing image data from the input table")
  @AutofillAttributeName
  var inputImageColumn: String = ""

  @JsonProperty(value = "inputAudioColumn", required = false)
  @JsonSchemaTitle("Input Audio Column")
  @JsonPropertyDescription("Column containing audio data from the input table")
  @AutofillAttributeName
  var inputAudioColumn: String = ""

  @JsonProperty(value = "audioInput", required = false)
  @JsonSchemaTitle("Audio Upload")
  @JsonPropertyDescription("Upload audio for Hugging Face audio tasks")
  var audioInput: String = ""

  @JsonProperty(
    value = "systemPrompt",
    required = false,
    defaultValue = "You are a helpful assistant."
  )
  @JsonSchemaTitle("System Prompt")
  @JsonPropertyDescription("Optional system message to set model behavior")
  var systemPrompt: String = "You are a helpful assistant."

  @JsonProperty(value = "maxNewTokens", required = false, defaultValue = "256")
  @JsonSchemaTitle("Max New Tokens")
  @JsonPropertyDescription("Maximum number of tokens to generate (1-4096)")
  var maxNewTokens: java.lang.Integer = 256

  @JsonProperty(value = "temperature", required = false)
  @JsonSchemaTitle("Temperature")
  @JsonPropertyDescription("Sampling temperature (0.0 = deterministic, up to 2.0)")
  var temperature: java.lang.Double = 0.7

  @JsonProperty(
    value = "resultColumn",
    required = false,
    defaultValue = "hf_response"
  )
  @JsonSchemaTitle("Result Column Name")
  @JsonPropertyDescription("Name of the new column added to the output table")
  var resultColumn: String = "hf_response"

  // ── Group 2 fields ──

  @JsonProperty(value = "contextColumn", required = false)
  @JsonSchemaTitle("Context Column")
  @JsonPropertyDescription("Column containing the context passage (for Question Answering)")
  @AutofillAttributeName
  var contextColumn: String = ""

  // ── Group 3 fields ──

  @JsonProperty(value = "candidateLabels", required = false)
  @JsonSchemaTitle("Candidate Labels")
  @JsonPropertyDescription("Comma-separated candidate labels (for Zero-Shot Classification)")
  var candidateLabels: String = ""

  @JsonProperty(value = "sentencesColumn", required = false)
  @JsonSchemaTitle("Sentences Column")
  @JsonPropertyDescription(
    "Column with comma-separated sentences to compare (for Sentence Similarity / Text Ranking)"
  )
  @AutofillAttributeName
  var sentencesColumn: String = ""

  override def generatePythonCode(): String = {
    val safeTask = if (task == null || task.trim.isEmpty) "text-generation" else task
    val requiresPromptColumn =
      !imageOnlyTasks.contains(safeTask) &&
        !imagePromptTasks.contains(safeTask) &&
        !audioOnlyTasks.contains(safeTask)

    if (requiresPromptColumn) {
      assert(
        promptColumn != null && promptColumn.trim.nonEmpty,
        "Prompt Column must not be empty"
      )
    }
    assert(
      modelId != null && modelId.trim.nonEmpty,
      "Model ID must not be empty"
    )

    val pyToken = escapePython(hfApiToken)
    val pyModelId = escapePython(modelId)
    val pyPromptCol = escapePython(promptColumn)
    val pyResultCol = escapePython(
      if (resultColumn == null || resultColumn.trim.isEmpty) "hf_response" else resultColumn
    )

    val safeMaxTokens = math.max(1, math.min(if (maxNewTokens != null) maxNewTokens.intValue else 256, 4096))
    val safeTemp = math.max(0.0, math.min(if (temperature != null) temperature.doubleValue else 0.7, 2.0))
    val pySystemPrompt = escapePython(systemPrompt)
    val pyContextCol = escapePython(contextColumn)
    val pyCandidateLabels = escapePython(candidateLabels)
    val pySentencesCol = escapePython(sentencesColumn)
    val pyImageInput = escapePython(imageInput)
    val pyAudioInput = escapePython(audioInput)
    val pyInputImageColumn = escapePython(inputImageColumn)
    val pyInputAudioColumn = escapePython(inputAudioColumn)
    generateInferencePython(
      pyToken, pyModelId, pyPromptCol, pyResultCol,
      escapePython(safeTask), pySystemPrompt, safeMaxTokens, safeTemp,
      pyContextCol, pyCandidateLabels, pySentencesCol,
      pyImageInput, pyAudioInput, pyInputImageColumn, pyInputAudioColumn
    )
  }

  private def generateInferencePython(
      pyToken: String,
      pyModelId: String,
      pyPromptCol: String,
      pyResultCol: String,
      pyTask: String,
      pySystemPrompt: String,
      safeMaxTokens: Int,
      safeTemp: Double,
      pyContextCol: String,
      pyCandidateLabels: String,
      pySentencesCol: String,
      pyImageInput: String,
      pyAudioInput: String,
      pyInputImageColumn: String,
      pyInputAudioColumn: String
  ): String = {
    s"""import os
       |import re
       |import json
       |import base64
       |import requests
       |import pandas as pd
       |from urllib.parse import urlparse
       |from pytexera import *
       |
       |class ProcessTableOperator(UDFTableOperator):
       |
       |    # ---- configuration injected at code-generation time ----
       |    HF_API_TOKEN      = "$pyToken"
       |    MODEL_ID          = "$pyModelId"
       |    PROMPT_COLUMN     = "$pyPromptCol"
       |    RESULT_COLUMN     = "$pyResultCol"
       |    TASK              = "$pyTask"
       |    CONTEXT_COLUMN    = "$pyContextCol"
       |    CANDIDATE_LABELS  = "$pyCandidateLabels"
       |    SENTENCES_COLUMN  = "$pySentencesCol"
       |    IMAGE_INPUT        = "$pyImageInput"
       |    AUDIO_INPUT        = "$pyAudioInput"
       |    INPUT_IMAGE_COLUMN = "$pyInputImageColumn"
       |    INPUT_AUDIO_COLUMN = "$pyInputAudioColumn"
       |    SYSTEM_PROMPT  = "$pySystemPrompt"
       |    MAX_NEW_TOKENS = $safeMaxTokens
       |    TEMPERATURE    = $safeTemp
       |
       |    # Providers ranked cheapest-first (lower index = cheaper).
       |    # Unknown providers are appended at the end.
       |    PROVIDER_COST_PRIORITY = [
       |        "hf-inference",
       |        "cerebras",
       |        "sambanova",
       |        "groq",
       |        "novita",
       |        "nebius",
       |        "fireworks-ai",
       |        "together",
       |        "hyperbolic",
       |        "scaleway",
       |        "nscale",
       |        "ovhcloud",
       |        "deepinfra",
       |        "featherless-ai",
       |        "baseten",
       |        "publicai",
       |        "nvidia",
       |        "openai",
       |        "replicate",
       |        "fal-ai",
       |        "black-forest-labs",
       |        "wavespeed",
       |        "cohere",
       |        "clarifai",
       |    ]
       |
       |    def _resolve_providers(self, token):
       |        \"\"\"Query the HF Hub API to get available inference providers for this model.
       |        Returns a list of dicts with 'name' and 'providerId' sorted cheapest-first;
       |        falls back to hf-inference.
       |        \"\"\"
       |        try:
       |            resp = requests.get(
       |                f"https://huggingface.co/api/models/{self.MODEL_ID}",
       |                headers={"Authorization": f"Bearer {token}"},
       |                params={"expand[]": "inferenceProviderMapping"},
       |                timeout=30,
       |            )
       |            if resp.status_code == 200:
       |                data = resp.json()
       |                mapping = (
       |                    data.get("inferenceProviderMapping")
       |                    or data.get("inference_provider_mapping")
       |                    or {}
       |                )
       |                if mapping:
       |                    live = [
       |                        {
       |                            "name": p,
       |                            "providerId": v.get("providerId", self.MODEL_ID),
       |                            "task": v.get("task", ""),
       |                            "isModelAuthor": v.get("isModelAuthor", False),
       |                        }
       |                        for p, v in mapping.items()
       |                        if isinstance(v, dict) and v.get("status") == "live"
       |                    ]
       |                    if live:
       |                        priority = {name: idx for idx, name in enumerate(self.PROVIDER_COST_PRIORITY)}
       |                        live.sort(key=lambda prov: priority.get(prov["name"], len(self.PROVIDER_COST_PRIORITY)))
       |                        return live
       |        except Exception:
       |            pass
       |        return [{"name": "hf-inference", "providerId": self.MODEL_ID}]
       |
       |    def _post_with_fallback(self, providers, json_headers, raw_binary_headers, pipeline_payload, use_raw_binary_body, prompt_value):
       |        \"\"\"Try providers in order, using the correct API format for each.
       |        Returns (response, provider_summary) tuple.
       |        provider_summary is None on success, or a string describing what failed.
       |        \"\"\"
       |        RETRYABLE = (400, 404, 422, 429, 502, 503)
       |        last_resp = None
       |        errors = []
       |        for prov in providers:
       |            provider_name = prov["name"]
       |            provider_id = prov["providerId"]
       |            is_model_author = prov.get("isModelAuthor", False)
       |            prov_task = prov.get("task", "")
       |            try:
       |                if self.TASK in ("text-generation", "image-text-to-text"):
       |                    chat_routes = {
       |                        "groq": "openai/v1/chat/completions",
       |                        "fireworks-ai": "inference/v1/chat/completions",
       |                        "cohere": "compatibility/v1/chat/completions",
       |                        "clarifai": "v2/ext/openai/v1/chat/completions",
       |                        "deepinfra": "v1/openai/chat/completions",
       |                    }
       |                    route = chat_routes.get(provider_name, "v1/chat/completions")
       |                    url = f"https://router.huggingface.co/{provider_name}/{route}"
       |                    resp = requests.post(url, headers=json_headers, json=pipeline_payload, timeout=120)
       |                elif is_model_author and prov_task in ("image-to-text", "image-text-to-text") and provider_name not in ("zai-org",):
       |                    # Model-author vision providers use chat completions with base64 image
       |                    url = f"https://router.huggingface.co/{provider_name}/v1/chat/completions"
       |                    img_b64 = ""
       |                    if use_raw_binary_body and isinstance(pipeline_payload, bytes):
       |                        img_b64 = base64.b64encode(pipeline_payload).decode("utf-8")
       |                    chat_payload = {
       |                        "model": provider_id,
       |                        "messages": [{
       |                            "role": "user",
       |                            "content": [
       |                                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}} if img_b64 else None,
       |                                {"type": "text", "text": prompt_value if prompt_value else "What is in this image?"},
       |                            ],
       |                        }],
       |                    }
       |                    # Remove None entries
       |                    chat_payload["messages"][0]["content"] = [c for c in chat_payload["messages"][0]["content"] if c is not None]
       |                    resp = requests.post(url, headers=json_headers, json=chat_payload, timeout=120)
       |                elif provider_name == "hf-inference":
       |                    url = f"https://router.huggingface.co/hf-inference/models/{self.MODEL_ID}"
       |                    if use_raw_binary_body:
       |                        resp = requests.post(url, headers=raw_binary_headers, data=pipeline_payload, timeout=120)
       |                    else:
       |                        resp = requests.post(url, headers=json_headers, json=pipeline_payload, timeout=120)
       |                else:
       |                    # Provider-specific routing via native API format
       |                    resp = self._call_provider(provider_name, provider_id, json_headers, raw_binary_headers, pipeline_payload, use_raw_binary_body, prompt_value)
       |            except Exception as e:
       |                errors.append(f"{provider_name}: {type(e).__name__}")
       |                continue
       |            if resp.status_code in (200, 201):
       |                return resp, None
       |            if resp.status_code == 401:
       |                return resp, None
       |            try:
       |                detail = resp.json().get("error", resp.text[:200])
       |            except Exception:
       |                detail = resp.text[:200] if resp.text else "no details"
       |            errors.append(f"{provider_name}: HTTP {resp.status_code} - {detail}")
       |            last_resp = resp
       |            if resp.status_code not in RETRYABLE:
       |                return resp, "; ".join(errors)
       |        summary = "; ".join(errors) if errors else "no providers available"
       |        return last_resp, summary
       |
       |    def _call_provider(self, provider_name, provider_id, json_headers, raw_binary_headers, pipeline_payload, use_raw_binary_body, prompt_value):
       |        \"\"\"Route request to a third-party provider using its native API format.\"\"\"
       |        base = f"https://router.huggingface.co/{provider_name}"
       |        task = self.TASK
       |        img_b64 = ""
       |        if use_raw_binary_body and isinstance(pipeline_payload, bytes):
       |            img_b64 = base64.b64encode(pipeline_payload).decode("utf-8")
       |
       |        # ── zai-org ──
       |        # Custom API at /api/paas/v4/...
       |        # image-to-text: POST /api/paas/v4/layout_parsing  {"model": id, "file": "data:image/...;base64,..."}
       |        # chat:          POST /api/paas/v4/chat/completions  {model, messages}
       |        if provider_name == "zai-org":
       |            zai_headers = {**json_headers, "x-source-channel": "hugging_face", "accept-language": "en-US,en"}
       |            if task in ("image-to-text", "image-text-to-text"):
       |                url = f"{base}/api/paas/v4/layout_parsing"
       |                file_data = f"data:image/png;base64,{img_b64}" if img_b64 else ""
       |                return requests.post(url, headers=zai_headers, json={"model": provider_id, "file": file_data}, timeout=120)
       |            else:
       |                url = f"{base}/api/paas/v4/chat/completions"
       |                messages = [{"role": "user", "content": prompt_value}]
       |                if img_b64:
       |                    messages = [{"role": "user", "content": [
       |                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
       |                        {"type": "text", "text": prompt_value if prompt_value else "What is in this image?"},
       |                    ]}]
       |                return requests.post(url, headers=zai_headers, json={"model": provider_id, "messages": messages}, timeout=120)
       |
       |        # ── Replicate ──
       |        # All tasks: POST /v1/models/{providerId}/predictions  + Prefer: wait
       |        # Payload: {"input": {<task-specific fields>}}
       |        # If the model is too slow for sync, Replicate returns 202 with a polling URL.
       |        if provider_name == "replicate":
       |            url = f"{base}/v1/models/{provider_id}/predictions"
       |            hdrs = {**json_headers, "Prefer": "wait"}
       |            inp = {}
       |            if task == "text-to-speech":
       |                inp = {"text": prompt_value}
       |            elif task in ("text-to-image", "text-to-video"):
       |                inp = {"prompt": prompt_value}
       |            elif task == "automatic-speech-recognition" and img_b64:
       |                inp = {"audio": f"data:audio/wav;base64,{img_b64}"}
       |            elif task == "image-to-image" and img_b64:
       |                data_url = f"data:image/png;base64,{img_b64}"
       |                inp = {"image": data_url, "images": [data_url], "input_image": data_url, "prompt": prompt_value}
       |            elif img_b64:
       |                inp = {"image": f"data:image/png;base64,{img_b64}", "prompt": prompt_value}
       |            else:
       |                inp = {"prompt": prompt_value}
       |            resp = requests.post(url, headers=hdrs, json={"input": inp}, timeout=120)
       |            # If Replicate returns 202, the prediction is still running — poll until done
       |            if resp.status_code == 202:
       |                import time as _time
       |                pred = resp.json()
       |                poll_url = pred.get("urls", {}).get("get", "")
       |                if not poll_url:
       |                    return resp
       |                # Route poll through HF router
       |                from urllib.parse import urlparse as _urlparse
       |                poll_path = _urlparse(poll_url).path
       |                poll_url = f"{base}{poll_path}"
       |                for _ in range(300):
       |                    _time.sleep(2)
       |                    poll_resp = requests.get(poll_url, headers=json_headers, timeout=30)
       |                    if poll_resp.status_code != 200:
       |                        continue
       |                    poll_data = poll_resp.json()
       |                    status = poll_data.get("status", "")
       |                    if status == "succeeded":
       |                        return poll_resp
       |                    elif status in ("failed", "canceled"):
       |                        return poll_resp
       |                return poll_resp
       |            return resp
       |
       |        # ── Fal-ai ──
       |        # Route: /{providerId}
       |        # Payload varies by task
       |        if provider_name == "fal-ai":
       |            url = f"{base}/{provider_id}"
       |            if task == "text-to-speech":
       |                return requests.post(url, headers=json_headers, json={"text": prompt_value}, timeout=120)
       |            elif task in ("text-to-image", "text-to-video"):
       |                return requests.post(url, headers=json_headers, json={"prompt": prompt_value}, timeout=120)
       |            elif task == "image-to-image" and img_b64:
       |                data_url = f"data:image/png;base64,{img_b64}"
       |                return requests.post(url, headers=json_headers, json={"image_url": data_url, "image_urls": [data_url], "prompt": prompt_value}, timeout=120)
       |            elif img_b64:
       |                return requests.post(url, headers=json_headers, json={"image_url": f"data:image/png;base64,{img_b64}", "prompt": prompt_value}, timeout=120)
       |            else:
       |                return requests.post(url, headers=json_headers, json={"prompt": prompt_value}, timeout=120)
       |
       |        # ── Wavespeed ──
       |        # Async queue: POST /api/v3/{providerId}, then poll for result
       |        if provider_name == "wavespeed":
       |            url = f"{base}/api/v3/{provider_id}"
       |            payload = {"prompt": prompt_value}
       |            if img_b64:
       |                payload["image"] = img_b64
       |                payload["images"] = [img_b64]
       |            # Submit task
       |            submit_resp = requests.post(url, headers=json_headers, json=payload, timeout=120)
       |            if submit_resp.status_code not in (200, 201):
       |                return submit_resp
       |            submit_data = submit_resp.json()
       |            # Poll for result
       |            get_path = submit_data.get("data", {}).get("urls", {}).get("get", "")
       |            if not get_path:
       |                return submit_resp
       |            from urllib.parse import urlparse as _urlparse
       |            result_path = _urlparse(get_path).path
       |            result_url = f"{base}{result_path}"
       |            import time as _time
       |            for _ in range(120):
       |                _time.sleep(1)
       |                poll_resp = requests.get(result_url, headers=json_headers, timeout=30)
       |                if poll_resp.status_code != 200:
       |                    continue
       |                poll_data = poll_resp.json()
       |                status = poll_data.get("data", {}).get("status", "")
       |                if status == "completed":
       |                    return poll_resp
       |                elif status == "failed":
       |                    return poll_resp
       |            return poll_resp
       |
       |        # ── OpenAI-compatible providers ──
       |        # Most use v1/chat/completions; only these three differ:
       |        CUSTOM_CHAT_ROUTES = {"groq": "openai/v1/chat/completions", "fireworks-ai": "inference/v1/chat/completions", "cohere": "compatibility/v1/chat/completions", "clarifai": "v2/ext/openai/v1/chat/completions", "deepinfra": "v1/openai/chat/completions"}
       |        openai_providers = ("cerebras", "sambanova", "groq", "novita", "nebius", "fireworks-ai", "together", "hyperbolic", "cohere", "clarifai", "deepinfra", "featherless-ai", "nscale", "nvidia", "openai", "ovhcloud", "publicai", "scaleway", "baseten")
       |        if provider_name in openai_providers:
       |            if task in ("text-to-image",):
       |                url = f"{base}/v1/images/generations"
       |                return requests.post(url, headers=json_headers, json={"model": provider_id, "prompt": prompt_value}, timeout=120)
       |            elif task == "text-to-speech":
       |                url = f"{base}/v1/audio/speech"
       |                return requests.post(url, headers=json_headers, json={"model": provider_id, "input": prompt_value}, timeout=120)
       |            else:
       |                # Chat completions with provider-specific route
       |                url = f"{base}/{CUSTOM_CHAT_ROUTES.get(provider_name, 'v1/chat/completions')}"
       |                messages = [{"role": "user", "content": prompt_value}]
       |                if img_b64:
       |                    messages = [{"role": "user", "content": [
       |                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
       |                        {"type": "text", "text": prompt_value if prompt_value else "What is in this image?"},
       |                    ]}]
       |                return requests.post(url, headers=json_headers, json={"model": provider_id, "messages": messages}, timeout=120)
       |
       |        # ── Unknown provider: try pipeline format, then chat completions ──
       |        url = f"{base}/{provider_id}"
       |        if use_raw_binary_body:
       |            resp = requests.post(url, headers=raw_binary_headers, data=pipeline_payload, timeout=120)
       |        else:
       |            resp = requests.post(url, headers=json_headers, json=pipeline_payload, timeout=120)
       |        if resp.status_code in (400, 404, 422):
       |            # Pipeline format failed — try chat completions as fallback
       |            url = f"{base}/v1/chat/completions"
       |            messages = [{"role": "user", "content": prompt_value}]
       |            if img_b64:
       |                messages = [{"role": "user", "content": [
       |                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
       |                    {"type": "text", "text": prompt_value if prompt_value else "Describe this image."},
       |                ]}]
       |            resp2 = requests.post(url, headers=json_headers, json={"model": provider_id, "messages": messages}, timeout=120)
       |            if resp2.status_code == 200:
       |                return resp2
       |        return resp
       |
       |    @overrides
       |    def process_table(self, table: Table, port: int) -> Iterator[Optional[TableLike]]:
       |        prompt_col = self.PROMPT_COLUMN
       |        result_col = self.RESULT_COLUMN
       |        task = self.TASK
       |        image_only_tasks = ("image-classification", "object-detection", "image-segmentation", "image-to-text")
       |        image_prompt_tasks = ("visual-question-answering", "document-question-answering", "zero-shot-image-classification", "image-text-to-text", "image-to-image")
       |        image_tasks = image_only_tasks + image_prompt_tasks
       |        audio_only_tasks = ("automatic-speech-recognition", "audio-classification")
       |
       |        # --- resolve API token ---
       |        token = self.HF_API_TOKEN if self.HF_API_TOKEN else os.environ.get("HF_TOKEN", "")
       |        if not token:
       |            raise ValueError(
       |                "Hugging Face API token is not set. "
       |                "Provide it in the operator config or via HF_TOKEN env var."
       |            )
       |
       |        # --- resolve all available inference providers for this model (tried in order) ---
       |        providers = self._resolve_providers(token)
       |
       |        # --- validate prompt column exists ---
       |        if task not in image_tasks and task not in audio_only_tasks:
       |            assert prompt_col in table.columns, (
       |                f"Prompt column '{prompt_col}' not found in input table. "
       |                f"Available columns: {list(table.columns)}"
       |            )
       |
       |        # --- validate task-specific columns ---
       |        if task == "question-answering":
       |            ctx_col = self.CONTEXT_COLUMN
       |            assert ctx_col and ctx_col in table.columns, (
       |                f"Context column '{ctx_col}' not found in input table. "
       |                f"Available columns: {list(table.columns)}"
       |            )
       |        if task in ("sentence-similarity", "text-ranking"):
       |            sent_col = self.SENTENCES_COLUMN
       |            assert sent_col and sent_col in table.columns, (
       |                f"Sentences column '{sent_col}' not found in input table. "
       |                f"Available columns: {list(table.columns)}"
       |            )
       |
       |        # --- handle empty table ---
       |        if table.empty:
       |            table[result_col] = pd.Series(dtype="object")
       |            yield table
       |            return
       |
       |        json_headers = {
       |            "Authorization": f"Bearer {token}",
       |            "Content-Type": "application/json",
       |        }
       |        image_headers = {
       |            "Authorization": f"Bearer {token}",
       |            "Content-Type": "application/octet-stream",
       |        }
       |        audio_headers = {
       |            "Authorization": f"Bearer {token}",
       |            "Content-Type": self._get_audio_content_type(),
       |        }
       |
       |        # --- pre-compute table dict for table-question-answering ---
       |        table_dict = None
       |        if task == "table-question-answering":
       |            table_dict = {}
       |            for col in table.columns:
       |                if col != prompt_col and col != result_col:
       |                    table_dict[col] = [
       |                        str(v) if not pd.isna(v) else "" for v in table[col].tolist()
       |                    ]
       |
       |        has_image_upload = bool(self.IMAGE_INPUT) and bool(str(self.IMAGE_INPUT).strip())
       |        has_audio_upload = bool(self.AUDIO_INPUT) and bool(str(self.AUDIO_INPUT).strip())
       |        use_image_column = not has_image_upload and bool(self.INPUT_IMAGE_COLUMN) and self.INPUT_IMAGE_COLUMN in table.columns
       |        use_audio_column = not has_audio_upload and bool(self.INPUT_AUDIO_COLUMN) and self.INPUT_AUDIO_COLUMN in table.columns
       |        results = []
       |        image_bytes = None
       |        image_error = None
       |        audio_bytes = None
       |        audio_error = None
       |        if task in image_tasks and not use_image_column:
       |            if not self.IMAGE_INPUT or not str(self.IMAGE_INPUT).strip():
       |                image_error = "No image source. Set an Input Image Column or upload an image."
       |            else:
       |                try:
       |                    image_bytes = self._read_image_input()
       |                except Exception as e:
       |                    image_error = f"Could not read image input ({type(e).__name__}: {e})"
       |        if task in audio_only_tasks and not use_audio_column:
       |            if not self.AUDIO_INPUT or not str(self.AUDIO_INPUT).strip():
       |                audio_error = "No audio source. Set an Input Audio Column or upload audio."
       |            else:
       |                try:
       |                    audio_bytes = self._read_audio_input()
       |                except Exception as e:
       |                    audio_error = f"Could not read audio input ({type(e).__name__}: {e})"
       |        for idx, row in table.iterrows():
       |            if image_error is not None:
       |                results.append(self._format_error("Image task configuration error", image_error))
       |                continue
       |            if audio_error is not None:
       |                results.append(self._format_error("Audio task configuration error", audio_error))
       |                continue
       |
       |            if task in image_only_tasks:
       |                prompt_value = ""
       |            elif task in audio_only_tasks:
       |                prompt_value = ""
       |            elif task in image_prompt_tasks and prompt_col not in table.columns:
       |                prompt_value = "What is shown in this image?"
       |            else:
       |                prompt_value = row[prompt_col]
       |                # Convert None / NaN to empty string
       |                if pd.isna(prompt_value):
       |                    prompt_value = ""
       |                else:
       |                    prompt_value = str(prompt_value)
       |
       |            # --- resolve per-row binary data from columns ---
       |            current_image_bytes = image_bytes
       |            if task in image_tasks and use_image_column:
       |                try:
       |                    raw = self._read_binary_value(row[self.INPUT_IMAGE_COLUMN])
       |                    if raw is None:
       |                        results.append(self._format_error("Image data error", f"Row {idx}: image column is empty"))
       |                        continue
       |                    current_image_bytes = self._compress_image_bytes(raw)
       |                except Exception as e:
       |                    results.append(self._format_error("Image data error", f"Row {idx}: {type(e).__name__}: {e}"))
       |                    continue
       |            current_audio_bytes = audio_bytes
       |            if task in audio_only_tasks and use_audio_column:
       |                try:
       |                    current_audio_bytes = self._read_binary_value(row[self.INPUT_AUDIO_COLUMN])
       |                    if current_audio_bytes is None:
       |                        results.append(self._format_error("Audio data error", f"Row {idx}: audio column is empty"))
       |                        continue
       |                except Exception as e:
       |                    results.append(self._format_error("Audio data error", f"Row {idx}: {type(e).__name__}: {e}"))
       |                    continue
       |
       |            # --- build task-specific payload ---
       |            use_raw_binary_body = False
       |            raw_binary_headers = image_headers
       |            if task in image_only_tasks:
       |                payload = current_image_bytes
       |                use_raw_binary_body = True
       |                raw_binary_headers = image_headers
       |            elif task in audio_only_tasks:
       |                payload = current_audio_bytes
       |                use_raw_binary_body = True
       |                raw_binary_headers = audio_headers
       |            elif task in ("visual-question-answering", "document-question-answering"):
       |                payload = {
       |                    "inputs": {
       |                        "image": self._image_input_as_base64(current_image_bytes),
       |                        "question": prompt_value,
       |                    }
       |                }
       |            elif task == "image-text-to-text":
       |                # Vision LLM: send image + prompt via chat completions format
       |                img_b64 = self._image_input_as_base64(current_image_bytes)
       |                payload = {
       |                    "model": self.MODEL_ID,
       |                    "messages": [{
       |                        "role": "user",
       |                        "content": [
       |                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
       |                            {"type": "text", "text": prompt_value if prompt_value else "Describe this image."},
       |                        ],
       |                    }],
       |                    "max_tokens": self.MAX_NEW_TOKENS,
       |                }
       |            elif task == "image-to-image":
       |                # Send image as raw binary to pipeline (hf-inference)
       |                # or as base64 to third-party providers (handled in _call_provider)
       |                payload = current_image_bytes
       |                use_raw_binary_body = True
       |                raw_binary_headers = image_headers
       |            elif task == "zero-shot-image-classification":
       |                labels = [l.strip() for l in self.CANDIDATE_LABELS.split(",") if l.strip()]
       |                if not labels:
       |                    labels = ["person", "animal", "vehicle", "food", "indoor", "outdoor", "object"]
       |                payload = {
       |                    "inputs": self._image_input_as_base64(current_image_bytes),
       |                    "parameters": {"candidate_labels": labels},
       |                }
       |            elif task == "question-answering":
       |                ctx_val = row[self.CONTEXT_COLUMN]
       |                ctx_val = "" if pd.isna(ctx_val) else str(ctx_val)
       |                payload = {"inputs": {"question": prompt_value, "context": ctx_val}}
       |            elif task == "table-question-answering":
       |                payload = {"inputs": {"query": prompt_value, "table": table_dict}}
       |            elif task == "zero-shot-classification":
       |                labels = [l.strip() for l in self.CANDIDATE_LABELS.split(",") if l.strip()]
       |                payload = {
       |                    "inputs": prompt_value,
       |                    "parameters": {"candidate_labels": labels},
       |                }
       |            elif task in ("sentence-similarity", "text-ranking"):
       |                sent_val = row[self.SENTENCES_COLUMN]
       |                sent_val = "" if pd.isna(sent_val) else str(sent_val)
       |                sentences_list = [s.strip() for s in sent_val.split(",") if s.strip()]
       |                payload = {
       |                    "inputs": {
       |                        "source_sentence": prompt_value,
       |                        "sentences": sentences_list,
       |                    }
       |                }
       |            elif task == "text-generation":
       |                payload = {
       |                    "model": self.MODEL_ID,
       |                    "messages": [
       |                        {"role": "system", "content": self.SYSTEM_PROMPT},
       |                        {"role": "user", "content": prompt_value},
       |                    ],
       |                    "max_tokens": self.MAX_NEW_TOKENS,
       |                    "temperature": self.TEMPERATURE,
       |                }
       |            else:
       |                payload = {"inputs": prompt_value}
       |
       |            try:
       |                resp, provider_summary = self._post_with_fallback(
       |                    providers, json_headers, raw_binary_headers, payload, use_raw_binary_body, prompt_value
       |                )
       |
       |                if resp is None:
       |                    results.append(
       |                        self._format_error(
       |                            "All inference providers failed",
       |                            f"No provider could serve model '{self.MODEL_ID}'. "
       |                            f"Tried: {provider_summary}"
       |                        )
       |                    )
       |                    continue
       |
       |                if resp.status_code == 429:
       |                    results.append(
       |                        self._format_http_error(
       |                            "HF API rate limit hit, retry later", resp.status_code, resp.text
       |                        )
       |                    )
       |                    continue
       |                if resp.status_code == 401:
       |                    results.append(
       |                        self._format_http_error("Invalid HF API token", resp.status_code, resp.text)
       |                    )
       |                    continue
       |                if resp.status_code not in (200, 201):
       |                    results.append(
       |                        self._format_error(
       |                            "All inference providers failed",
       |                            f"No provider could serve model '{self.MODEL_ID}'. "
       |                            f"Tried: {provider_summary}"
       |                        )
       |                    )
       |                    continue
       |
       |                content_type = resp.headers.get("Content-Type", "")
       |                if content_type.startswith("image/"):
       |                    b64 = base64.b64encode(resp.content).decode("utf-8")
       |                    results.append(f"data:{content_type};base64,{b64}")
       |                    continue
       |                if content_type.startswith("audio/"):
       |                    b64 = base64.b64encode(resp.content).decode("utf-8")
       |                    results.append(f"data:{content_type};base64,{b64}")
       |                    continue
       |                if content_type.startswith("video/"):
       |                    b64 = base64.b64encode(resp.content).decode("utf-8")
       |                    results.append(f"data:{content_type};base64,{b64}")
       |                    continue
       |
       |                try:
       |                    body = resp.json()
       |                except ValueError:
       |                    body = resp.text
       |                content = self._parse_response(body)
       |                results.append(content)
       |
       |            except Exception as e:
       |                # Per-row failures should still produce a visible result row.
       |                import warnings
       |                warnings.warn(
       |                    f"Row {idx}: request failed ({type(e).__name__}: {e}), "
       |                    f"setting result to readable error text."
       |                )
       |                results.append(self._format_error("Request failed", f"{type(e).__name__}: {e}"))
       |
       |        table[result_col] = results
       |        yield table
       |
       |    def _read_image_input(self):
       |        image_input = str(self.IMAGE_INPUT or "").strip()
       |        if image_input.startswith("data:"):
       |            _, encoded = image_input.split(",", 1)
       |            return base64.b64decode(encoded)
       |        if image_input.startswith("http://") or image_input.startswith("https://"):
       |            resp = requests.get(image_input, timeout=120)
       |            resp.raise_for_status()
       |            return resp.content
       |        if not os.path.exists(image_input):
       |            raise FileNotFoundError(f"Image file not found at path: {image_input}")
       |        if not os.path.isfile(image_input):
       |            raise ValueError(f"Image input path is not a file: {image_input}")
       |        with open(image_input, "rb") as image_file:
       |            return image_file.read()
       |
       |    def _compress_image_bytes(self, image_bytes, max_bytes=33000):
       |        from io import BytesIO
       |        from PIL import Image as PILImage
       |        if len(image_bytes) <= max_bytes:
       |            return image_bytes
       |        try:
       |            img = PILImage.open(BytesIO(image_bytes))
       |            img = img.convert("RGB")
       |            max_dim = 512
       |            quality = 75
       |            while max_dim >= 160:
       |                scale = min(1, max_dim / max(img.width, img.height))
       |                w = max(1, round(img.width * scale))
       |                h = max(1, round(img.height * scale))
       |                resized = img.resize((w, h), PILImage.LANCZOS)
       |                q = quality
       |                while q >= 35:
       |                    buf = BytesIO()
       |                    resized.save(buf, format="JPEG", quality=q)
       |                    if buf.tell() <= max_bytes:
       |                        return buf.getvalue()
       |                    q -= 10
       |                max_dim = int(max_dim * 0.75)
       |            buf = BytesIO()
       |            resized.save(buf, format="JPEG", quality=35)
       |            return buf.getvalue()
       |        except Exception:
       |            return image_bytes
       |
       |    def _image_input_as_base64(self, image_bytes):
       |        return base64.b64encode(image_bytes).decode("utf-8")
       |
       |    def _read_audio_input(self):
       |        audio_input = str(self.AUDIO_INPUT or "").strip()
       |        if audio_input.startswith("data:"):
       |            _, encoded = audio_input.split(",", 1)
       |            return base64.b64decode(encoded)
       |        if audio_input.startswith("http://") or audio_input.startswith("https://"):
       |            resp = requests.get(audio_input, timeout=120)
       |            resp.raise_for_status()
       |            return resp.content
       |        if not os.path.exists(audio_input):
       |            raise FileNotFoundError(f"Audio file not found at path: {audio_input}")
       |        if not os.path.isfile(audio_input):
       |            raise ValueError(f"Audio input path is not a file: {audio_input}")
       |        with open(audio_input, "rb") as audio_file:
       |            return audio_file.read()
       |
       |    def _read_binary_value(self, value):
       |        if value is None or (isinstance(value, float) and pd.isna(value)):
       |            return None
       |        if isinstance(value, bytes):
       |            return value
       |        val = str(value).strip()
       |        if not val:
       |            return None
       |        if self._looks_like_html(val):
       |            return self._html_to_image_bytes(val)
       |        if val.startswith("data:"):
       |            _, encoded = val.split(",", 1)
       |            return base64.b64decode(encoded)
       |        if val.startswith("http://") or val.startswith("https://"):
       |            resp = requests.get(val, timeout=120)
       |            resp.raise_for_status()
       |            return resp.content
       |        if os.path.exists(val) and os.path.isfile(val):
       |            with open(val, "rb") as f:
       |                return f.read()
       |        try:
       |            return base64.b64decode(val)
       |        except Exception:
       |            return val.encode("utf-8")
       |
       |    def _looks_like_html(self, val):
       |        s = val.lstrip()[:200].lower()
       |        if s.startswith("<!doctype html") or s.startswith("<html"):
       |            return True
       |        if "plotly.newplot" in val[:5000].lower() or "plotly.react" in val[:5000].lower():
       |            return True
       |        if "<img" in s and "base64," in s:
       |            return True
       |        return False
       |
       |    def _html_to_image_bytes(self, html_string):
       |        # Case 1: Extract embedded base64 image (WordCloud, ImageVisualizer, etc.)
       |        match = re.search(r'data:image/[^;]+;base64,([A-Za-z0-9+/\\n\\r =]+)', html_string)
       |        if match:
       |            b64 = match.group(1).replace('\\n', '').replace('\\r', '').replace(' ', '')
       |            return base64.b64decode(b64)
       |        # Case 2: Extract Plotly figure and render as PNG via Kaleido
       |        if "Plotly." in html_string:
       |            try:
       |                import plotly.graph_objects as go
       |                import plotly.io as pio
       |                plotly_match = re.search(r'Plotly\\.(?:newPlot|react)\\s*\\(\\s*', html_string)
       |                if plotly_match:
       |                    pos = plotly_match.end()
       |                    # Skip first arg (div id string)
       |                    if pos < len(html_string) and html_string[pos] in ('"', "'"):
       |                        q = html_string[pos]
       |                        pos += 1
       |                        while pos < len(html_string) and html_string[pos] != q:
       |                            if html_string[pos] == '\\\\':
       |                                pos += 1
       |                            pos += 1
       |                        pos += 1
       |                    # Skip comma/whitespace to data array
       |                    while pos < len(html_string) and html_string[pos] in ' ,\\n\\r\\t':
       |                        pos += 1
       |                    data_json, pos = self._extract_json_arg(html_string, pos)
       |                    # Skip comma/whitespace to layout object
       |                    while pos < len(html_string) and html_string[pos] in ' ,\\n\\r\\t':
       |                        pos += 1
       |                    layout_json, _ = self._extract_json_arg(html_string, pos)
       |                    if data_json:
       |                        data = json.loads(data_json)
       |                        layout = json.loads(layout_json) if layout_json else {}
       |                        fig = go.Figure(data=data, layout=layout)
       |                        return pio.to_image(fig, format="png", width=800, height=600)
       |            except ImportError as ie:
       |                missing = str(ie)
       |                raise ValueError(
       |                    f"Plotly chart detected but cannot render to image: {missing}. "
       |                    f"Install kaleido: pip install kaleido"
       |                )
       |            except json.JSONDecodeError:
       |                pass
       |        raise ValueError(
       |            "Cannot convert HTML to image. The HTML does not contain "
       |            "an extractable base64 image or a parseable Plotly chart."
       |        )
       |
       |    def _extract_json_arg(self, text, start_pos):
       |        if start_pos >= len(text):
       |            return None, start_pos
       |        ch = text[start_pos]
       |        openers = {'[': ']', '{': '}'}
       |        if ch not in openers:
       |            return None, start_pos
       |        closer = openers[ch]
       |        depth = 1
       |        pos = start_pos + 1
       |        in_string = False
       |        while pos < len(text) and depth > 0:
       |            c = text[pos]
       |            if in_string:
       |                if c == '\\\\':
       |                    pos += 2
       |                    continue
       |                if c == '"':
       |                    in_string = False
       |            else:
       |                if c == '"':
       |                    in_string = True
       |                elif c == ch:
       |                    depth += 1
       |                elif c == closer:
       |                    depth -= 1
       |            pos += 1
       |        if depth == 0:
       |            return text[start_pos:pos], pos
       |        return None, start_pos
       |
       |    def _get_audio_content_type(self):
       |        audio_input = str(self.AUDIO_INPUT or "").strip().lower()
       |        if audio_input.startswith("data:"):
       |            header = audio_input.split(",", 1)[0]
       |            if ";" in header:
       |                return header[5:header.index(";")]
       |            return header[5:]
       |        extension_map = {
       |            ".mp3": "audio/mpeg",
       |            ".mpeg": "audio/mpeg",
       |            ".wav": "audio/wav",
       |            ".flac": "audio/flac",
       |            ".ogg": "audio/ogg",
       |            ".oga": "audio/ogg",
       |            ".webm": "audio/webm",
       |            ".opus": "audio/webm;codecs=opus",
       |            ".amr": "audio/amr",
       |            ".m4a": "audio/m4a",
       |        }
       |        _, ext = os.path.splitext(audio_input)
       |        return extension_map.get(ext, "audio/mpeg")
       |
       |    def _audio_url_to_data_url(self, url):
       |        resp = requests.get(url, timeout=120)
       |        resp.raise_for_status()
       |        content_type = resp.headers.get("Content-Type", "").strip()
       |        if not content_type or content_type == "application/octet-stream":
       |            parsed = urlparse(url)
       |            _, ext = os.path.splitext(parsed.path.lower())
       |            extension_map = {
       |                ".mp3": "audio/mpeg",
       |                ".mpeg": "audio/mpeg",
       |                ".wav": "audio/wav",
       |                ".flac": "audio/flac",
       |                ".ogg": "audio/ogg",
       |                ".oga": "audio/ogg",
       |                ".webm": "audio/webm",
       |                ".opus": "audio/webm;codecs=opus",
       |                ".amr": "audio/amr",
       |                ".m4a": "audio/m4a",
       |            }
       |            content_type = extension_map.get(ext, "audio/mpeg")
       |        b64 = base64.b64encode(resp.content).decode("utf-8")
       |        return f"data:{content_type};base64,{b64}"
       |
       |    def _url_to_data_url(self, url):
       |        \"\"\"Fetch a URL and return a data URL with the correct MIME type.\"\"\"
       |        resp = requests.get(url, timeout=120)
       |        resp.raise_for_status()
       |        content_type = resp.headers.get("Content-Type", "").split(";")[0].strip()
       |        if not content_type or content_type == "application/octet-stream":
       |            from urllib.parse import urlparse
       |            ext = os.path.splitext(urlparse(url).path.lower())[1]
       |            mime_map = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".gif": "image/gif", ".webp": "image/webp", ".svg": "image/svg+xml", ".mp4": "video/mp4", ".webm": "video/webm"}
       |            guessed = mime_map.get(ext, "")
       |            if guessed:
       |                content_type = guessed
       |            else:
       |                # Infer from task when all else fails
       |                task_mime = {"text-to-video": "video/mp4", "text-to-image": "image/png", "image-to-image": "image/png", "text-to-speech": "audio/mpeg"}
       |                content_type = task_mime.get(self.TASK, "application/octet-stream")
       |        b64 = base64.b64encode(resp.content).decode("utf-8")
       |        return f"data:{content_type};base64,{b64}"
       |
       |    def _format_error(self, title, detail):
       |        return f"{title}: {detail}"
       |
       |    def _format_http_error(self, title, status_code, response_text):
       |        detail = response_text.strip()
       |        if not detail:
       |            detail = "<empty response>"
       |        return f"{title} [status={status_code}] response={detail}"
       |
       |    def _parse_response(self, body):
       |        task = self.TASK
       |        try:
       |            if isinstance(body, str):
       |                return body
       |            if task == "text-generation":
       |                return body["choices"][0]["message"]["content"]
       |            if task == "text-classification":
       |                data = body[0] if isinstance(body, list) and len(body) > 0 and isinstance(body[0], list) else body
       |                return json.dumps(data)
       |            elif task == "token-classification":
       |                return json.dumps(body)
       |            elif task == "translation":
       |                return body[0]["translation_text"]
       |            elif task == "summarization":
       |                return body[0]["summary_text"]
       |            elif task == "fill-mask":
       |                return json.dumps(body)
       |            elif task == "feature-extraction":
       |                return json.dumps(body)
       |            elif task == "question-answering":
       |                return body.get("answer", json.dumps(body))
       |            elif task == "table-question-answering":
       |                return body.get("answer", json.dumps(body))
       |            elif task == "text-to-image":
       |                # Always return data:image/...;base64,... for consistency
       |                if isinstance(body, dict):
       |                    # Replicate: {"output": "url"} or {"output": ["url"]}
       |                    if "output" in body:
       |                        out = body["output"]
       |                        url = out[0] if isinstance(out, list) else out
       |                        if isinstance(url, str) and url.startswith("http"):
       |                            return self._url_to_data_url(url)
       |                    # fal-ai: {"images": [{"url": "..."}]}
       |                    if "images" in body:
       |                        images = body["images"]
       |                        if images and isinstance(images[0], dict) and "url" in images[0]:
       |                            return self._url_to_data_url(images[0]["url"])
       |                    # OpenAI format: {"data": [{"b64_json": "...", "url": "..."}]}
       |                    if "data" in body:
       |                        data = body["data"]
       |                        # Wavespeed: {"data": {"outputs": ["url"], "status": "completed"}}
       |                        if isinstance(data, dict) and "outputs" in data:
       |                            outputs = data["outputs"]
       |                            if outputs and isinstance(outputs[0], str) and outputs[0].startswith("http"):
       |                                return self._url_to_data_url(outputs[0])
       |                        # OpenAI format: {"data": [{"b64_json": "...", "url": "..."}]}
       |                        if isinstance(data, list) and data and isinstance(data[0], dict):
       |                            if "b64_json" in data[0]:
       |                                return f"data:image/png;base64,{data[0]['b64_json']}"
       |                            if "url" in data[0]:
       |                                return self._url_to_data_url(data[0]["url"])
       |                return json.dumps(body)
       |            elif task == "text-to-video":
       |                if isinstance(body, dict):
       |                    # Replicate: {"output": "url"}
       |                    if "output" in body:
       |                        out = body["output"]
       |                        url = out[0] if isinstance(out, list) else out
       |                        if isinstance(url, str) and url.startswith("http"):
       |                            return self._url_to_data_url(url)
       |                    # fal-ai / others: {"video": {"url": "..."}}
       |                    if "video" in body:
       |                        video = body["video"]
       |                        if isinstance(video, dict) and "url" in video:
       |                            return self._url_to_data_url(video["url"])
       |                return json.dumps(body)
       |            elif task == "text-to-speech":
       |                # Always return data:audio/...;base64,... for consistency
       |                if isinstance(body, dict):
       |                    # Replicate: {"output": "url"}
       |                    if "output" in body:
       |                        out = body["output"]
       |                        url = out[0] if isinstance(out, list) else out
       |                        if isinstance(url, str) and url.startswith("http"):
       |                            return self._audio_url_to_data_url(url)
       |                    # fal-ai: {"audio": {"url": "..."}}
       |                    if "audio" in body:
       |                        audio = body["audio"]
       |                        if isinstance(audio, dict):
       |                            if "url" in audio:
       |                                return self._audio_url_to_data_url(audio["url"])
       |                            if "b64_json" in audio:
       |                                return f"data:audio/mpeg;base64,{audio['b64_json']}"
       |                    if "data" in body:
       |                        data = body["data"]
       |                        if data and isinstance(data[0], dict):
       |                            if "url" in data[0]:
       |                                return self._audio_url_to_data_url(data[0]["url"])
       |                            if "b64_json" in data[0]:
       |                                return f"data:audio/mpeg;base64,{data[0]['b64_json']}"
       |                return json.dumps(body)
       |            elif task == "automatic-speech-recognition":
       |                if isinstance(body, dict):
       |                    if "text" in body:
       |                        return body["text"]
       |                    if "generated_text" in body:
       |                        return body["generated_text"]
       |                return json.dumps(body)
       |            elif task == "image-to-text":
       |                if isinstance(body, dict):
       |                    # zai-org layout_parsing: {"md_results": "..."}
       |                    if "md_results" in body:
       |                        return body["md_results"]
       |                    # Chat completions format (model-author providers)
       |                    if "choices" in body:
       |                        return body["choices"][0]["message"]["content"]
       |                # Pipeline format (hf-inference): [{"generated_text": "..."}]
       |                if isinstance(body, list) and body and isinstance(body[0], dict):
       |                    return body[0].get("generated_text", json.dumps(body))
       |                return json.dumps(body)
       |            elif task in ("visual-question-answering", "document-question-answering"):
       |                if isinstance(body, dict):
       |                    return body.get("answer", json.dumps(body))
       |                return json.dumps(body)
       |            elif task == "image-text-to-text":
       |                # Chat completions format from vision LLMs
       |                if isinstance(body, dict) and "choices" in body:
       |                    return body["choices"][0]["message"]["content"]
       |                if isinstance(body, list) and body and isinstance(body[0], dict):
       |                    return body[0].get("generated_text", json.dumps(body))
       |                return json.dumps(body)
       |            elif task == "image-to-image":
       |                # Raw image bytes handled by Content-Type check above;
       |                # JSON responses (replicate/fal-ai) contain URLs
       |                if isinstance(body, dict):
       |                    if "output" in body:
       |                        out = body["output"]
       |                        url = out[0] if isinstance(out, list) else out
       |                        if isinstance(url, str) and url.startswith("http"):
       |                            return self._url_to_data_url(url)
       |                    if "images" in body:
       |                        images = body["images"]
       |                        if images and isinstance(images[0], dict) and "url" in images[0]:
       |                            return self._url_to_data_url(images[0]["url"])
       |                    if "data" in body:
       |                        data = body["data"]
       |                        # Wavespeed: {"data": {"outputs": ["url"], "status": "completed"}}
       |                        if isinstance(data, dict) and "outputs" in data:
       |                            outputs = data["outputs"]
       |                            if outputs and isinstance(outputs[0], str) and outputs[0].startswith("http"):
       |                                return self._url_to_data_url(outputs[0])
       |                        # OpenAI format: {"data": [{"b64_json": "...", "url": "..."}]}
       |                        if isinstance(data, list) and data and isinstance(data[0], dict):
       |                            if "b64_json" in data[0]:
       |                                return f"data:image/png;base64,{data[0]['b64_json']}"
       |                            if "url" in data[0]:
       |                                return self._url_to_data_url(data[0]["url"])
       |                return json.dumps(body)
       |            elif task in ("zero-shot-classification", "sentence-similarity", "text-ranking", "image-classification", "object-detection", "image-segmentation", "zero-shot-image-classification", "audio-classification"):
       |                return json.dumps(body)
       |            else:
       |                return json.dumps(body)
       |        except (KeyError, IndexError, TypeError):
       |            return json.dumps(body)
       |""".stripMargin
  }

  override def operatorInfo: OperatorInfo =
    OperatorInfo(
      "Hugging Face",
      "Call a Hugging Face model via the Inference API",
      OperatorGroupConstants.HUGGINGFACE_GROUP,
      inputPorts = List(InputPort()),
      outputPorts = List(OutputPort())
    )

  override def getOutputSchemas(
      inputSchemas: Map[PortIdentity, Schema]
  ): Map[PortIdentity, Schema] = {
    val resCol =
      if (resultColumn == null || resultColumn.trim.isEmpty) "hf_response"
      else resultColumn
    Map(
      operatorInfo.outputPorts.head.id -> inputSchemas.values.head
        .add(resCol, AttributeType.STRING)
    )
  }

  /** Escape a string for safe embedding inside a Python string literal (double-quoted). */
  private def escapePython(s: String): String = {
    if (s == null) return ""
    s.replace("\\", "\\\\")
      .replace("\"", "\\\"")
      .replace("\n", "\\n")
      .replace("\r", "\\r")
      .replace("\t", "\\t")
  }
}
