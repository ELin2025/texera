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
import org.apache.texera.amber.operator.{PortDescription, PythonOperatorDescriptor}
import org.apache.texera.amber.operator.metadata.annotations.AutofillAttributeName
import org.apache.texera.amber.operator.metadata.{OperatorGroupConstants, OperatorInfo}

class HuggingFaceInferenceOpDesc extends PythonOperatorDescriptor {

  private val InputPortColumn = "hf_input_port"

  @JsonIgnore
  var hfApiToken: String = ""

  @JsonProperty(value = "task", required = false, defaultValue = "text-generation")
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

  @JsonProperty(value = "promptColumn", required = true)
  @JsonSchemaTitle("Prompt Column")
  @JsonPropertyDescription("Column in the input table to use as the user prompt")
  @AutofillAttributeName
  var promptColumn: String = ""

  @JsonProperty(
    value = "systemPrompt",
    required = false,
    defaultValue = "You are a helpful assistant."
  )
  @JsonSchemaTitle("System Prompt")
  @JsonPropertyDescription("Optional system message to set model behavior")
  var systemPrompt: String = "You are a helpful assistant."

  @JsonProperty(value = "maxNewTokens", required = true, defaultValue = "256")
  @JsonSchemaTitle("Max New Tokens")
  @JsonPropertyDescription("Maximum number of tokens to generate (1-4096)")
  var maxNewTokens: Int = 256

  @JsonProperty(value = "temperature", required = true)
  @JsonSchemaTitle("Temperature")
  @JsonPropertyDescription("Sampling temperature (0.0 = deterministic, up to 2.0)")
  var temperature: Double = 0.7

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
    assert(
      promptColumn != null && promptColumn.trim.nonEmpty,
      "Prompt Column must not be empty"
    )
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
    val safeTask = if (task == null || task.trim.isEmpty) "text-generation" else task

    if (safeTask == "text-generation") {
      generateTextGenPython(pyToken, pyModelId, pyPromptCol, pyResultCol)
    } else {
      val pyContextCol = escapePython(contextColumn)
      val pyCandidateLabels = escapePython(candidateLabels)
      val pySentencesCol = escapePython(sentencesColumn)
      generateInferencePython(
        pyToken, pyModelId, pyPromptCol, pyResultCol,
        escapePython(safeTask), pyContextCol, pyCandidateLabels, pySentencesCol
      )
    }
  }

  private def generateTextGenPython(
      pyToken: String,
      pyModelId: String,
      pyPromptCol: String,
      pyResultCol: String
  ): String = {
    val safeMaxTokens = math.max(1, math.min(maxNewTokens, 4096))
    val safeTemp = math.max(0.0, math.min(temperature, 2.0))
    val pySystemPrompt = escapePython(systemPrompt)

    s"""import os
       |import requests
       |import pandas as pd
       |from pytexera import *
       |
       |class ProcessTableOperator(UDFTableOperator):
       |
       |    # ---- configuration injected at code-generation time ----
       |    HF_API_TOKEN   = "$pyToken"
       |    MODEL_ID       = "$pyModelId"
       |    PROMPT_COLUMN  = "$pyPromptCol"
       |    SYSTEM_PROMPT  = "$pySystemPrompt"
       |    MAX_NEW_TOKENS = $safeMaxTokens
       |    TEMPERATURE    = $safeTemp
       |    RESULT_COLUMN  = "$pyResultCol"
       |    HF_API_URL     = "https://router.huggingface.co/v1/chat/completions"
       |
       |    @overrides
       |    def process_table(self, table: Table, port: int) -> Iterator[Optional[TableLike]]:
       |        prompt_col = self.PROMPT_COLUMN
       |        result_col = self.RESULT_COLUMN
       |
       |        # --- resolve API token ---
       |        token = self.HF_API_TOKEN if self.HF_API_TOKEN else os.environ.get("HF_TOKEN", "")
       |        if not token:
       |            raise ValueError(
       |                "Hugging Face API token is not set. "
       |                "Provide it in the operator config or via HF_TOKEN env var."
       |            )
       |
       |        # --- validate prompt column exists ---
       |        if prompt_col not in table.columns:
       |            import warnings
       |            warnings.warn(
       |                f"Skipping input port {port}: prompt column '{prompt_col}' "
       |                f"not found. Available columns: {list(table.columns)}"
       |            )
       |            return
       |
       |        # --- handle result column conflict: overwrite policy ---
       |        # If resultColumn already exists in the table, it will be overwritten.
       |        # This is by design so that re-runs do not fail.
       |
       |        # --- handle empty table ---
       |        if table.empty:
       |            table[result_col] = pd.Series(dtype="object")
       |            table["$InputPortColumn"] = pd.Series(dtype="int64")
       |            yield table
       |            return
       |
       |        headers = {
       |            "Authorization": f"Bearer {token}",
       |            "Content-Type": "application/json",
       |        }
       |
       |        results = []
       |        for idx, row in table.iterrows():
       |            prompt_value = row[prompt_col]
       |            # Convert None / NaN to empty string
       |            if pd.isna(prompt_value):
       |                prompt_value = ""
       |            else:
       |                prompt_value = str(prompt_value)
       |
       |            payload = {
       |                "model": self.MODEL_ID,
       |                "messages": [
       |                    {"role": "system", "content": self.SYSTEM_PROMPT},
       |                    {"role": "user",   "content": prompt_value},
       |                ],
       |                "max_tokens": self.MAX_NEW_TOKENS,
       |                "temperature": self.TEMPERATURE,
       |            }
       |
       |            try:
       |                resp = requests.post(
       |                    self.HF_API_URL, headers=headers, json=payload, timeout=120
       |                )
       |
       |                if resp.status_code == 429:
       |                    raise RuntimeError(
       |                        f"HF API rate limit hit, retry later: "
       |                        f"{resp.status_code} {resp.text}"
       |                    )
       |                if resp.status_code == 401:
       |                    raise ValueError(
       |                        f"Invalid HF API token: {resp.status_code} {resp.text}"
       |                    )
       |                if resp.status_code != 200:
       |                    raise RuntimeError(
       |                        f"HF API error for model '{self.MODEL_ID}': "
       |                        f"{resp.status_code} {resp.text}"
       |                    )
       |
       |                body = resp.json()
       |                try:
       |                    content = body["choices"][0]["message"]["content"]
       |                except (KeyError, IndexError, TypeError):
       |                    import warnings
       |                    warnings.warn(
       |                        f"Row {idx}: unexpected response structure, "
       |                        f"setting result to empty string. Response: {body}"
       |                    )
       |                    content = ""
       |
       |                results.append(content)
       |
       |            except (RuntimeError, ValueError):
       |                # Fatal errors (rate-limit, auth) should propagate immediately
       |                raise
       |            except Exception as e:
       |                # Per-row non-fatal failures: log and continue
       |                import warnings
       |                warnings.warn(
       |                    f"Row {idx}: request failed ({type(e).__name__}: {e}), "
       |                    f"setting result to empty string."
       |                )
       |                results.append("")
       |
       |        table[result_col] = results
       |        table["$InputPortColumn"] = port
       |        yield table
       |""".stripMargin
  }

  private def generateInferencePython(
      pyToken: String,
      pyModelId: String,
      pyPromptCol: String,
      pyResultCol: String,
      pyTask: String,
      pyContextCol: String,
      pyCandidateLabels: String,
      pySentencesCol: String
  ): String = {
    s"""import os
       |import json
       |import requests
       |import pandas as pd
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
       |    HF_API_URL        = "https://router.huggingface.co/hf-inference/models/$pyModelId"
       |
       |    @overrides
       |    def process_table(self, table: Table, port: int) -> Iterator[Optional[TableLike]]:
       |        prompt_col = self.PROMPT_COLUMN
       |        result_col = self.RESULT_COLUMN
       |        task = self.TASK
       |
       |        # --- resolve API token ---
       |        token = self.HF_API_TOKEN if self.HF_API_TOKEN else os.environ.get("HF_TOKEN", "")
       |        if not token:
       |            raise ValueError(
       |                "Hugging Face API token is not set. "
       |                "Provide it in the operator config or via HF_TOKEN env var."
       |            )
       |
       |        # --- validate prompt column exists ---
       |        if prompt_col not in table.columns:
       |            import warnings
       |            warnings.warn(
       |                f"Skipping input port {port}: prompt column '{prompt_col}' "
       |                f"not found. Available columns: {list(table.columns)}"
       |            )
       |            return
       |
       |        # --- validate task-specific columns ---
       |        if task == "question-answering":
       |            ctx_col = self.CONTEXT_COLUMN
       |            if not ctx_col or ctx_col not in table.columns:
       |                import warnings
       |                warnings.warn(
       |                    f"Skipping input port {port}: context column '{ctx_col}' "
       |                    f"not found. Available columns: {list(table.columns)}"
       |                )
       |                return
       |        if task in ("sentence-similarity", "text-ranking"):
       |            sent_col = self.SENTENCES_COLUMN
       |            if not sent_col or sent_col not in table.columns:
       |                import warnings
       |                warnings.warn(
       |                    f"Skipping input port {port}: sentences column '{sent_col}' "
       |                    f"not found. Available columns: {list(table.columns)}"
       |                )
       |                return
       |
       |        # --- handle empty table ---
       |        if table.empty:
       |            table[result_col] = pd.Series(dtype="object")
       |            table["$InputPortColumn"] = pd.Series(dtype="int64")
       |            yield table
       |            return
       |
       |        headers = {
       |            "Authorization": f"Bearer {token}",
       |            "Content-Type": "application/json",
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
       |        results = []
       |        for idx, row in table.iterrows():
       |            prompt_value = row[prompt_col]
       |            # Convert None / NaN to empty string
       |            if pd.isna(prompt_value):
       |                prompt_value = ""
       |            else:
       |                prompt_value = str(prompt_value)
       |
       |            # --- build task-specific payload ---
       |            if task == "question-answering":
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
       |            else:
       |                payload = {"inputs": prompt_value}
       |
       |            try:
       |                resp = requests.post(
       |                    self.HF_API_URL, headers=headers, json=payload, timeout=120
       |                )
       |
       |                if resp.status_code == 429:
       |                    raise RuntimeError(
       |                        f"HF API rate limit hit, retry later: "
       |                        f"{resp.status_code} {resp.text}"
       |                    )
       |                if resp.status_code == 401:
       |                    raise ValueError(
       |                        f"Invalid HF API token: {resp.status_code} {resp.text}"
       |                    )
       |                if resp.status_code != 200:
       |                    raise RuntimeError(
       |                        f"HF API error for model '{self.MODEL_ID}': "
       |                        f"{resp.status_code} {resp.text}"
       |                    )
       |
       |                body = resp.json()
       |                content = self._parse_response(body)
       |                results.append(content)
       |
       |            except (RuntimeError, ValueError):
       |                # Fatal errors (rate-limit, auth) should propagate immediately
       |                raise
       |            except Exception as e:
       |                # Per-row non-fatal failures: log and continue
       |                import warnings
       |                warnings.warn(
       |                    f"Row {idx}: request failed ({type(e).__name__}: {e}), "
       |                    f"setting result to empty string."
       |                )
       |                results.append("")
       |
       |        table[result_col] = results
       |        table["$InputPortColumn"] = port
       |        yield table
       |
       |    def _parse_response(self, body):
       |        task = self.TASK
       |        try:
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
       |            elif task in ("zero-shot-classification", "sentence-similarity", "text-ranking"):
       |                return json.dumps(body)
       |            else:
       |                return json.dumps(body)
       |        except (KeyError, IndexError, TypeError):
       |            return json.dumps(body)
       |""".stripMargin
  }

  override def operatorInfo: OperatorInfo = {
    val inputPortInfo = if (inputPorts != null) {
      inputPorts.zipWithIndex.map {
        case (portDesc: PortDescription, idx) =>
          InputPort(
            PortIdentity(idx),
            displayName = portDesc.displayName,
            disallowMultiLinks = portDesc.disallowMultiInputs,
            dependencies = portDesc.dependencies.map(idx => PortIdentity(idx))
          )
      }
    } else {
      List(InputPort())
    }

    OperatorInfo(
      "Hugging Face",
      "Call a Hugging Face model via the Inference API",
      OperatorGroupConstants.HUGGINGFACE_GROUP,
      inputPorts = inputPortInfo,
      outputPorts = List(OutputPort()),
      dynamicInputPorts = true,
      dynamicOutputPorts = false,
      allowPortCustomization = true
    )
  }

  override def getOutputSchemas(
      inputSchemas: Map[PortIdentity, Schema]
  ): Map[PortIdentity, Schema] = {
    val resCol =
      if (resultColumn == null || resultColumn.trim.isEmpty) "hf_response"
      else resultColumn
    val outputSchema = inputSchemas.values
      .find(schema => promptColumn != null && schema.containsAttribute(promptColumn))
      .getOrElse(inputSchemas.values.head)
      .add(resCol, AttributeType.STRING)
      .add(InputPortColumn, AttributeType.INTEGER)

    Map(operatorInfo.outputPorts.head.id -> outputSchema)
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
