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

import com.fasterxml.jackson.annotation.{JsonProperty, JsonPropertyDescription}
import com.kjetland.jackson.jsonSchema.annotations.JsonSchemaTitle
import org.apache.texera.amber.core.tuple.{AttributeType, Schema}
import org.apache.texera.amber.core.workflow.{InputPort, OutputPort, PortIdentity}
import org.apache.texera.amber.operator.PythonOperatorDescriptor
import org.apache.texera.amber.operator.metadata.annotations.AutofillAttributeName
import org.apache.texera.amber.operator.metadata.{OperatorGroupConstants, OperatorInfo}

class HuggingFaceTextGenOpDesc extends PythonOperatorDescriptor {

  @JsonProperty(value = "hfApiToken", required = false, defaultValue = "")
  @JsonSchemaTitle("Hugging Face API Token")
  @JsonPropertyDescription(
    "Your HF token from huggingface.co/settings/tokens. If empty, reads from env var HF_TOKEN."
  )
  var hfApiToken: String = ""

  @JsonProperty(
    value = "modelId",
    required = true,
    defaultValue = "Qwen/Qwen2.5-72B-Instruct"
  )
  @JsonSchemaTitle("Model ID")
  @JsonPropertyDescription(
    "Select a Hugging Face text-generation model"
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

  override def generatePythonCode(): String = {
    // Validate fields before generating code
    assert(
      promptColumn != null && promptColumn.trim.nonEmpty,
      "Prompt Column must not be empty"
    )
    assert(
      modelId != null && modelId.trim.nonEmpty,
      "Model ID must not be empty"
    )

    // Clamp maxNewTokens and temperature at code-gen time
    val safeMaxTokens = math.max(1, math.min(maxNewTokens, 4096))
    val safeTemp = math.max(0.0, math.min(temperature, 2.0))

    // Escape strings for safe embedding in Python source
    val pyToken = escapePython(if (hfApiToken == null) "" else hfApiToken)
    val pyModelId = escapePython(modelId)
    val pyPromptCol = escapePython(promptColumn)
    val pySystemPrompt = escapePython(systemPrompt)
    val pyResultCol = escapePython(if (resultColumn == null || resultColumn.trim.isEmpty) "hf_response" else resultColumn)

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
       |        assert prompt_col in table.columns, (
       |            f"Prompt column '{prompt_col}' not found in input table. "
       |            f"Available columns: {list(table.columns)}"
       |        )
       |
       |        # --- handle result column conflict: overwrite policy ---
       |        # If resultColumn already exists in the table, it will be overwritten.
       |        # This is by design so that re-runs do not fail.
       |
       |        # --- handle empty table ---
       |        if table.empty:
       |            table[result_col] = pd.Series(dtype="object")
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
       |        yield table
       |""".stripMargin
  }

  override def operatorInfo: OperatorInfo =
    OperatorInfo(
      "Hugging Face Text Generation",
      "Call a Hugging Face text-generation model via the Inference API for each row",
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
