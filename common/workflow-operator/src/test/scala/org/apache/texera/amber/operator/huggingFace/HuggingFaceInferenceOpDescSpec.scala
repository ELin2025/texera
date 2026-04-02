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

import org.apache.texera.amber.core.tuple.{Attribute, AttributeType, Schema}
import org.apache.texera.amber.core.workflow.PortIdentity
import org.scalatest.BeforeAndAfter
import org.scalatest.flatspec.AnyFlatSpec

class HuggingFaceInferenceOpDescSpec extends AnyFlatSpec with BeforeAndAfter {


  var opDesc: HuggingFaceInferenceOpDesc = _

  before {
    opDesc = new HuggingFaceInferenceOpDesc()
    // Set required defaults
    opDesc.modelId = "Qwen/Qwen2.5-72B-Instruct"
    opDesc.promptColumn = "text"
    opDesc.systemPrompt = "You are a helpful assistant."
    opDesc.maxNewTokens = 256
    opDesc.temperature = 0.7
    opDesc.resultColumn = "hf_response"
    opDesc.hfApiToken = "test-token"
  }

  // ===================== Validation Tests =====================

  it should "throw AssertionError when promptColumn is empty" in {
    opDesc.promptColumn = ""
    assertThrows[AssertionError] {
      opDesc.generatePythonCode()
    }
  }

  it should "throw AssertionError when promptColumn is null" in {
    opDesc.promptColumn = null
    assertThrows[AssertionError] {
      opDesc.generatePythonCode()
    }
  }

  it should "throw AssertionError when modelId is null" in {
    opDesc.modelId = null
    assertThrows[AssertionError] {
      opDesc.generatePythonCode()
    }
  }

  it should "throw AssertionError when modelId is empty string" in {
    opDesc.modelId = ""
    assertThrows[AssertionError] {
      opDesc.generatePythonCode()
    }
  }

  // ===================== Generated Python Structure Tests =====================

  it should "generate Python code containing required imports" in {
    val code = opDesc.generatePythonCode()
    assert(code.contains("import os"))
    assert(code.contains("import requests"))
    assert(code.contains("import pandas as pd"))
    assert(code.contains("from pytexera import *"))
  }

  it should "generate Python code with the correct HF API URL" in {
    val code = opDesc.generatePythonCode()
    assert(code.contains("https://router.huggingface.co/v1/chat/completions"))
  }

  it should "generate Python code that includes configured modelId" in {
    opDesc.modelId = "mistralai/Mistral-7B-Instruct-v0.3"
    val code = opDesc.generatePythonCode()
    assert(code.contains("mistralai/Mistral-7B-Instruct-v0.3"))
  }

  it should "generate Python code that includes configured systemPrompt" in {
    opDesc.systemPrompt = "You are a coding assistant."
    val code = opDesc.generatePythonCode()
    assert(code.contains("You are a coding assistant."))
  }

  it should "generate Python code that includes configured resultColumn" in {
    opDesc.resultColumn = "my_result"
    val code = opDesc.generatePythonCode()
    assert(code.contains("my_result"))
  }

  it should "generate Python code that includes configured maxNewTokens" in {
    opDesc.maxNewTokens = 512
    val code = opDesc.generatePythonCode()
    assert(code.contains("MAX_NEW_TOKENS = 512"))
  }

  it should "generate Python code that includes configured temperature" in {
    opDesc.temperature = 1.5
    val code = opDesc.generatePythonCode()
    assert(code.contains("TEMPERATURE    = 1.5"))
  }

  it should "generate Python code using UDFTableOperator and process_table" in {
    val code = opDesc.generatePythonCode()
    assert(code.contains("class ProcessTableOperator(UDFTableOperator)"))
    assert(code.contains("def process_table(self, table: Table, port: int)"))
  }

  // ===================== Token Logic Tests =====================

  it should "embed hfApiToken in generated code when provided" in {
    opDesc.hfApiToken = "hf_abc123"
    val code = opDesc.generatePythonCode()
    assert(code.contains("hf_abc123"))
  }

  it should "include HF_TOKEN env var fallback logic" in {
    opDesc.hfApiToken = ""
    val code = opDesc.generatePythonCode()
    assert(code.contains("os.environ.get(\"HF_TOKEN\"")))
  }

  it should "include ValueError raise when no token is available" in {
    val code = opDesc.generatePythonCode()
    assert(code.contains("raise ValueError"))
    assert(code.contains("Hugging Face API token is not set"))
  }

  // ===================== Error Handling Snippets Tests =====================

  it should "include empty table handling code path" in {
    val code = opDesc.generatePythonCode()
    assert(code.contains("if table.empty"))
  }

  it should "include missing promptColumn assertion" in {
    val code = opDesc.generatePythonCode()
    assert(code.contains("assert prompt_col in table.columns"))
  }

  it should "include HTTP 429 rate-limit handling" in {
    val code = opDesc.generatePythonCode()
    assert(code.contains("status_code == 429"))
    assert(code.contains("rate limit hit"))
  }

  it should "include HTTP 401 auth error handling" in {
    val code = opDesc.generatePythonCode()
    assert(code.contains("status_code == 401"))
    assert(code.contains("Invalid HF API token"))
  }

  it should "include generic non-200 error handling" in {
    val code = opDesc.generatePythonCode()
    assert(code.contains("status_code != 200"))
    assert(code.contains("HF API error"))
  }

  it should "include per-row failure isolation" in {
    val code = opDesc.generatePythonCode()
    // Per-row exceptions are caught and result is set to empty string
    assert(code.contains("results.append(\"\")"))
  }

  // ===================== Edge Case Tests =====================

  it should "allow temperature = 0.0 (deterministic)" in {
    opDesc.temperature = 0.0
    val code = opDesc.generatePythonCode()
    assert(code.contains("TEMPERATURE    = 0.0"))
  }

  it should "clamp maxNewTokens to 4096 maximum" in {
    opDesc.maxNewTokens = 9999
    val code = opDesc.generatePythonCode()
    assert(code.contains("MAX_NEW_TOKENS = 4096"))
  }

  it should "clamp maxNewTokens to 1 minimum" in {
    opDesc.maxNewTokens = -5
    val code = opDesc.generatePythonCode()
    assert(code.contains("MAX_NEW_TOKENS = 1"))
  }

  it should "clamp temperature to 2.0 maximum" in {
    opDesc.temperature = 5.0
    val code = opDesc.generatePythonCode()
    assert(code.contains("TEMPERATURE    = 2.0"))
  }

  it should "use default resultColumn 'hf_response' when resultColumn is empty" in {
    opDesc.resultColumn = ""
    val code = opDesc.generatePythonCode()
    assert(code.contains("RESULT_COLUMN  = \"hf_response\""))
  }

  it should "handle NaN/None prompt values in generated Python" in {
    val code = opDesc.generatePythonCode()
    assert(code.contains("pd.isna(prompt_value)"))
  }

  // ===================== Schema Tests =====================

  it should "add resultColumn to the output schema" in {
    val inputSchema = new Schema(
      new Attribute("text", AttributeType.STRING),
      new Attribute("id", AttributeType.INTEGER)
    )
    opDesc.resultColumn = "hf_response"
    val outputSchemas = opDesc.getOutputSchemas(Map(PortIdentity() -> inputSchema))
    val outputSchema = outputSchemas.values.head
    assert(outputSchema.containsAttribute("hf_response"))
    assert(outputSchema.getAttribute("hf_response").getType == AttributeType.STRING)
    // Original columns should still be present
    assert(outputSchema.containsAttribute("text"))
    assert(outputSchema.containsAttribute("id"))
  }

  it should "default resultColumn to 'hf_response' in schema when null" in {
    val inputSchema = new Schema(
      new Attribute("text", AttributeType.STRING)
    )
    opDesc.resultColumn = null
    val outputSchemas = opDesc.getOutputSchemas(Map(PortIdentity() -> inputSchema))
    val outputSchema = outputSchemas.values.head
    assert(outputSchema.containsAttribute("hf_response"))
  }

  // ===================== Operator Info Tests =====================

  it should "have correct operator info" in {
    val info = opDesc.operatorInfo
    assert(info.userFriendlyName == "Hugging Face")
    assert(info.inputPorts.nonEmpty)
    assert(info.outputPorts.nonEmpty)
  }

  it should "escape special characters in Python strings" in {
    opDesc.systemPrompt = "Say \"hello\" and\nnewline"
    val code = opDesc.generatePythonCode()
    // Should contain escaped quotes and newlines
    assert(code.contains("\\\"hello\\\""))
    assert(code.contains("\\n"))
  }
}
