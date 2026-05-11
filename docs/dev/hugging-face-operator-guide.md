# Hugging Face Operator - Developer Maintenance Guide

This document explains the architecture of the Hugging Face Inference operator in Texera and provides step-by-step instructions for common maintenance tasks: adding a new task, adding a new field, and changing how a task's parameters work.

---

## Architecture Overview

The operator spans four layers. A user-facing change typically touches all four.

```
┌─────────────────────────────────────────────────────────┐
│  1. Scala Descriptor (fields, validation, Python codegen)│
│     HuggingFaceInferenceOpDesc.scala                    │
├─────────────────────────────────────────────────────────┤
│  2. Frontend Field Visibility (which fields show per task)│
│     operator-property-edit-frame.component.ts           │
├─────────────────────────────────────────────────────────┤
│  3. Frontend Task/Model Selector (task list, model list) │
│     hugging-face.component.ts                           │
├─────────────────────────────────────────────────────────┤
│  4. Tests                                                │
│     HuggingFaceInferenceOpDescSpec.scala                 │
└─────────────────────────────────────────────────────────┘
```

### How a field goes from Scala to the UI

1. A `var` in `HuggingFaceInferenceOpDesc` annotated with `@JsonProperty` is serialized into a JSON Schema.
2. The Angular frontend receives the schema and renders it as a Formly form.
3. `operator-property-edit-frame.component.ts` intercepts HuggingFace fields and attaches **expressions** that hide/show them based on the selected task.
4. When the workflow runs, `generatePythonCode()` injects every field value into a Python string template that becomes the runtime UDF.

---

## Key Files

| File | Role |
|------|------|
| `common/workflow-operator/.../huggingFace/HuggingFaceInferenceOpDesc.scala` | Operator descriptor: all fields, validation, Python code generation |
| `common/workflow-operator/.../huggingFace/HuggingFaceInferenceOpDescSpec.scala` | Unit tests for validation and generated Python |
| `frontend/.../operator-property-edit-frame/operator-property-edit-frame.component.ts` | Dynamic field visibility per task (lines ~765-928) |
| `frontend/.../hugging-face/hugging-face.component.ts` | Task selector, model browser, state preservation |
| `frontend/.../hugging-face-image-upload/hugging-face-image-upload.component.ts` | Image upload with compression |
| `frontend/.../hugging-face-audio-upload/hugging-face-audio-upload.component.ts` | Audio upload with backend storage |
| `frontend/.../common/formly/formly-config.ts` | Registers custom Formly field types (`huggingface`, `huggingface-image-upload`, `huggingface-audio-upload`) |
| `amber/.../web/resource/HuggingFaceModelResource.scala` | REST API: model/task browsing, media upload, proxy |
| `common/workflow-operator/.../metadata/OperatorGroupConstants.scala` | Registers the "Hugging Face" operator group |

---

## Task Categories

The operator classifies tasks into categories that determine which fields are visible and how the Python runtime processes input/output.

| Category | Tasks | Key Behavior |
|----------|-------|--------------|
| **Text generation** | `text-generation` | Uses `/v1/chat/completions` endpoint; shows system prompt, temperature, max tokens |
| **Text-in, text-out** | `text-classification`, `token-classification`, `translation`, `summarization`, `fill-mask`, `feature-extraction` | Uses HF pipeline endpoint; needs prompt column only |
| **Question answering** | `question-answering` | Needs prompt column + context column |
| **Table QA** | `table-question-answering` | Sends entire table as structured input |
| **Zero-shot text** | `zero-shot-classification` | Needs prompt column + candidate labels |
| **Sentence pair** | `sentence-similarity`, `text-ranking` | Needs prompt column + sentences column |
| **Image-only** | `image-classification`, `object-detection`, `image-segmentation`, `image-to-text` | Needs image upload; no prompt column |
| **Image + prompt** | `visual-question-answering`, `document-question-answering`, `zero-shot-image-classification` | Needs image upload + optional prompt column |
| **Audio-only** | `automatic-speech-recognition`, `audio-classification` | Needs audio upload; no prompt column |
| **Text-to-media** | `text-to-image`, `text-to-video`, `text-to-speech` | Needs prompt column; returns binary media |

These categories are defined as `Set`s in the Scala descriptor (`imageOnlyTasks`, `imagePromptTasks`, `audioOnlyTasks`) and mirrored as arrays in the frontend visibility logic.

---

## Media Input Architecture

Images and audio are handled differently. Understanding this is important when adding a new media type.

### Images: inline data URLs

The image upload component (`hugging-face-image-upload.component.ts`) compresses images client-side and embeds them as `data:image/...;base64,...` strings directly in the operator's `imageInput` field. The compression pipeline targets a max data URL size of ~45KB (resizes down to 512px then 160px, reduces JPEG quality from 0.75 to 0.35). This means the image data travels through the JSON schema as a string value and is embedded directly in the generated Python code.

### Audio: server-side file storage

Audio files are too large to embed inline. The audio upload component (`hugging-face-audio-upload.component.ts`) POSTs the raw bytes to `/api/huggingface/upload-audio`, which stores them in a temp directory (`/tmp/texera-hf-audio/`). The `audioInput` field stores the **file path** on the server, not the audio data. The generated Python code reads the file from disk at runtime.

### `imageInput` vs `inputImageColumn` mutual exclusivity

For image tasks, users can either upload an image directly (`imageInput`) or select a column from the input table that contains image data (`inputImageColumn`). In the frontend, when `inputImageColumn` has a value, the `imageInput` upload field is hidden. The validator only requires one of the two to be filled.

### Adding a new media type

If you need to support a new media type (e.g., video upload):
1. Decide whether it should be **inline** (small, < 45KB after encoding) or **server-side** (large files).
2. For server-side: add upload/preview endpoints to `HuggingFaceModelResource.scala`, following the audio upload pattern.
3. Create a custom Formly component and register it in `formly-config.ts`.
4. Add the corresponding field to the Scala descriptor and wire up visibility in the property editor.

---

## Backend API (`HuggingFaceModelResource.scala`)

The operator has a backend REST API at `/api/huggingface/` that handles model discovery and media management.

### Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/models?task=...&search=...` | Browse or search models for a task. Browse mode fetches all models (paginated internally) and caches them. Search mode forwards the query to HF Hub. |
| GET | `/tasks` | Fetch available pipeline tags from HF Hub, filtered to tasks with hosted inference. Cached for process lifetime. |
| POST | `/upload-audio?filename=...` | Upload raw audio bytes. Stores in `/tmp/texera-hf-audio/`, returns `{path, fileName}`. |
| GET | `/audio-preview?path=...` | Stream an uploaded audio file back. Path-validated to stay within the temp directory. |
| GET | `/media-proxy?url=...` | Proxy remote media through the backend to avoid CORS issues. Only allows `http(s)` URLs. |

### Caching

Both model lists and the task list are cached in `ConcurrentHashMap`s for the lifetime of the JVM process. There is **no cache invalidation** — if HuggingFace adds new models or tasks, the server must be restarted. This is worth knowing when debugging stale model lists.

### Authentication

The backend reads `HF_TOKEN` from the environment for server-side HF Hub API calls (model browsing, task listing). This is separate from the user-facing `hfApiToken` field, which is used at runtime for inference.

---
## Security: String Escaping

All user-supplied string values **must** pass through `escapePython()` before being interpolated into the generated Python code. This method escapes backslashes, double quotes, newlines, carriage returns, and tabs. Skipping this step would allow a user to inject arbitrary Python code through fields like `systemPrompt` or `modelId`.

When adding a new string field, always follow the existing pattern:
```scala
val pyMyField = escapePython(myField)
// then use $pyMyField in the string template
```

Integer and numeric fields (like `maxNewTokens`, `temperature`) don't need escaping since they're converted with `.toString` and clamped to valid ranges.

---

## Running Tests

```bash
# Run only the HuggingFace operator tests
sbt "workflow-operator/testOnly *HuggingFaceInferenceOpDescSpec"

# Run all workflow-operator tests
sbt "workflow-operator/test"
```

The test class uses ScalaTest's `AnyFlatSpec` style. The `before` block creates a fresh `HuggingFaceInferenceOpDesc` with sensible defaults before each test. Tests primarily call `generatePythonCode()` and assert on the generated Python string content.

---

## How To: Add a New Task

### Example: adding `image-text-to-text`

#### Step 1 - Scala descriptor

Decide which category the task belongs to. If it fits an existing category, add it to the corresponding `Set`. If it needs a new input type, define a new set.

```scala
// HuggingFaceInferenceOpDesc.scala
private val imagePromptTasks = Set(
  "visual-question-answering",
  "document-question-answering",
  "zero-shot-image-classification",
  "image-text-to-text"              // <-- add here
)
```

#### Step 2 - Python code generation

In `generateInferencePython()`, add the payload-building branch if the new task doesn't match any existing pattern. If it matches an existing category (e.g., it sends an image + question), no change is needed.

```scala
// Inside the generated Python, within the payload-building section:
// elif task == "image-text-to-text":
//     payload = { ... }
```

Also add the response-parsing branch in `_parse_response` if the API returns a novel format:

```scala
// elif task == "image-text-to-text":
//     return body.get("generated_text", json.dumps(body))
```

#### Step 3 - Frontend visibility

In `operator-property-edit-frame.component.ts`, add the task tag to the relevant arrays:

```typescript
// ~line 769
const imageInputTasks = [
  ...imageOnlyTasks,
  "visual-question-answering",
  "document-question-answering",
  "zero-shot-image-classification",
  "image-text-to-text",             // <-- add here
];
```

If the task requires `promptColumn`, add it to `promptRequiredTasks` too:

```typescript
const promptRequiredTasks = [
  ...
  "image-text-to-text",             // <-- add here
];
```

#### Step 4 - Static fallback task list

In `hugging-face.component.ts`, add the task to `STATIC_TASK_OPTIONS`:

```typescript
{ tag: "image-text-to-text", label: "Image Text to Text" },
```

The dynamic task list fetched from the API will include it automatically if HuggingFace supports it, but the static list serves as a fallback.

#### Step 5 - Tests

Add tests in `HuggingFaceInferenceOpDescSpec.scala`:

```scala
it should "allow image-text-to-text with image and prompt" in {
  opDesc.task = "image-text-to-text"
  opDesc.promptColumn = "text"
  opDesc.imageInput = "data:image/png;base64,abcd"
  val code = opDesc.generatePythonCode()
  assert(code.contains("IMAGE_INPUT"))
  assert(code.contains("PROMPT_COLUMN"))
}
```

---

## How To: Add a New Field

### Example: adding `topK` for ranking tasks

#### Step 1 - Scala field declaration

```scala
// HuggingFaceInferenceOpDesc.scala
@JsonProperty(value = "topK", required = false, defaultValue = "5")
@JsonSchemaTitle("Top K")
@JsonPropertyDescription("Number of top results to return")
var topK: Int = 5
```

**Annotation reference:**
- `@JsonProperty` - Makes it visible in JSON schema; `required` controls validation; `defaultValue` sets the form default.
- `@JsonSchemaTitle` - Display label in the UI form.
- `@JsonPropertyDescription` - Tooltip/help text in the UI.
- `@AutofillAttributeName` - (Optional) If the field should auto-populate from input table column names.

#### Step 2 - Inject into generated Python

In `generateInferencePython()`, pass it through and embed it in the Python template:

```scala
// Add to the method parameters
val pyTopK = topK.toString

// Add to the Python class constants section
|    TOP_K             = $pyTopK
```

Then use it in the payload-building section:

```python
elif task in ("text-ranking", "sentence-similarity"):
    payload = {
        "inputs": { ... },
        "parameters": {"top_k": self.TOP_K},
    }
```

#### Step 3 - Frontend visibility

In `operator-property-edit-frame.component.ts`, add a visibility rule:

```typescript
if (hfKey === "topK") {
  mappedField.expressions = {
    ...mappedField.expressions,
    hide: (field: FormlyFieldConfig) => {
      const t = getSelectedTask(field);
      return t !== "sentence-similarity" && t !== "text-ranking";
    },
  };
}
```

#### Step 4 - Task state preservation (optional)

If users should retain the value when switching tasks, add the field to `taskScopedKeys` in `hugging-face.component.ts`:

```typescript
private taskScopedKeys = [
  "modelId", "promptColumn", ..., "topK"  // <-- add here
];
```

#### Step 5 - Tests

```scala
it should "include topK in generated code for ranking tasks" in {
  opDesc.task = "text-ranking"
  opDesc.promptColumn = "text"
  opDesc.sentencesColumn = "sentences"
  opDesc.topK = 10
  val code = opDesc.generatePythonCode()
  assert(code.contains("TOP_K"))
}
```

---

## How To: Add a Custom Input Component

If a field needs special UI beyond a standard text/number input (like the image or audio uploaders):

1. **Create the Angular component** under `frontend/src/app/workspace/component/`. Follow the pattern of `hugging-face-image-upload` or `hugging-face-audio-upload`.

2. **Register it in Formly** in `formly-config.ts`:
   ```typescript
   { name: "my-custom-type", component: MyCustomComponent, wrappers: ["form-field"] }
   ```

3. **Override the field type** in `operator-property-edit-frame.component.ts`:
   ```typescript
   if (hfKey === "myField") {
     mappedField.type = "my-custom-type";
   }
   ```

4. **If it needs a backend endpoint** (e.g., file upload/storage), add it to `HuggingFaceModelResource.scala`.

---

## How To: Change a Task's Parameters

If a HuggingFace API changes how a task expects its payload:

1. **Update the payload branch** in `generateInferencePython()` inside `HuggingFaceInferenceOpDesc.scala`. Find the `elif task == "your-task":` block and modify the `payload = { ... }` dict.

2. **Update response parsing** in `_parse_response` if the response format changed.

3. **Update tests** to assert the new payload structure appears in the generated code.

The frontend visibility logic typically does not need changes unless the task now requires different fields.

---

## Provider Fallback System

The generated Python code includes a provider fallback mechanism:

1. `_resolve_providers()` queries the HF Hub API to discover which inference providers support the model.
2. Providers are sorted by `PROVIDER_COST_PRIORITY` (cheapest first).
3. `_post_with_fallback()` tries providers in order, skipping on retryable HTTP errors (400, 404, 422, 429, 502, 503).

**Two code generation paths exist:**
- `generateTextGenPython()` - For `text-generation` only. Uses the OpenAI-compatible `/v1/chat/completions` endpoint.
- `generateInferencePython()` - For all other tasks. Uses HF pipeline endpoints and handles binary request/response bodies.

If a new provider requires a different URL pattern or payload format, modify the `_post_with_fallback` method in the relevant codegen function.

---

## Checklist: Adding a New Task

- [ ] Classify the task into an existing category or define a new one
- [ ] Add to category `Set` in `HuggingFaceInferenceOpDesc.scala` if applicable
- [ ] Add payload-building branch in `generateInferencePython()`
- [ ] Add response-parsing branch in `_parse_response()`
- [ ] Add to frontend task arrays in `operator-property-edit-frame.component.ts`
- [ ] Add to `STATIC_TASK_OPTIONS` in `hugging-face.component.ts`
- [ ] Add unit tests in `HuggingFaceInferenceOpDescSpec.scala`
- [ ] Test end-to-end in the Texera UI

## Checklist: Adding a New Field

- [ ] Declare `var` with `@JsonProperty`, `@JsonSchemaTitle`, `@JsonPropertyDescription` in `HuggingFaceInferenceOpDesc.scala`
- [ ] Inject into generated Python via `escapePython()` and string interpolation
- [ ] Add `hide` expression in `operator-property-edit-frame.component.ts`
- [ ] Add to `taskScopedKeys` in `hugging-face.component.ts` if it should preserve state across task switches
- [ ] Add unit tests in `HuggingFaceInferenceOpDescSpec.scala`
