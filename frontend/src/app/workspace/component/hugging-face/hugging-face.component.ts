/**
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

import { Component, OnInit, OnDestroy, ChangeDetectorRef } from "@angular/core";
import { FieldType, FieldTypeConfig } from "@ngx-formly/core";
import { HttpClient } from "@angular/common/http";
import { AppSettings } from "../../../common/app-setting";
import { Subscription } from "rxjs";

export interface HuggingFaceModelOption {
  id: string;
  label: string;
  pipeline_tag?: string;
  downloads?: number;
  likes?: number;
}

// ── NLP task → Hugging Face pipeline_tag mapping ──
export const TASK_TAG_MAP: Record<string, string> = {
  "Text Classification": "text-classification",
  "Token Classification": "token-classification",
  "Table Question Answering": "table-question-answering",
  "Question Answering": "question-answering",
  "Zero-Shot Classification": "zero-shot-classification",
  "Translation": "translation",
  "Summarization": "summarization",
  "Feature Extraction": "feature-extraction",
  "Text Generation": "text-generation",
  "Fill-Mask": "fill-mask",
  "Sentence Similarity": "sentence-similarity",
  "Text Ranking": "text-ranking",
};

export const TASK_NAMES = Object.keys(TASK_TAG_MAP);

// ── Reverse map: pipeline_tag → display name ──
const TAG_TASK_MAP: Record<string, string> = {};
for (const [name, tag] of Object.entries(TASK_TAG_MAP)) {
  TAG_TASK_MAP[tag] = name;
}

// ──────────────────────────────────────────────────────────────
// Module-level cache, keyed by pipeline_tag.
// Each tag is fetched at most ONCE per browser session.
// ──────────────────────────────────────────────────────────────
const modelCacheByTag: Map<string, HuggingFaceModelOption[]> = new Map();
const inFlightByTag: Map<string, Subscription> = new Map();
const errorByTag: Map<string, string> = new Map();

/** Clear all cached data (useful for tests or manual invalidation). */
export function invalidateHuggingFaceModelCache(): void {
  modelCacheByTag.clear();
  errorByTag.clear();
  inFlightByTag.forEach(sub => sub.unsubscribe());
  inFlightByTag.clear();
}

@Component({
  selector: "texera-hugging-face-model-select",
  templateUrl: "./hugging-face.component.html",
  styleUrls: ["hugging-face.component.scss"],
})
export class HuggingFaceComponent extends FieldType<FieldTypeConfig> implements OnInit, OnDestroy {
  // ── Task chip state ──
  taskNames = TASK_NAMES;
  selectedTask = "Text Generation";

  // ── Model dropdown state ──
  models: HuggingFaceModelOption[] = [];
  filteredModels: HuggingFaceModelOption[] = [];
  loading = false;
  errorMessage: string | null = null;

  // Custom filter: always true because we filter locally in onSearch
  nzFilterOptionFn = (): boolean => true;

  private subscription: Subscription | null = null;
  private loadingTimer: any = null;

  constructor(private http: HttpClient, private cdr: ChangeDetectorRef) {
    super();
  }

  ngOnInit(): void {
    // Restore task from saved operator state
    const savedTag = this.getCurrentTaskTag();
    if (savedTag && TAG_TASK_MAP[savedTag]) {
      this.selectedTask = TAG_TASK_MAP[savedTag];
    }
    // Always persist current selected task into hidden `task` field.
    this.persistTaskSelection(this.selectedTask);
    this.loadModelsForTask(this.selectedTask);
  }

  ngOnDestroy(): void {
    this.clearLoadingTimer();
    if (this.subscription) {
      this.subscription.unsubscribe();
      this.subscription = null;
    }
  }

  /** Called when a task is selected from the dropdown. */
  onTaskSelected(taskName: string): void {
    this.selectedTask = taskName;
    this.persistTaskSelection(taskName);
    this.formControl.setValue(null);
    this.clearTaskSpecificFields();
    this.loadModelsForTask(taskName);
  }

  /** Clear all task-specific fields so stale values don't leak across tasks. */
  private clearTaskSpecificFields(): void {
    const fieldsToReset = ["contextColumn", "candidateLabels", "sentencesColumn"];
    for (const key of fieldsToReset) {
      const ctrl = this.field.form?.get(key) ?? this.formControl?.parent?.get(key);
      if (ctrl) {
        ctrl.setValue("");
      }
      if (this.model) {
        (this.model as any)[key] = "";
      }
    }
  }

  /** Load models for the given task, using per-tag cache. */
  loadModelsForTask(taskName: string): void {
    const tag = TASK_TAG_MAP[taskName] || "text-generation";

    // ── Cancel any pending loading timer ──
    this.clearLoadingTimer();
    this.loading = false;
    this.errorMessage = null;

    // ── Fast path: already cached ──
    if (modelCacheByTag.has(tag)) {
      this.applyModels(modelCacheByTag.get(tag)!);
      return;
    }

    // ── Previous error for this tag → show it, allow retry ──
    if (errorByTag.has(tag)) {
      this.errorMessage = errorByTag.get(tag)!;
      this.models = [];
      this.filteredModels = [];
      return;
    }

    // ── Cancel previous subscription ──
    if (this.subscription) {
      this.subscription.unsubscribe();
      this.subscription = null;
    }

    // ── Show loader only after 300ms delay ──
    this.models = [];
    this.filteredModels = [];
    this.loadingTimer = setTimeout(() => {
      this.loading = true;
      this.cdr.detectChanges();
    }, 300);

    // ── Fire request ──
    this.subscription = this.http
      .get<HuggingFaceModelOption[]>(
        `${AppSettings.getApiEndpoint()}/huggingface/models?task=${encodeURIComponent(tag)}&limit=100`
      )
      .subscribe({
        next: models => {
          modelCacheByTag.set(tag, models);
          inFlightByTag.delete(tag);
          this.clearLoadingTimer();
          this.loading = false;
          this.applyModels(models);
        },
        error: err => {
          console.error(`Failed to load HuggingFace models for task '${tag}':`, err);
          const msg = "Failed to load models. Click retry to try again.";
          errorByTag.set(tag, msg);
          inFlightByTag.delete(tag);
          this.clearLoadingTimer();
          this.loading = false;
          this.errorMessage = msg;
        },
      });

    inFlightByTag.set(tag, this.subscription);
  }

  private clearLoadingTimer(): void {
    if (this.loadingTimer) {
      clearTimeout(this.loadingTimer);
      this.loadingTimer = null;
    }
  }

  private getCurrentTaskTag(): string | undefined {
    const fromModel = this.model?.task;
    if (typeof fromModel === "string" && fromModel.trim().length > 0) {
      return fromModel;
    }
    const fromParentControl = this.formControl?.parent?.get("task")?.value;
    if (typeof fromParentControl === "string" && fromParentControl.trim().length > 0) {
      return fromParentControl;
    }
    const fromFieldForm = this.field.form?.get("task")?.value;
    if (typeof fromFieldForm === "string" && fromFieldForm.trim().length > 0) {
      return fromFieldForm;
    }
    return undefined;
  }

  private persistTaskSelection(taskName: string): void {
    const tag = TASK_TAG_MAP[taskName] || "text-generation";

    // Update hidden task control regardless of how formly nests this field.
    const taskControlFromField = this.field.form?.get("task");
    if (taskControlFromField) {
      taskControlFromField.setValue(tag);
    }

    const taskControlFromParent = this.formControl?.parent?.get("task");
    if (taskControlFromParent) {
      taskControlFromParent.setValue(tag);
    }

    // Also sync backing model so operator JSON persists consistently.
    if (this.model) {
      this.model.task = tag;
    }

    // Force formly expression re-evaluation so task-specific fields update immediately.
    this.field.options?.detectChanges?.(this.field);
  }

  /** Retry loading models for the currently selected task. */
  retryLoad(): void {
    const tag = TASK_TAG_MAP[this.selectedTask] || "text-generation";
    errorByTag.delete(tag);
    this.loadModelsForTask(this.selectedTask);
  }

  /** Client-side search/filter within the loaded model list. */
  onSearch(searchText: string): void {
    if (!searchText) {
      this.filteredModels = [...this.models];
    } else {
      const lower = searchText.toLowerCase();
      this.filteredModels = this.models.filter(m => m.id.toLowerCase().includes(lower));
    }
  }

  onModelSelected(modelId: string): void {
    this.formControl.setValue(modelId);
  }

  // ── private helpers ──

  /** Apply a model list, preserving the current formControl value if not in list. */
  private applyModels(models: HuggingFaceModelOption[]): void {
    this.models = [...models];

    const currentValue = this.formControl.value;
    if (currentValue && !models.find(m => m.id === currentValue)) {
      this.models = [{ id: currentValue, label: currentValue }, ...this.models];
    }

    this.filteredModels = [...this.models];
  }
}
