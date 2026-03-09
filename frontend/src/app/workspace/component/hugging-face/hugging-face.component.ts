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

import { Component, OnInit, OnDestroy } from "@angular/core";
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

  constructor(private http: HttpClient) {
    super();
  }

  ngOnInit(): void {
    this.loadModelsForTask(this.selectedTask);
  }

  ngOnDestroy(): void {
    if (this.subscription) {
      this.subscription.unsubscribe();
      this.subscription = null;
    }
  }

  /** Called when a task chip is clicked. */
  onTaskSelected(taskName: string): void {
    this.selectedTask = taskName;
    this.loadModelsForTask(taskName);
  }

  /** Load models for the given task, using per-tag cache. */
  loadModelsForTask(taskName: string): void {
    const tag = TASK_TAG_MAP[taskName] || "text-generation";

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

    // ── In-flight request exists → poll for result ──
    if (inFlightByTag.has(tag) && !inFlightByTag.get(tag)!.closed) {
      this.loading = true;
      this.errorMessage = null;
      const poll = setInterval(() => {
        if (modelCacheByTag.has(tag)) {
          clearInterval(poll);
          this.loading = false;
          this.applyModels(modelCacheByTag.get(tag)!);
        } else if (errorByTag.has(tag)) {
          clearInterval(poll);
          this.loading = false;
          this.errorMessage = errorByTag.get(tag)!;
        }
      }, 100);
      return;
    }

    // ── Fire a new request ──
    this.loading = true;
    this.errorMessage = null;
    this.models = [];
    this.filteredModels = [];

    this.subscription = this.http
      .get<HuggingFaceModelOption[]>(
        `${AppSettings.getApiEndpoint()}/huggingface/models?task=${encodeURIComponent(tag)}&limit=100`
      )
      .subscribe({
        next: models => {
          modelCacheByTag.set(tag, models);
          inFlightByTag.delete(tag);
          this.loading = false;
          this.applyModels(models);
        },
        error: err => {
          console.error(`Failed to load HuggingFace models for task '${tag}':`, err);
          const msg = "Failed to load models. Click retry to try again.";
          errorByTag.set(tag, msg);
          inFlightByTag.delete(tag);
          this.loading = false;
          this.errorMessage = msg;
        },
      });

    inFlightByTag.set(tag, this.subscription);
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
