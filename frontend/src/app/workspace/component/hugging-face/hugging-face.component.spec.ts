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

import { ComponentFixture, TestBed, waitForAsync } from "@angular/core/testing";
import { FormControl, ReactiveFormsModule, FormsModule } from "@angular/forms";
import { HuggingFaceComponent, invalidateHuggingFaceModelCache, TASK_TAG_MAP } from "./hugging-face.component";
import { HttpClientTestingModule, HttpTestingController } from "@angular/common/http/testing";
import { commonTestProviders } from "../../../common/testing/test-utils";
import { AppSettings } from "../../../common/app-setting";

describe("HuggingFaceComponent", () => {
  let component: HuggingFaceComponent;
  let fixture: ComponentFixture<HuggingFaceComponent>;
  let httpMock: HttpTestingController;

  beforeEach(waitForAsync(() => {
    TestBed.configureTestingModule({
      declarations: [HuggingFaceComponent],
      imports: [ReactiveFormsModule, FormsModule, HttpClientTestingModule],
      providers: [...commonTestProviders],
    }).compileComponents();
  }));

  beforeEach(() => {
    // Clear the module-level cache before each test so every test starts fresh
    invalidateHuggingFaceModelCache();

    fixture = TestBed.createComponent(HuggingFaceComponent);
    component = fixture.componentInstance;
    component.field = { props: {}, formControl: new FormControl() };
    httpMock = TestBed.inject(HttpTestingController);
    fixture.detectChanges();
  });

  afterEach(() => {
    httpMock.verify();
  });

  it("should create and default to Text Generation", () => {
    const req = httpMock.expectOne(
      `${AppSettings.getApiEndpoint()}/huggingface/models?task=text-generation&limit=100`
    );
    req.flush([]);
    expect(component).toBeTruthy();
    expect(component.selectedTask).toBe("Text Generation");
  });

  it("should load models for the default task on init", () => {
    const mockModels = [
      { id: "Qwen/Qwen2.5-72B-Instruct", label: "Qwen/Qwen2.5-72B-Instruct" },
      { id: "meta-llama/Llama-3.1-8B-Instruct", label: "meta-llama/Llama-3.1-8B-Instruct" },
    ];

    const req = httpMock.expectOne(
      `${AppSettings.getApiEndpoint()}/huggingface/models?task=text-generation&limit=100`
    );
    req.flush(mockModels);

    expect(component.models.length).toBe(2);
    expect(component.filteredModels.length).toBe(2);
    expect(component.loading).toBe(false);
  });

  it("should switch tasks and fetch models for new task", () => {
    // Flush the initial Text Generation request
    const initialReq = httpMock.expectOne(
      `${AppSettings.getApiEndpoint()}/huggingface/models?task=text-generation&limit=100`
    );
    initialReq.flush([]);

    // Switch to Summarization
    component.onTaskSelected("Summarization");

    const sumReq = httpMock.expectOne(
      `${AppSettings.getApiEndpoint()}/huggingface/models?task=summarization&limit=100`
    );
    sumReq.flush([{ id: "facebook/bart-large-cnn", label: "facebook/bart-large-cnn" }]);

    expect(component.selectedTask).toBe("Summarization");
    expect(component.models.length).toBe(1);
    expect(component.models[0].id).toBe("facebook/bart-large-cnn");
  });

  it("should use cache when switching back to a previously loaded task", () => {
    // Flush initial Text Generation
    const initialReq = httpMock.expectOne(
      `${AppSettings.getApiEndpoint()}/huggingface/models?task=text-generation&limit=100`
    );
    initialReq.flush([{ id: "model-a", label: "model-a" }]);

    // Switch to Summarization
    component.onTaskSelected("Summarization");
    const sumReq = httpMock.expectOne(
      `${AppSettings.getApiEndpoint()}/huggingface/models?task=summarization&limit=100`
    );
    sumReq.flush([{ id: "model-b", label: "model-b" }]);

    // Switch back to Text Generation — should NOT fire another request
    component.onTaskSelected("Text Generation");
    // httpMock.verify() in afterEach will fail if an unexpected request was made

    expect(component.models.length).toBe(1);
    expect(component.models[0].id).toBe("model-a");
  });

  it("should handle API error gracefully", () => {
    const req = httpMock.expectOne(
      `${AppSettings.getApiEndpoint()}/huggingface/models?task=text-generation&limit=100`
    );
    req.error(new ErrorEvent("Network error"));

    expect(component.errorMessage).toBeTruthy();
    expect(component.models.length).toBe(0);
    expect(component.loading).toBe(false);
  });

  it("should keep current value as fallback when not in list", () => {
    component.formControl.setValue("my-custom/model");

    const req = httpMock.expectOne(
      `${AppSettings.getApiEndpoint()}/huggingface/models?task=text-generation&limit=100`
    );
    req.flush([{ id: "Qwen/Qwen2.5-72B-Instruct", label: "Qwen/Qwen2.5-72B-Instruct" }]);

    expect(component.models.length).toBe(2);
    expect(component.models[0].id).toBe("my-custom/model");
  });

  it("should filter models on search", () => {
    const mockModels = [
      { id: "Qwen/Qwen2.5-72B-Instruct", label: "Qwen/Qwen2.5-72B-Instruct" },
      { id: "meta-llama/Llama-3.1-8B-Instruct", label: "meta-llama/Llama-3.1-8B-Instruct" },
    ];

    const req = httpMock.expectOne(
      `${AppSettings.getApiEndpoint()}/huggingface/models?task=text-generation&limit=100`
    );
    req.flush(mockModels);

    component.onSearch("qwen");
    expect(component.filteredModels.length).toBe(1);
    expect(component.filteredModels[0].id).toBe("Qwen/Qwen2.5-72B-Instruct");
  });
});
