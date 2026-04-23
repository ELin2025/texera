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

import { Component, inject, OnChanges } from "@angular/core";
import { NZ_MODAL_DATA, NzModalRef } from "ng-zorro-antd/modal";
import { WorkflowResultService } from "../../service/workflow-result/workflow-result.service";
import { UntilDestroy, untilDestroyed } from "@ngneat/until-destroy";
import { PanelResizeService } from "../../service/workflow-result/panel-resize/panel-resize.service";
import { NotificationService } from "../../../common/service/notification/notification.service";

/**
 *
 * The pop-up window that will be
 *  displayed when the user clicks on a specific row
 *  to show the displays of that row.
 *
 * User can exit the pop-up window by
 *  1. Clicking the dismiss button on the top-right hand corner
 *      of the Modal
 *  2. Clicking the `Close` button at the bottom-right
 *  3. Clicking any shaded area that is not the pop-up window
 *  4. Pressing `Esc` button on the keyboard
 */
@UntilDestroy()
@Component({
  selector: "texera-row-modal-content",
  templateUrl: "./result-panel-modal.component.html",
  styleUrls: ["./result-panel-model.component.scss"],
})
export class RowModalComponent implements OnChanges {
  // Index of current displayed row in currentResult
  readonly modalData: { operatorId: string; rowIndex: number; rowData?: Record<string, unknown> } = inject(NZ_MODAL_DATA);
  readonly operatorId: string = this.modalData.operatorId;
  rowIndex: number = this.modalData.rowIndex;
  currentDisplayRowData: Record<string, unknown> = {};

  constructor(
    public modal: NzModalRef<any, number>,
    private workflowResultService: WorkflowResultService,
    private resizeService: PanelResizeService,
    private notificationService: NotificationService
  ) {
    this.currentDisplayRowData = this.modalData.rowData ?? {};
    this.ngOnChanges();
  }

  ngOnChanges(): void {
    if (this.modalData.rowData && this.rowIndex === this.modalData.rowIndex) {
      this.currentDisplayRowData = this.modalData.rowData;
      return;
    }
    this.workflowResultService
      .getPaginatedResultService(this.operatorId)
      ?.selectTuple(this.rowIndex, this.resizeService.pageSize)
      .pipe(untilDestroyed(this))
      .subscribe(res => {
        this.currentDisplayRowData = res.tuple;
      });
  }

  get rowEntries(): Array<{ key: string; value: string }> {
    return Object.entries(this.currentDisplayRowData).map(([key, value]) => ({
      key,
      value: this.stringifyValue(value),
    }));
  }

  get prettyRowJson(): string {
    return JSON.stringify(this.currentDisplayRowData, null, 2);
  }

  copyText(value: string): void {
    navigator.clipboard.writeText(value).then(
      () => this.notificationService.success("Copied to clipboard."),
      () => this.fallbackCopyText(value)
    );
  }

  private stringifyValue(value: unknown): string {
    if (typeof value === "string") {
      return value;
    }
    if (value === null || value === undefined) {
      return String(value);
    }
    return JSON.stringify(value, null, 2);
  }

  private fallbackCopyText(value: string): void {
    const textArea = document.createElement("textarea");
    textArea.value = value;
    textArea.style.position = "fixed";
    textArea.style.opacity = "0";
    document.body.appendChild(textArea);
    textArea.focus();
    textArea.select();
    try {
      const copied = document.execCommand("copy");
      if (copied) {
        this.notificationService.success("Copied to clipboard.");
      } else {
        this.notificationService.error("Copy failed.");
      }
    } catch {
      this.notificationService.error("Copy failed.");
    } finally {
      document.body.removeChild(textArea);
    }
  }
}
