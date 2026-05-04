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
import { CommonModule } from "@angular/common";
import { NZ_MODAL_DATA, NzModalRef } from "ng-zorro-antd/modal";
import { NzButtonModule } from "ng-zorro-antd/button";
import { NzIconModule } from "ng-zorro-antd/icon";
import { WorkflowResultService } from "../../service/workflow-result/workflow-result.service";
import { UntilDestroy, untilDestroyed } from "@ngneat/until-destroy";
import { PanelResizeService } from "../../service/workflow-result/panel-resize/panel-resize.service";
import { NgxJsonViewerModule } from "ngx-json-viewer";
import { NotificationService } from "../../../common/service/notification/notification.service";
import { isVideoUrl, isImageUrl } from "src/app/common/util/media-type.util";

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
  imports: [CommonModule, NzButtonModule, NzIconModule, NgxJsonViewerModule],
})
export class RowModalComponent implements OnChanges {
  // Index of current displayed row in currentResult
  readonly operatorId: string = inject(NZ_MODAL_DATA).operatorId;
  rowIndex: number = inject(NZ_MODAL_DATA).rowIndex;
  currentDisplayRowData: Record<string, unknown> = {};

  constructor(
    public modal: NzModalRef<any, number>,
    private workflowResultService: WorkflowResultService,
    private resizeService: PanelResizeService,
    private notificationService: NotificationService
  ) {
    this.ngOnChanges();
  }

  get prettyRowJson(): string {
    return JSON.stringify(this.currentDisplayRowData, null, 2);
  }

  get rowEntries(): { key: string; value: string; isVideo: boolean; isImage: boolean }[] {
    return Object.entries(this.currentDisplayRowData).map(([key, val]) => {
      const value = typeof val === "string" ? val : JSON.stringify(val);
      return {
        key,
        value,
        isVideo: typeof val === "string" && isVideoUrl(val),
        isImage: typeof val === "string" && isImageUrl(val),
      };
    });
  }

  copyText(text: string): void {
    navigator.clipboard.writeText(text).then(
      () => this.notificationService.success("Copied to clipboard"),
      () => this.notificationService.error("Failed to copy")
    );
  }

  ngOnChanges(): void {
    this.workflowResultService
      .getPaginatedResultService(this.operatorId)
      ?.selectTuple(this.rowIndex, this.resizeService.pageSize)
      .pipe(untilDestroyed(this))
      .subscribe(res => {
        this.currentDisplayRowData = res.tuple;
      });
  }
}
