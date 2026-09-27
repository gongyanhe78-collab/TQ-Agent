<template>
  <div class="message-actions">
    <div v-if="message.version_navigation?.total > 1" class="version-switcher" aria-label="回答版本切换">
      <button
        class="version-button"
        type="button"
        aria-label="查看上一版本"
        :disabled="!message.version_navigation.can_previous"
        @click="selectVersion(-1)"
      >
        <el-icon><ArrowLeft /></el-icon>
      </button>
      <span class="version-label">{{ message.version_navigation.current }} / {{ message.version_navigation.total }}</span>
      <button
        class="version-button"
        type="button"
        aria-label="查看下一版本"
        :disabled="!message.version_navigation.can_next"
        @click="selectVersion(1)"
      >
        <el-icon><ArrowRight /></el-icon>
      </button>
    </div>

    <el-tooltip content="复制响应" placement="bottom">
      <button class="action-button" type="button" aria-label="复制响应" @click="copyResponse">
        <el-icon><CopyDocument /></el-icon>
      </button>
    </el-tooltip>
    <el-tooltip content="创建共享链接" placement="bottom">
      <button class="action-button" type="button" aria-label="创建共享链接" @click="createShareLink">
        <el-icon><Share /></el-icon>
      </button>
    </el-tooltip>
    <el-tooltip content="喜欢这个回答" placement="bottom">
      <button
        class="action-button"
        :class="{ active: feedback === 'love' }"
        type="button"
        aria-label="喜欢这个回答"
        :aria-pressed="feedback === 'love'"
        @click="toggleLoveFeedback"
      >
        <el-icon><Star /></el-icon>
      </button>
    </el-tooltip>
    <el-tooltip content="需要改进" placement="bottom">
      <button
        class="action-button"
        :class="{ active: feedback === 'needs_improvement' }"
        type="button"
        aria-label="需要改进"
        :aria-pressed="feedback === 'needs_improvement'"
        @click="openImprovementDialog"
      >
        <el-icon><Warning /></el-icon>
      </button>
    </el-tooltip>
    <el-tooltip content="重新生成" placement="bottom">
      <button class="action-button" type="button" aria-label="重新生成" @click="emit('regenerate', message)">
        <el-icon><RefreshRight /></el-icon>
      </button>
    </el-tooltip>

    <el-dropdown trigger="click" @command="handleMoreCommand">
      <button class="action-button" type="button" aria-label="更多操作">
        <el-icon><MoreFilled /></el-icon>
      </button>
      <template #dropdown>
        <el-dropdown-menu>
          <el-dropdown-item command="issue">
            <el-icon><ChatDotSquare /></el-icon>报告问题
          </el-dropdown-item>
          <el-dropdown-item command="pdf" :disabled="pdfExporting">
            <el-icon><Download /></el-icon>{{ pdfExporting ? "正在导出" : "导出到 PDF" }}
          </el-dropdown-item>
        </el-dropdown-menu>
      </template>
    </el-dropdown>
  </div>

  <el-dialog
    v-model="improvementDialogVisible"
    title="这条回答哪里需要改进？"
    width="min(460px, calc(100vw - 32px))"
    :close-on-click-modal="false"
    append-to-body
  >
    <el-radio-group v-model="improvementCategory" class="feedback-reasons">
      <el-radio-button
        v-for="option in improvementOptions"
        :key="option.value"
        :value="option.value"
      >
        {{ option.label }}
      </el-radio-button>
    </el-radio-group>
    <el-input
      v-model="improvementDescription"
      type="textarea"
      :rows="3"
      maxlength="300"
      show-word-limit
      placeholder="可补充具体问题，例如遗漏了哪项事实或哪部分证据不匹配"
      class="feedback-description"
    />
    <template #footer>
      <div class="feedback-dialog-actions">
        <el-button @click="improvementDialogVisible = false">取消</el-button>
        <el-button :loading="feedbackSubmitting" @click="submitImprovement(false)">
          仅提交反馈
        </el-button>
        <el-button type="primary" :loading="feedbackSubmitting" @click="submitImprovement(true)">
          按此原因重新生成
        </el-button>
      </div>
    </template>
  </el-dialog>

  <Teleport to="body">
    <div v-if="pdfExporting" class="pdf-download-status" role="status" aria-live="polite">
      <el-icon class="pdf-download-spinner"><Loading /></el-icon>
      <span>正在生成并下载 PDF</span>
    </div>
  </Teleport>
</template>

<script setup>
import { ref, watch } from "vue";
import { ElMessage, ElMessageBox } from "element-plus";
import {
  ArrowLeft,
  ArrowRight,
  ChatDotSquare,
  CopyDocument,
  Download,
  Loading,
  MoreFilled,
  RefreshRight,
  Share,
  Star,
  Warning,
} from "@element-plus/icons-vue";
import {
  API_ORIGIN,
  exportMessagePdf,
  getFeedbackClientId,
  reportMessageIssue,
  saveMessageFeedback,
  shareMessage,
} from "../api";

const props = defineProps({
  sessionId: { type: String, required: true },
  message: { type: Object, required: true },
});

const emit = defineEmits(["regenerate", "select-version"]);
const feedback = ref(props.message.feedback || "none");
const improvementDialogVisible = ref(false);
const improvementCategory = ref("off_topic");
const improvementDescription = ref("");
const feedbackSubmitting = ref(false);
const pdfExporting = ref(false);

const improvementOptions = [
  { value: "off_topic", label: "答非所问" },
  { value: "fact_error", label: "事实错误" },
  { value: "missing_information", label: "遗漏关键信息" },
  { value: "evidence_mismatch", label: "证据不匹配" },
  { value: "layout", label: "排版问题" },
  { value: "other", label: "其他" },
];

// 切换回答版本或重新载入会话时，同步恢复该消息已保存的反馈状态。
watch(
  () => [props.message.message_id, props.message.feedback, props.message.feedback_reason],
  () => {
    feedback.value = props.message.feedback || "none";
    improvementDescription.value = props.message.feedback_reason || "";
  },
);

const writeClipboard = async (text) => {
  if (navigator.clipboard?.writeText) {
    await navigator.clipboard.writeText(text);
    return;
  }
  // 兼容未开放 Clipboard API 的浏览器环境。
  const textarea = document.createElement("textarea");
  textarea.value = text;
  textarea.style.position = "fixed";
  textarea.style.opacity = "0";
  document.body.appendChild(textarea);
  textarea.select();
  document.execCommand("copy");
  textarea.remove();
};

const copyResponse = async () => {
  await writeClipboard(String(props.message.content || ""));
  ElMessage.success("回答已复制");
};

const selectVersion = (offset) => {
  const navigation = props.message.version_navigation || {};
  const nextIndex = Number(navigation.current || 1) - 1 + offset;
  emit("select-version", { groupId: props.message.version_group_id, index: nextIndex });
};

const createShareLink = async () => {
  try {
    const result = await shareMessage(props.sessionId, props.message.message_id);
    await writeClipboard(result.share_url);
    ElMessage.success("共享链接已复制");
  } catch (error) {
    ElMessage.error(`创建共享链接失败：${error.response?.data?.detail || error.message}`);
  }
};

const persistFeedback = async (rating, reason = "") => {
  const result = await saveMessageFeedback(props.sessionId, props.message.message_id, {
    rating,
    client_id: getFeedbackClientId(),
    reason,
  });
  feedback.value = result.rating || rating;
  props.message.feedback = feedback.value;
  props.message.feedback_reason = reason;
  return result;
};

const toggleLoveFeedback = async () => {
  const nextRating = feedback.value === "love" ? "none" : "love";
  try {
    await persistFeedback(nextRating);
  } catch (error) {
    ElMessage.error(`保存反馈失败：${error.response?.data?.detail || error.message}`);
  }
};

const openImprovementDialog = async () => {
  if (feedback.value === "needs_improvement") {
    try {
      await persistFeedback("none");
    } catch (error) {
      ElMessage.error(`取消反馈失败：${error.response?.data?.detail || error.message}`);
    }
    return;
  }
  improvementDialogVisible.value = true;
};

const improvementReasonText = () => {
  const label = improvementOptions.find((item) => item.value === improvementCategory.value)?.label || "其他";
  const description = improvementDescription.value.trim();
  return description ? `${label}：${description}` : label;
};

const submitImprovement = async (regenerate) => {
  if (feedbackSubmitting.value) return;
  feedbackSubmitting.value = true;
  const reason = improvementReasonText();
  try {
    const result = await persistFeedback("needs_improvement", reason);
    improvementDialogVisible.value = false;
    // 仅提交会异步生成当前会话的临时回答要求，提示用户它不会阻塞本轮页面。
    const savedMessage = result?.guidance_status === "pending"
      ? "反馈已保存，正在应用到本会话"
      : "反馈已保存";
    ElMessage.success(regenerate ? "反馈已保存，正在按原因重新生成" : savedMessage);
    if (regenerate) {
      emit("regenerate", { message: props.message, instruction: reason });
    }
  } catch (error) {
    ElMessage.error(`保存反馈失败：${error.response?.data?.detail || error.message}`);
  } finally {
    feedbackSubmitting.value = false;
  }
};

const submitIssue = async () => {
  try {
    const { value } = await ElMessageBox.prompt("请简要说明这条回答存在的问题", "报告问题", {
      confirmButtonText: "提交",
      cancelButtonText: "取消",
      inputPlaceholder: "例如：事实不准确、遗漏关键信息或排版有问题",
      inputValidator: (text) => Boolean(String(text || "").trim()) || "请输入问题说明",
    });
    await reportMessageIssue(props.sessionId, props.message.message_id, {
      category: "user_report",
      description: String(value || "").trim(),
      client_id: getFeedbackClientId(),
    });
    ElMessage.success("问题已提交");
  } catch (error) {
    // 用户主动取消弹窗时不显示错误提示。
    if (error !== "cancel" && error !== "close") {
      ElMessage.error(`问题提交失败：${error.response?.data?.detail || error.message}`);
    }
  }
};

const exportPdf = async () => {
  if (pdfExporting.value) return;
  pdfExporting.value = true;
  try {
    const report = await exportMessagePdf(props.sessionId, props.message.message_id);
    // 消息级报告只下载当前可见版本，不挂载缩略图卡片，也不新增聊天消息。
    const anchor = document.createElement("a");
    anchor.href = `${API_ORIGIN}${report.url.startsWith("/") ? report.url : `/${report.url}`}`;
    anchor.download = report.filename || "气象灾害分析报告.pdf";
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    ElMessage.success("PDF 已生成并开始下载");
  } catch (error) {
    ElMessage.error(`PDF 导出失败：${error.response?.data?.detail || error.message}`);
  } finally {
    pdfExporting.value = false;
  }
};

const handleMoreCommand = (command) => {
  if (command === "issue") submitIssue();
  if (command === "pdf") exportPdf();
};
</script>

<style scoped>
.message-actions {
  align-items: center;
  display: flex;
  gap: 2px;
  margin-top: 12px;
  min-height: 30px;
}

.action-button {
  align-items: center;
  background: transparent;
  border: 0;
  border-radius: 5px;
  color: #777771;
  cursor: pointer;
  display: inline-flex;
  height: 30px;
  justify-content: center;
  padding: 0;
  width: 30px;
}

.action-button:hover {
  background: #ecece8;
  color: #292927;
}

.action-button.active {
  background: #292927;
  color: #ffffff;
  box-shadow: inset 0 0 0 1px #292927;
}

.action-button.active:hover {
  background: #171715;
  color: #ffffff;
}

.version-switcher {
  align-items: center;
  display: inline-flex;
  gap: 2px;
  margin-right: 6px;
}

.version-button {
  align-items: center;
  background: transparent;
  border: 0;
  border-radius: 4px;
  color: #555552;
  cursor: pointer;
  display: inline-flex;
  height: 26px;
  justify-content: center;
  padding: 0;
  width: 26px;
}

.version-button:hover:not(:disabled) {
  background: #ecece8;
}

.version-button:disabled {
  color: #c5c5c0;
  cursor: default;
}

.version-label {
  color: #777771;
  font-size: 11px;
  min-width: 34px;
  text-align: center;
}

.feedback-reasons {
  display: grid;
  gap: 8px;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  width: 100%;
}

.feedback-reasons :deep(.el-radio-button),
.feedback-reasons :deep(.el-radio-button__inner) {
  width: 100%;
}

.feedback-description {
  margin-top: 16px;
}

.feedback-dialog-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  justify-content: flex-end;
}

/* 消息级导出只在右下角显示轻量进度，不占用回答区域，也不创建报告缩略图。 */
:global(.pdf-download-status) {
  align-items: center;
  background: #292927;
  border: 1px solid #454542;
  border-radius: 7px;
  bottom: 24px;
  box-shadow: 0 8px 26px rgba(25, 25, 23, 0.2);
  color: #ffffff;
  display: flex;
  font-size: 13px;
  gap: 9px;
  padding: 10px 13px;
  position: fixed;
  right: 24px;
  z-index: 4000;
}

:global(.pdf-download-spinner) {
  animation: pdf-download-spin 0.9s linear infinite;
}

@keyframes pdf-download-spin {
  to { transform: rotate(360deg); }
}

@media (max-width: 520px) {
  .feedback-dialog-actions {
    align-items: stretch;
    flex-direction: column-reverse;
  }

  .feedback-dialog-actions .el-button {
    margin-left: 0;
    width: 100%;
  }
}
</style>
