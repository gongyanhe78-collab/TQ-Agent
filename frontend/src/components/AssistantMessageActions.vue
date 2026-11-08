<template>
  <AssistantActionsToolbar
    :message="message"
    :feedback="feedback"
    :pdf-exporting="pdfExporting"
    @copy="copyResponse"
    @share="createShareLink"
    @toggle-love="toggleLoveFeedback"
    @improve="openImprovementDialog"
    @select-version="selectVersion"
    @regenerate="emit('regenerate', $event)"
    @more-command="handleMoreCommand"
  />

  <AssistantFeedbackDialog
    v-model:visible="improvementDialogVisible"
    v-model:category="improvementCategory"
    v-model:description="improvementDescription"
    :submitting="feedbackSubmitting"
    @submit="submitImprovement"
  />

  <Teleport to="body">
    <div v-if="pdfExporting" class="pdf-download-status" role="status" aria-live="polite">
      <el-icon class="pdf-download-spinner"><Loading /></el-icon>
      <span>正在生成并下载 PDF</span>
    </div>
  </Teleport>
</template>

<script setup>
import { reactive, ref, watch } from "vue";
import { ElMessage, ElMessageBox } from "element-plus";
import { Loading } from "@element-plus/icons-vue";
import {
  downloadAuthenticatedFile,
  exportMessagePdf,
  getFeedbackClientId,
  reportMessageIssue,
  saveMessageFeedback,
  shareMessage,
} from "../api";
import AssistantActionsToolbar from "./chat/AssistantActionsToolbar.vue";
import AssistantFeedbackDialog from "./chat/AssistantFeedbackDialog.vue";

const props = defineProps({
  sessionId: { type: String, required: true },
  message: { type: Object, required: true },
});

const emit = defineEmits(["regenerate", "select-version"]);
// 通过本地响应式引用同步反馈状态，避免直接改写传入的 message 对象。
const localMessage = reactive(props.message);
const feedback = ref(props.message.feedback || "none");
const improvementDialogVisible = ref(false);
const improvementCategory = ref("off_topic");
const improvementDescription = ref("");
const feedbackSubmitting = ref(false);
const pdfExporting = ref(false);

// 切换回答版本或重新载入会话时，同步恢复该消息已保存的反馈状态。
watch(
  () => [props.message.message_id, props.message.feedback, props.message.feedback_reason],
  () => {
    feedback.value = props.message.feedback || "none";
    improvementDescription.value = props.message.feedback_reason || "";
  }
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
  localMessage.feedback = feedback.value;
  localMessage.feedback_reason = reason;
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
  const labels = {
    off_topic: "答非所问",
    fact_error: "事实错误",
    missing_information: "遗漏关键信息",
    evidence_mismatch: "证据不匹配",
    layout: "排版问题",
    other: "其他",
  };
  const label = labels[improvementCategory.value] || "其他";
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
    const savedMessage =
      result?.guidance_status === "pending" ? "反馈已保存，正在应用到本会话" : "反馈已保存";
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
    // 消息级报告通过带访问令牌的请求下载，避免裸 <a> 标签不带凭证触发 401。
    await downloadAuthenticatedFile(report.url, report.filename || "气象灾害分析报告.pdf");
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
  to {
    transform: rotate(360deg);
  }
}
</style>
