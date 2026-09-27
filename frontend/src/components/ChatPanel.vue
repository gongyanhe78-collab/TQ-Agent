<template>
  <section class="chat-page">
    <header class="topbar">
      <h1>气象灾害个例问答助手</h1>
    </header>

    <div ref="messageList" class="messages">
      <div v-if="!messages.length" class="message assistant">
        <div class="bubble answer">
          <div class="markdown-body"><p>我是你的气象灾害问答助手，请问你有什么想咨询的吗？</p></div>
        </div>
      </div>

      <div
        v-for="(message, index) in messages"
        :key="message.local_id || message.message_id"
        class="message"
        :class="message.role"
      >
        <div class="bubble" :class="{ answer: message.role === 'assistant' }">
          <ProcessingStatus
            v-if="isProcessingMessage(message)"
            :progress="message.progress"
          />
          <template v-if="showTextContent(message)">
            <div
              v-if="message.role === 'assistant'"
              class="markdown-body"
              v-html="renderMarkdown(message.content)"
              @click="openMarkdownImage($event, message)"
            />
            <span v-else class="user-text">{{ message.content }}</span>
          </template>

          <SmartCaseResult
            v-if="message.role === 'assistant' && message.agent_type === 'smart_case_match' && message.agent_result?.matched_cases"
            :result="message.agent_result"
          />

          <div v-if="message.role === 'assistant' && message.visuals?.length" class="answer-visuals">
            <section
              v-for="visual in message.visuals"
              :key="visual.title || JSON.stringify(visual)"
              class="answer-visual"
            >
              <p class="visual-title">{{ visual.title }}</p>
              <div v-if="visual.type === 'bar'" class="bar-chart">
                <div v-for="item in visual.items || []" :key="item.label" class="bar-row">
                  <span class="bar-label">{{ item.label }}</span>
                  <div class="bar-track">
                    <span class="bar-fill" :style="{ width: `${barPercent(item.value, visual.items)}%` }" />
                  </div>
                  <span class="bar-value">{{ item.value }}</span>
                </div>
              </div>
              <div v-else-if="visual.type === 'table'" class="visual-table-wrap">
                <table class="visual-table">
                  <thead>
                    <tr>
                      <th v-for="column in visual.columns || []" :key="column.key">{{ column.label }}</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr v-for="(row, rowIndex) in visual.rows || []" :key="rowIndex">
                      <td v-for="column in visual.columns || []" :key="column.key">{{ row[column.key] }}</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </section>
          </div>

          <div v-if="message.role === 'assistant' && visibleReports(message).length" class="answer-reports">
            <PdfReportCard
              v-for="report in visibleReports(message)"
              :key="report.report_id || report.url"
              :report="report"
            />
          </div>

          <AssistantMessageActions
            v-if="showMessageActions(message)"
            :session-id="sessionId"
            :message="message"
            @regenerate="emit('regenerate', $event)"
            @select-version="emit('select-version', $event)"
          />
        </div>
      </div>
    </div>

    <form class="composer" @submit.prevent="submit">
      <el-input
        v-model="question"
        type="textarea"
        :rows="3"
        resize="none"
        placeholder="输入问题"
        @keydown.ctrl.enter.prevent="submit"
      />
      <el-button type="primary" native-type="submit" :icon="Promotion" :loading="loading">发送</el-button>
    </form>

    <el-dialog
      v-model="imagePreviewOpen"
      class="image-preview-dialog"
      title="图片预览"
      width="min(960px, 92vw)"
      top="5vh"
      append-to-body
    >
      <div v-if="imagePreview" class="image-preview-content">
        <img
          :key="imagePreview.image_id"
          :src="assetUrl(imagePreview.url)"
          :alt="imagePreview.caption || imagePreview.image_id"
          @error="hideBrokenImage"
        />
        <p v-if="imagePreview.caption">{{ imagePreview.caption }}</p>
      </div>
    </el-dialog>
  </section>
</template>

<script setup>
import { nextTick, ref, watch } from "vue";
import { Promotion } from "@element-plus/icons-vue";
import DOMPurify from "dompurify";
import { marked } from "marked";
import PdfReportCard from "./PdfReportCard.vue";
import SmartCaseResult from "./SmartCaseResult.vue";
import AssistantMessageActions from "./AssistantMessageActions.vue";
import ProcessingStatus from "./ProcessingStatus.vue";
import { API_ORIGIN } from "../api";
import "../styles/answer-content.css";

const props = defineProps({
  messages: {
    type: Array,
    default: () => [],
  },
  loading: {
    type: Boolean,
    default: false,
  },
  sessionId: {
    type: String,
    default: "",
  },
});

const emit = defineEmits(["query", "regenerate", "select-version"]);
const question = ref("");
const messageList = ref(null);
const imagePreviewOpen = ref(false);
const imagePreview = ref(null);

// 回答采用 GFM Markdown 渲染，清洗 HTML 后再写入页面，兼顾表格展示与内容安全。
marked.setOptions({
  breaks: true,
  gfm: true,
});

const escapeHtml = (value) => String(value)
  .replaceAll("&", "&amp;")
  .replaceAll("<", "&lt;")
  .replaceAll(">", "&gt;")
  .replaceAll('"', "&quot;")
  .replaceAll("'", "&#39;");

// 图片 ID 按固定命名格式识别，不依赖当前回答附带的图片列表是否包含该资源。
const imageIdPattern = /^.+-page-\d{3}-(?:snapshot|image-\d{3})$/;

// 只将图片 ID 格式的行内代码变为点击入口，并按要求简化“对应图片ID”提示。
const renderMarkdown = (content) => {
  const markdown = String(content || "")
    .replace(/(\*\*|__)?对应图片ID[:：]?(\*\*|__)?[:：]?/g, (_match, open = "", close = "") => `${open}对应图片${close}`)
    .replace(/`([^`\r\n]+)`/g, (source, rawImageId) => {
      const imageId = rawImageId.trim();
      if (!imageIdPattern.test(imageId)) return source;
      const safeImageId = escapeHtml(imageId);
      return `<a href="#" class="markdown-image-id" data-image-id="${safeImageId}" role="button" aria-label="查看图片 ${safeImageId}"><code>${safeImageId}</code></a>`;
    });
  return DOMPurify.sanitize(marked.parse(markdown), {
    ADD_ATTR: ["data-image-id", "role", "aria-label"],
  });
};

// 通过事件委托捕获 v-html 中的图片 ID；未随回答返回的图片改由 ID 接口直接加载。
const openMarkdownImage = (event, message) => {
  const link = event.target?.closest?.(".markdown-image-id[data-image-id]");
  if (!link) return;
  event.preventDefault();
  const imageId = link.getAttribute("data-image-id");
  const image = (message.images || []).find((item) => String(item.image_id || "") === imageId);
  imagePreview.value = {
    ...(image || {}),
    image_id: imageId,
    url: image?.url || `/api/image-evidence/${encodeURIComponent(imageId)}`,
  };
  imagePreviewOpen.value = true;
};

// 输入框只负责提交自然语言，具体使用哪条检索链路由后端决定。
const submit = () => {
  const text = question.value.trim();
  if (!text || props.loading) return;
  emit("query", text);
  question.value = "";
};

const assetUrl = (url) => {
  if (!url || /^https?:\/\//i.test(url)) return url || "";
  return `${API_ORIGIN}${url.startsWith("/") ? url : `/${url}`}`;
};

// 图片资源失效时隐藏预览元素，证据按钮仍保留以便识别对应资源。
const hideBrokenImage = (event) => {
  event.currentTarget.style.display = "none";
};

// 柱状图以当前结果中的最大值为基准计算相对宽度。
const barPercent = (value, items = []) => {
  const maximum = Math.max(...items.map((item) => Number(item.value) || 0), 1);
  return Math.max(2, Math.round(((Number(value) || 0) / maximum) * 100));
};

// Smart 的完整文字答案仍会保存到会话，但页面只展示结构化结果，避免综合研判重复出现。
const showTextContent = (message) => Boolean(
  message?.content && message?.agent_type !== "smart_case_match"
);

// 后端 metadata 阶段使用 running，前端占位消息使用 streaming，两者都属于处理中状态。
const isProcessingMessage = (message) => Boolean(
  message?.role === "assistant"
  && (message.status === "streaming" || message.status === "running")
);

// 消息级 PDF 已经直接下载，历史 metadata 中即使保留记录也不应重新显示报告卡。
const visibleReports = (message) => (message.reports || []).filter((report) => report.report_mode !== "message_export");

// 只有已经落库且不再流式生成的回答才允许执行消息级操作。
const showMessageActions = (message) => Boolean(
  props.sessionId
  && message?.role === "assistant"
  && message?.message_id
  && message?.status === "completed"
);

// 新消息和流式增量到达时保持视图停留在最新内容。
watch(
  () => props.messages
    .map((message) => `${message.local_id || message.message_id}:${message.content}:${message.progress?.message || ""}`)
    .join("|"),
  async () => {
    await nextTick();
    if (messageList.value) {
      messageList.value.scrollTop = messageList.value.scrollHeight;
    }
  }
);
</script>

<style scoped>
.chat-page {
  background: #fbfbfa;
  display: grid;
  grid-template-rows: 52px minmax(0, 1fr) auto;
  height: 100%;
  min-height: 0;
}

.topbar {
  align-items: center;
  background: rgba(251, 251, 250, 0.94);
  border-bottom: 1px solid #ededeb;
  display: flex;
  padding: 0 max(24px, calc((100% - 800px) / 2));
}

h1 {
  color: #555552;
  font-size: 14px;
  font-weight: 600;
  letter-spacing: 0;
  margin: 0;
}

.messages {
  min-height: 0;
  overflow-y: auto;
  padding: 30px max(24px, calc((100% - 800px) / 2)) 44px;
  scrollbar-color: #d8d8d4 transparent;
  scrollbar-width: thin;
}

.message {
  display: flex;
  margin-bottom: 28px;
  width: 100%;
}

.message.user {
  justify-content: flex-end;
}

.bubble {
  color: #262624;
  font-size: 15px;
  line-height: 1.72;
  max-width: min(680px, 82%);
  overflow-wrap: anywhere;
}

.message.user .bubble {
  background: #eeeeec;
  border: 1px solid #e8e8e5;
  border-radius: 16px;
  color: #242422;
  padding: 9px 14px;
}

.message.assistant .bubble {
  max-width: 100%;
  width: 100%;
}

.user-text {
  white-space: pre-wrap;
}

.answer-visuals,
.answer-reports {
  border-top: 1px solid #e4e4e0;
  margin-top: 18px;
  padding-top: 16px;
}

.answer-visuals {
  display: grid;
  gap: 12px;
}

.answer-visual {
  background: #ffffff;
  border: 1px solid #deded9;
  border-radius: 7px;
  display: grid;
  gap: 10px;
  overflow: hidden;
  padding: 10px;
}

.visual-title {
  font-size: 13px;
  font-weight: 650;
  margin: 0;
}

.bar-chart {
  display: grid;
  gap: 8px;
}

.bar-row {
  align-items: center;
  display: grid;
  gap: 8px;
  grid-template-columns: 70px minmax(0, 1fr) 42px;
}

.bar-label,
.bar-value {
  color: #656560;
  font-size: 12px;
}

.bar-value {
  text-align: right;
}

.bar-track {
  background: #e9e9e5;
  border-radius: 4px;
  height: 10px;
  overflow: hidden;
}

.bar-fill {
  background: #3c806b;
  display: block;
  height: 100%;
}

.visual-table-wrap {
  overflow: auto;
}

.visual-table {
  border-collapse: separate;
  border-spacing: 0;
  border: 1px solid #d8d8d3;
  border-radius: 7px;
  overflow: hidden;
  min-width: 100%;
  white-space: normal;
}

.visual-table th,
.visual-table td {
  border-bottom: 1px solid #e2e2de;
  color: #343431;
  font-size: 13px;
  line-height: 1.5;
  padding: 7px 8px;
  text-align: left;
  vertical-align: top;
}

.visual-table th {
  background: #343432;
  color: #ffffff;
  font-weight: 650;
}

.visual-table tbody tr:nth-child(even) {
  background: #f3f3f0;
}

/* 可点击图片 ID 保留原有代码样式，并用蓝色下划线提示可点击。 */
.markdown-body :deep(.markdown-image-id) {
  color: inherit;
  cursor: pointer;
  text-decoration: underline;
  text-decoration-color: #1677ff;
  text-underline-offset: 3px;
}

.markdown-body :deep(.markdown-image-id:focus-visible) {
  outline: 2px solid #3c806b;
  outline-offset: 2px;
}

/* 弹窗中的图片完整显示并按视口高度缩放，避免宽幅天气图被裁切。 */
:global(.image-preview-dialog .image-preview-content) {
  display: grid;
  gap: 12px;
  justify-items: center;
}

:global(.image-preview-dialog .image-preview-content img) {
  max-height: 76vh;
  max-width: 100%;
  object-fit: contain;
}

:global(.image-preview-dialog .image-preview-content p) {
  color: #656560;
  margin: 0;
  text-align: center;
}

.answer-reports {
  display: grid;
  gap: 10px;
}

.composer {
  align-items: flex-end;
  background: linear-gradient(to top, #fbfbfa 82%, rgba(251, 251, 250, 0));
  display: grid;
  gap: 10px;
  grid-template-columns: minmax(0, 1fr) auto;
  padding: 20px max(24px, calc((100% - 800px) / 2)) 22px;
}

.composer :deep(.el-textarea__inner) {
  background: #ffffff;
  border-radius: 16px;
  box-shadow: 0 0 0 1px #deded9 inset, 0 5px 18px rgba(35, 35, 32, 0.06);
  color: #292927;
  font-family: inherit;
  min-height: 78px !important;
  padding: 13px 15px;
}

.composer :deep(.el-textarea) {
  min-width: 0;
}

.composer :deep(.el-textarea__inner:focus) {
  box-shadow: 0 0 0 1px #999991 inset, 0 5px 18px rgba(35, 35, 32, 0.06);
}

.composer :deep(.el-button--primary) {
  background: #2d2d2b;
  border-color: #2d2d2b;
  border-radius: 14px;
  height: 44px;
  min-width: 44px;
}

.composer :deep(.el-button--primary:hover) {
  background: #454542;
  border-color: #454542;
}

@media (max-width: 760px) {
  .chat-page {
    grid-template-rows: 48px minmax(0, 1fr) auto;
  }

  .topbar,
  .messages,
  .composer {
    padding-left: 16px;
    padding-right: 16px;
  }

  .bubble {
    font-size: 14px;
    max-width: 90%;
  }

  .message.assistant .bubble {
    max-width: 100%;
  }

  .messages {
    padding-bottom: 28px;
  }

  .composer {
    grid-template-columns: minmax(0, 1fr) 44px;
    padding-bottom: 14px;
  }

  /* 手机端只保留发送图标，确保按钮始终位于输入框右侧且不会溢出屏幕。 */
  .composer :deep(.el-button--primary) {
    font-size: 0;
    min-width: 44px;
    padding: 0;
    width: 44px;
  }

  .composer :deep(.el-button--primary .el-icon) {
    font-size: 16px;
    margin-right: 0;
  }

}
</style>
