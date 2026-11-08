<template>
  <div v-if="!messages.length" class="message assistant">
    <div class="bubble answer">
      <div class="markdown-body"><p>我是你的气象灾害问答助手，请问你有什么想咨询的吗？</p></div>
    </div>
  </div>

  <div
    v-for="message in messages"
    :key="message.local_id || message.message_id"
    class="message"
    :class="message.role"
  >
    <div class="bubble" :class="{ answer: message.role === 'assistant' }">
      <ProcessingStatus v-if="isProcessingMessage(message)" :progress="message.progress" />
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
        v-if="
          message.role === 'assistant' &&
          message.agent_type === 'smart_case_match' &&
          message.agent_result?.matched_cases
        "
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
                <span
                  class="bar-fill"
                  :style="{ width: `${barPercent(item.value, visual.items)}%` }"
                />
              </div>
              <span class="bar-value">{{ item.value }}</span>
            </div>
          </div>
          <div v-else-if="visual.type === 'table'" class="visual-table-wrap">
            <table class="visual-table">
              <thead>
                <tr>
                  <th v-for="column in visual.columns || []" :key="column.key">
                    {{ column.label }}
                  </th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="(row, rowIndex) in visual.rows || []" :key="rowIndex">
                  <td v-for="column in visual.columns || []" :key="column.key">
                    {{ row[column.key] }}
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </section>
      </div>

      <div
        v-if="message.role === 'assistant' && visibleReports(message).length"
        class="answer-reports"
      >
        <PdfReportCard
          v-for="report in visibleReports(message)"
          :key="report.report_id || report.url"
          :report="report"
        />
      </div>

      <AssistantMessageActions
        v-if="showMessageActions(message, sessionId)"
        :session-id="sessionId"
        :message="message"
        @regenerate="emit('regenerate', $event)"
        @select-version="emit('select-version', $event)"
      />
    </div>
  </div>
</template>

<script setup>
import PdfReportCard from "../PdfReportCard.vue";
import SmartCaseResult from "../SmartCaseResult.vue";
import AssistantMessageActions from "../AssistantMessageActions.vue";
import ProcessingStatus from "../ProcessingStatus.vue";
import "../../styles/answer-content.css";
import {
  barPercent,
  isProcessingMessage,
  renderMarkdown,
  resolveMarkdownImage,
  showMessageActions,
  showTextContent,
  visibleReports,
} from "./chatRendering";

defineProps({
  messages: {
    type: Array,
    default: () => [],
  },
  sessionId: {
    type: String,
    default: "",
  },
});

const emit = defineEmits(["regenerate", "select-version", "preview-image"]);

const openMarkdownImage = (event, message) => {
  const image = resolveMarkdownImage(event, message);
  if (image) emit("preview-image", image);
};
</script>

<style scoped>
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

.answer-reports {
  display: grid;
  gap: 10px;
}

@media (max-width: 760px) {
  .bubble {
    font-size: 14px;
    max-width: 90%;
  }

  .message.assistant .bubble {
    max-width: 100%;
  }
}
</style>
