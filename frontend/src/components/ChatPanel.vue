<template>
  <section class="workspace">
    <header class="topbar">
      <div>
        <h2>气象灾害个例问答助手</h2>
      </div>
      <div class="status-strip">
        <el-tag :type="health?.vector_keys ? 'success' : 'info'" effect="plain">
          key {{ health?.vector_keys ?? 0 }}
        </el-tag>
        <el-tag :type="health?.vector_dimension === 1024 ? 'success' : 'warning'" effect="plain">
          维度 {{ health?.vector_dimension ?? "未知" }}
        </el-tag>
        <el-tag :type="health?.embedding_available ? 'success' : 'warning'" effect="plain">
          {{ health?.embedding_available ? "向量模型已接入" : "向量模型未接入" }}
        </el-tag>
        <el-tag :type="health?.llm_available ? 'success' : 'warning'" effect="plain">
          {{ health?.llm_available ? "大模型已接入" : "大模型未接入" }}
        </el-tag>
      </div>
    </header>

    <main class="content-grid">
      <section class="chat-column">
        <div class="question-row">
          <el-select
            v-model="selectedQuestion"
            placeholder="选择验收问题"
            clearable
            filterable
            @change="usePresetQuestion"
          >
            <el-option
              v-for="item in evalQuestions"
              :key="`${item.pdf_filename}-${item.question}`"
              :label="item.question"
              :value="item.question"
            />
          </el-select>
        </div>

        <div ref="messageList" class="messages">
          <div class="message assistant">
            <div class="bubble">我是你的气象灾害问答助手，请问你有什么想咨询的吗</div>
          </div>

          <div
            v-for="message in messages"
            :key="message.local_id || message.message_id"
            class="message"
            :class="message.role"
          >
            <div class="bubble" :class="{ answer: message.role === 'assistant' }">
              <span v-if="message.content">{{ message.content }}</span>
              <span v-else class="typing">正在检索并生成回答...</span>
              <div v-if="message.role === 'assistant' && message.images?.length" class="answer-images">
                <a
                  v-for="image in message.images"
                  :key="image.image_id"
                  class="evidence-image"
                  :href="imageUrl(image.url)"
                  target="_blank"
                  rel="noreferrer"
                >
                  <img
                    :src="imageUrl(image.url)"
                    :alt="image.caption || image.image_id"
                    @error="hideBrokenImage"
                  />
                  <span>{{ image.caption || `第${image.page_no}页图片` }}</span>
                </a>
              </div>
            </div>
          </div>
        </div>

        <div class="composer">
          <el-input
            v-model="question"
            type="textarea"
            :rows="4"
            resize="none"
            placeholder="输入问题，例如：请分析2025年3月14-15日暴雪天气过程"
            @keydown.ctrl.enter.prevent="submit"
          />
          <el-button type="primary" :icon="Promotion" :loading="loading" @click="submit">发送</el-button>
        </div>
      </section>

      <aside class="evidence-column">
        <section class="plain-panel standard-search-panel">
          <h3>个例智能检索</h3>
          <el-tabs v-model="caseToolTab" class="case-tool-tabs">
            <el-tab-pane label="多维检索" name="search">
              <div class="standard-search-form">
                <el-input
                  v-model="standardSearch.q"
                  clearable
                  placeholder="2025年5月山西北部雷暴大风，有雷达图的个例"
                  @keydown.enter.prevent="submitStandardCaseSearch"
                />
                <div class="filter-grid">
                  <el-input v-model="standardSearch.date" clearable placeholder="日期" />
                  <el-input v-model="standardSearch.disaster_type" clearable placeholder="灾种" />
                  <el-input v-model="standardSearch.area" clearable placeholder="地区" />
                  <el-select v-model="standardSearch.image_type" clearable placeholder="图片类型">
                    <el-option label="雷达图" value="radar" />
                    <el-option label="卫星图" value="satellite" />
                    <el-option label="降水图" value="precipitation" />
                    <el-option label="大风图" value="wind" />
                    <el-option label="探空图" value="sounding" />
                    <el-option label="形势图" value="synoptic" />
                    <el-option label="温度图" value="temperature" />
                    <el-option label="预警图" value="warning" />
                  </el-select>
                  <el-input v-model="standardSearch.data_category" clearable placeholder="数据类别" />
                  <el-input v-model="standardSearch.source_pdf" clearable placeholder="来源 PDF" />
                </div>
                <el-button
                  type="primary"
                  :icon="Search"
                  :loading="standardCaseLoading"
                  @click="submitStandardCaseSearch"
                >
                  检索
                </el-button>
              </div>
              <div class="panel-scroll tool-results">
                <template v-if="standardCaseResults">
                  <p class="result-count">匹配 {{ standardCaseResults.count || 0 }} 个标准个例</p>
                  <div v-if="standardCaseResults.cases?.length" class="standard-case-list">
                    <article
                      v-for="item in standardCaseResults.cases"
                      :key="item.case_id"
                      class="standard-case-card"
                    >
                      <p class="standard-case-title">{{ item.title || item.case_id }}</p>
                      <div class="case-meta">
                        <span>{{ item.date_range || "时段未知" }}</span>
                        <span>{{ (item.disaster_types || []).join("、") || "灾种未知" }}</span>
                        <span>{{ (item.affected_areas || []).join("、") || "地区未知" }}</span>
                        <span>{{ item.source_pdf }}</span>
                      </div>
                      <p v-if="item.summary" class="case-summary">{{ item.summary }}</p>
                      <div v-if="item.evidence_images?.length" class="image-strip">
                        <a
                          v-for="image in item.evidence_images"
                          :key="image.image_id"
                          class="evidence-image"
                          :href="imageUrl(image.url)"
                          target="_blank"
                          rel="noreferrer"
                        >
                          <img
                            :src="imageUrl(image.url)"
                            :alt="image.caption || image.image_id"
                            @error="hideBrokenImage"
                          />
                          <span>{{ image.caption || image.data_category || `第${image.page_no}页图片` }}</span>
                        </a>
                      </div>
                    </article>
                  </div>
                  <el-empty v-else description="暂无匹配个例" />
                </template>
                <el-empty v-else description="输入条件后检索标准个例" />
              </div>
            </el-tab-pane>

            <el-tab-pane label="相似匹配" name="similar">
              <div class="standard-search-form">
                <el-input
                  v-model="similarSearch.q"
                  clearable
                  placeholder="输入新过程，例如：2025年7月山西北部强对流，有雷达回波"
                  @keydown.enter.prevent="submitSimilarCaseSearch"
                />
                <div class="filter-grid">
                  <el-input v-model="similarSearch.date" clearable placeholder="日期" />
                  <el-input v-model="similarSearch.disaster_type" clearable placeholder="灾种" />
                  <el-input v-model="similarSearch.area" clearable placeholder="地区" />
                  <el-select v-model="similarSearch.image_type" clearable placeholder="图片类型">
                    <el-option label="雷达图" value="radar" />
                    <el-option label="卫星图" value="satellite" />
                    <el-option label="降水图" value="precipitation" />
                    <el-option label="大风图" value="wind" />
                    <el-option label="探空图" value="sounding" />
                    <el-option label="形势图" value="synoptic" />
                    <el-option label="温度图" value="temperature" />
                    <el-option label="预警图" value="warning" />
                  </el-select>
                  <el-input v-model="similarSearch.data_category" clearable placeholder="数据类别" />
                  <el-input v-model="similarSearch.source_pdf" clearable placeholder="来源 PDF" />
                </div>
                <el-button
                  type="primary"
                  :icon="Search"
                  :loading="similarCaseLoading"
                  @click="submitSimilarCaseSearch"
                >
                  匹配
                </el-button>
              </div>
              <div class="panel-scroll tool-results">
                <template v-if="similarCaseResults">
                  <p class="result-count">返回 {{ similarCaseResults.count || 0 }} 个相似个例</p>
                  <div v-if="similarCaseResults.matches?.length" class="standard-case-list">
                    <article
                      v-for="item in similarCaseResults.matches"
                      :key="item.case_id"
                      class="standard-case-card"
                    >
                      <div class="similar-title-row">
                        <p class="standard-case-title">{{ item.title || item.case_id }}</p>
                        <span class="similar-score">{{ formatScore(item.similarity_score) }}</span>
                      </div>
                      <div class="case-meta">
                        <span>{{ item.date_range || "时段未知" }}</span>
                        <span>{{ (item.disaster_types || []).join("、") || "灾种未知" }}</span>
                        <span>{{ (item.affected_areas || []).join("、") || "地区未知" }}</span>
                        <span>{{ item.source_pdf }}</span>
                      </div>
                      <div class="score-breakdown">
                        <span>灾种 {{ formatScore(item.score_breakdown?.disaster) }}</span>
                        <span>时空 {{ formatScore((item.score_breakdown?.temporal || 0) + (item.score_breakdown?.spatial || 0)) }}</span>
                        <span>图像 {{ formatScore(item.score_breakdown?.image) }}</span>
                        <span>文本 {{ formatScore(item.score_breakdown?.text) }}</span>
                      </div>
                      <ul v-if="item.match_reasons?.length" class="reason-list">
                        <li v-for="reason in item.match_reasons" :key="reason">{{ reason }}</li>
                      </ul>
                      <ul v-if="item.forecast_tips?.length" class="tip-list">
                        <li v-for="tip in item.forecast_tips" :key="tip">{{ tip }}</li>
                      </ul>
                      <div v-if="item.evidence_images?.length" class="image-strip">
                        <a
                          v-for="image in item.evidence_images"
                          :key="image.image_id"
                          class="evidence-image"
                          :href="imageUrl(image.url)"
                          target="_blank"
                          rel="noreferrer"
                        >
                          <img
                            :src="imageUrl(image.url)"
                            :alt="image.caption || image.image_id"
                            @error="hideBrokenImage"
                          />
                          <span>{{ image.caption || image.data_category || `第${image.page_no}页图片` }}</span>
                        </a>
                      </div>
                    </article>
                  </div>
                  <el-empty v-else description="暂无相似个例" />
                </template>
                <el-empty v-else description="输入新过程后匹配历史个例" />
              </div>
            </el-tab-pane>
          </el-tabs>
        </section>

        <section class="plain-panel">
          <h3>证据片段</h3>
          <div class="panel-scroll">
            <template v-if="activeCase">
              <p class="case-title">{{ activeCase.chunk_id }}</p>
              <dl>
                <dt>key</dt>
                <dd>{{ activeCase.chunk_id }}</dd>
                <dt>来源</dt>
                <dd>{{ activeCase.source_pdf }}</dd>
                <dt>序号</dt>
                <dd>{{ activeCase.chunk_no }}</dd>
              </dl>
              <p class="case-content">{{ activeCase.content }}</p>
            </template>
            <el-empty v-else description="请选择左侧 key" />
          </div>
        </section>

        <section class="plain-panel">
          <h3>检索证据</h3>
          <div class="panel-scroll">
            <template v-if="queryResult">
              <dl>
                <dt>检索模式</dt>
                <dd>{{ retrievalModeText }}</dd>
                <dt>大模型状态</dt>
                <dd>{{ llmStatusText }}</dd>
                <dt>命中数量</dt>
                <dd>{{ queryResult.hit_count }}</dd>
              </dl>
              <div class="hit-list">
                <article
                  v-for="hit in queryResult.hits"
                  :key="hit.chunk.chunk_id"
                  class="hit"
                >
                  <button class="hit-main" @click="$emit('select-hit', hit.chunk.chunk_id)">
                    <span class="hit-key">{{ hit.chunk.chunk_id }}</span>
                    <span class="hit-title">{{ hit.chunk.source_pdf }}</span>
                    <span class="hit-score">{{ formatScore(hit.score) }}</span>
                  </button>
                  <div v-if="hit.images?.length" class="image-strip">
                    <a
                      v-for="image in hit.images"
                      :key="image.image_id"
                      class="evidence-image"
                      :href="imageUrl(image.url)"
                      target="_blank"
                      rel="noreferrer"
                      @click.stop
                    >
                      <img
                        :src="imageUrl(image.url)"
                        :alt="image.caption || image.image_id"
                        @error="hideBrokenImage"
                      />
                      <span>{{ image.caption || `第${image.page_no}页图片` }}</span>
                    </a>
                  </div>
                </article>
              </div>
            </template>
            <el-empty v-else description="暂无检索结果" />
          </div>
        </section>
      </aside>
    </main>
  </section>
</template>

<script setup>
import { computed, nextTick, ref, watch } from "vue";
import { Promotion, Search } from "@element-plus/icons-vue";

const props = defineProps({
  activeCase: {
    type: Object,
    default: null,
  },
  queryResult: {
    type: Object,
    default: null,
  },
  evalQuestions: {
    type: Array,
    default: () => [],
  },
  messages: {
    type: Array,
    default: () => [],
  },
  loading: {
    type: Boolean,
    default: false,
  },
  health: {
    type: Object,
    default: null,
  },
  standardCaseResults: {
    type: Object,
    default: null,
  },
  standardCaseLoading: {
    type: Boolean,
    default: false,
  },
  similarCaseResults: {
    type: Object,
    default: null,
  },
  similarCaseLoading: {
    type: Boolean,
    default: false,
  },
});

const emit = defineEmits(["query", "select-hit", "standard-case-search", "similar-case-search"]);
const question = ref("");
const selectedQuestion = ref("");
const messageList = ref(null);
const caseToolTab = ref("search");
const standardSearch = ref({
  q: "",
  date: "",
  disaster_type: "",
  area: "",
  image_type: "",
  data_category: "",
  source_pdf: "",
});
const similarSearch = ref({
  q: "",
  date: "",
  disaster_type: "",
  area: "",
  image_type: "",
  data_category: "",
  source_pdf: "",
});

const retrievalModeText = computed(() => {
  if (!props.queryResult) return "";
  if (props.queryResult.retrieval_mode === "document_vector") return "证据片段向量检索";
  if (props.queryResult.retrieval_mode?.startsWith("document_vector_error_fallback")) {
    if (props.queryResult.retrieval_mode.includes("dimension mismatch")) {
      return "文档向量维度不匹配，请重建 document_index";
    }
    return "文档向量异常后回退";
  }
  if (props.queryResult.retrieval_mode === "vector") return "向量检索";
  if (props.queryResult.retrieval_mode?.startsWith("vector_error_fallback")) {
    if (props.queryResult.retrieval_mode.includes("dimension mismatch")) {
      return "向量维度不匹配，请重新入库";
    }
    return "向量异常后回退";
  }
  return props.queryResult.retrieval_mode;
});

const llmStatusText = computed(() => {
  if (!props.queryResult) return "";
  if (props.queryResult.llm_status === "streaming") return "正在生成";
  if (props.queryResult.llm_used) return "已调用大模型";
  if (props.queryResult.llm_status === "skipped_no_api_key") return "未配置或未启用 API Key";
  if (props.queryResult.llm_status?.startsWith("failed")) return props.queryResult.llm_status;
  return props.queryResult.llm_status;
});

const usePresetQuestion = (value) => {
  question.value = value || "";
};

const scrollToBottom = async () => {
  await nextTick();
  if (messageList.value) {
    messageList.value.scrollTop = messageList.value.scrollHeight;
  }
};

const submit = () => {
  const text = question.value.trim();
  if (!text || props.loading) return;
  question.value = "";
  emit("query", text);
};

const submitStandardCaseSearch = () => {
  const params = {};
  for (const [key, value] of Object.entries(standardSearch.value)) {
    const trimmed = `${value || ""}`.trim();
    if (trimmed) params[key] = trimmed;
  }
  emit("standard-case-search", params);
};

const submitSimilarCaseSearch = () => {
  const params = {};
  for (const [key, value] of Object.entries(similarSearch.value)) {
    const trimmed = `${value || ""}`.trim();
    if (trimmed) params[key] = trimmed;
  }
  params.top_n = 5;
  emit("similar-case-search", params);
};

const formatScore = (score) => Number(score || 0).toFixed(3);
const imageUrl = (url) => {
  if (!url) return "";
  if (url.startsWith("http")) return url;
  return `http://127.0.0.1:8000${url}`;
};
const hideBrokenImage = (event) => {
  event.currentTarget.closest(".evidence-image")?.remove();
};

watch(
  () => props.messages,
  () => {
    scrollToBottom();
  },
  { deep: true }
);
</script>

<style scoped>
.workspace {
  display: flex;
  flex-direction: column;
  height: 100vh;
  min-height: 0;
  overflow: hidden;
}

.topbar {
  align-items: center;
  background: #ffffff;
  border-bottom: 1px solid #dce1e7;
  display: flex;
  flex: 0 0 auto;
  justify-content: space-between;
  padding: 18px 24px;
}

h2,
h3,
p {
  margin: 0;
}

h2 {
  font-size: 22px;
  font-weight: 650;
}

.status-strip {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  justify-content: flex-end;
}

.content-grid {
  display: grid;
  flex: 1;
  gap: 1px;
  grid-template-columns: minmax(0, 1fr) 390px;
  min-height: 0;
  overflow: hidden;
}

.chat-column,
.evidence-column {
  background: #f7f9fb;
  min-height: 0;
  min-width: 0;
  overflow: hidden;
}

.chat-column {
  display: grid;
  grid-template-rows: auto 1fr auto;
  padding: 22px;
}

.question-row {
  margin-bottom: 16px;
}

.question-row :deep(.el-select) {
  width: 100%;
}

.messages {
  display: flex;
  flex-direction: column;
  gap: 12px;
  min-height: 0;
  overflow: auto;
  padding-bottom: 18px;
}

.message {
  display: flex;
}

.message.user {
  justify-content: flex-end;
}

.bubble {
  background: #ffffff;
  border: 1px solid #dce1e7;
  border-radius: 8px;
  line-height: 1.7;
  max-width: 78%;
  padding: 12px 14px;
  white-space: pre-wrap;
}

.message.user .bubble {
  background: #184f90;
  border-color: #184f90;
  color: #ffffff;
}

.answer {
  color: #17202a;
}

.typing {
  color: #667789;
}

.composer {
  align-items: flex-end;
  background: #ffffff;
  border: 1px solid #dce1e7;
  border-radius: 8px;
  display: grid;
  gap: 12px;
  grid-template-columns: 1fr auto;
  padding: 12px;
}

.composer :deep(.el-textarea__inner) {
  box-shadow: none;
}

.evidence-column {
  border-left: 1px solid #dce1e7;
  display: grid;
  gap: 1px;
  grid-template-rows: minmax(260px, 0.95fr) minmax(220px, 0.8fr) minmax(260px, 1fr);
}

.plain-panel {
  background: #ffffff;
  display: flex;
  flex-direction: column;
  min-height: 0;
  overflow: hidden;
  padding: 20px;
}

.plain-panel h3 {
  flex: 0 0 auto;
  font-size: 16px;
  font-weight: 650;
  margin-bottom: 14px;
}

.panel-scroll {
  flex: 1;
  min-height: 0;
  overflow: auto;
  padding-right: 4px;
}

.case-title {
  font-weight: 650;
  line-height: 1.5;
  margin-bottom: 12px;
}

dl {
  display: grid;
  gap: 8px 12px;
  grid-template-columns: 76px minmax(0, 1fr);
  margin: 0 0 14px;
}

dt {
  color: #667789;
}

dd {
  margin: 0;
  overflow-wrap: anywhere;
}

.case-content {
  color: #334155;
  line-height: 1.7;
  white-space: pre-wrap;
}

.standard-search-panel {
  padding-bottom: 14px;
}

.case-tool-tabs {
  display: flex;
  flex: 1;
  flex-direction: column;
  min-height: 0;
}

.case-tool-tabs :deep(.el-tabs__content) {
  flex: 1;
  min-height: 0;
  overflow: hidden;
}

.case-tool-tabs :deep(.el-tab-pane) {
  display: flex;
  flex-direction: column;
  height: 100%;
  min-height: 0;
}

.standard-search-form {
  display: grid;
  flex: 0 0 auto;
  gap: 8px;
  margin-bottom: 10px;
}

.standard-search-form :deep(.el-input__wrapper),
.standard-search-form :deep(.el-select__wrapper) {
  min-height: 32px;
}

.filter-grid {
  display: grid;
  gap: 8px;
  grid-template-columns: repeat(2, minmax(0, 1fr));
}

.filter-grid :deep(.el-select) {
  width: 100%;
}

.result-count {
  color: #667789;
  font-size: 12px;
  margin-bottom: 8px;
}

.tool-results {
  border-top: 1px solid #eef2f6;
  padding-top: 8px;
}

.standard-case-list {
  display: grid;
  gap: 8px;
}

.standard-case-card {
  background: #f7f9fb;
  border: 1px solid #dce1e7;
  border-radius: 6px;
  display: grid;
  gap: 8px;
  padding: 10px;
}

.standard-case-title {
  color: #17202a;
  font-weight: 650;
  line-height: 1.45;
}

.case-meta {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}

.case-meta span {
  background: #ffffff;
  border: 1px solid #dce1e7;
  border-radius: 4px;
  color: #526274;
  font-size: 11px;
  line-height: 1.4;
  max-width: 100%;
  overflow-wrap: anywhere;
  padding: 2px 6px;
}

.case-summary {
  color: #334155;
  display: -webkit-box;
  font-size: 13px;
  line-height: 1.55;
  overflow: hidden;
  -webkit-box-orient: vertical;
  -webkit-line-clamp: 3;
}

.similar-title-row {
  align-items: flex-start;
  display: flex;
  gap: 8px;
  justify-content: space-between;
}

.similar-score {
  background: #184f90;
  border-radius: 4px;
  color: #ffffff;
  flex: 0 0 auto;
  font-family: Consolas, "Courier New", monospace;
  font-size: 12px;
  line-height: 1;
  padding: 5px 6px;
}

.score-breakdown {
  display: grid;
  gap: 6px;
  grid-template-columns: repeat(2, minmax(0, 1fr));
}

.score-breakdown span {
  background: #ffffff;
  border: 1px solid #dce1e7;
  border-radius: 4px;
  color: #526274;
  font-size: 11px;
  padding: 4px 6px;
}

.reason-list,
.tip-list {
  display: grid;
  gap: 4px;
  margin: 0;
  padding-left: 16px;
}

.reason-list li,
.tip-list li {
  color: #334155;
  font-size: 12px;
  line-height: 1.5;
}

.tip-list li {
  color: #184f90;
}

.hit-list {
  display: grid;
  gap: 8px;
}

.hit {
  background: #f7f9fb;
  border: 1px solid #dce1e7;
  border-radius: 6px;
  display: grid;
  gap: 4px;
  padding: 10px;
}

.hit-main {
  background: transparent;
  border: 0;
  cursor: pointer;
  display: grid;
  gap: 4px;
  padding: 0;
  text-align: left;
  width: 100%;
}

.hit:hover {
  border-color: #8bb6e8;
}

.hit-key {
  color: #184f90;
  font-family: Consolas, "Courier New", monospace;
  font-size: 13px;
  overflow-wrap: anywhere;
}

.hit-title {
  color: #17202a;
  line-height: 1.45;
}

.hit-score {
  color: #667789;
  font-size: 12px;
}

.image-strip {
  display: grid;
  gap: 8px;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  margin-top: 6px;
}

.answer-images {
  border-top: 1px solid #e5e9ef;
  display: grid;
  gap: 8px;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  margin-top: 12px;
  padding-top: 12px;
}

.evidence-image {
  background: #ffffff;
  border: 1px solid #dce1e7;
  border-radius: 6px;
  color: #334155;
  display: grid;
  gap: 4px;
  overflow: hidden;
  text-decoration: none;
}

.evidence-image img {
  aspect-ratio: 4 / 3;
  background: #eef2f6;
  object-fit: cover;
  width: 100%;
}

.evidence-image span {
  display: -webkit-box;
  font-size: 11px;
  line-height: 1.35;
  overflow: hidden;
  padding: 0 6px 6px;
  -webkit-box-orient: vertical;
  -webkit-line-clamp: 2;
}

@media (max-width: 1100px) {
  .content-grid {
    grid-template-columns: 1fr;
  }

  .evidence-column {
    border-left: 0;
  }
}
</style>
