<template>
  <el-container class="app-shell">
    <el-aside :width="caseSidebarCollapsed ? '56px' : '340px'" class="sidebar">
      <button
        v-if="caseSidebarCollapsed"
        class="collapsed-case-button"
        title="展开向量节点"
        @click="caseSidebarCollapsed = false"
      >
        <el-icon><Expand /></el-icon>
        <span>节点</span>
      </button>
      <CaseList
        v-else
        :case-ids="caseIds"
        :active-case-id="activeCaseId"
        :loading="caseLoading"
        :case-count="caseCount"
        @select="handleSelect"
        @refresh-library="handleRefreshLibrary"
        @upload-material="handleUploadMaterial"
        @toggle-collapse="caseSidebarCollapsed = true"
      />
    </el-aside>

    <SessionList
      class="session-column"
      :sessions="sessions"
      :active-session-id="activeSessionId"
      :loading="sessionLoading"
      @new-session="handleCreateSession"
      @select-session="handleSelectSession"
    />

    <el-main class="main">
      <ChatPanel
        :active-case="activeCase"
        :query-result="queryResult"
        :eval-questions="evalQuestions"
        :messages="chatMessages"
        :loading="queryLoading"
        :health="health"
        :standard-case-results="standardCaseResults"
        :standard-case-loading="standardCaseLoading"
        :similar-case-results="similarCaseResults"
        :similar-case-loading="similarCaseLoading"
        @query="handleQuery"
        @select-hit="handleSelect"
        @standard-case-search="handleStandardCaseSearch"
        @similar-case-search="handleSimilarCaseSearch"
      />
    </el-main>
  </el-container>
</template>

<script setup>
import { onMounted, ref } from "vue";
import { ElMessage } from "element-plus";
import { Expand } from "@element-plus/icons-vue";
import CaseList from "./components/CaseList.vue";
import ChatPanel from "./components/ChatPanel.vue";
import SessionList from "./components/SessionList.vue";
import {
  createSession,
  fetchDocumentChunk,
  fetchDocumentKeys,
  fetchEvalQuestions,
  fetchHealth,
  fetchSessionMessages,
  fetchSessions,
  fetchSimilarCases,
  fetchStandardCases,
  streamQuery,
  updateSessionTitle,
  uploadMaterialsBatch,
} from "./api";

const caseIds = ref([]);
const caseCount = ref(0);
const activeCaseId = ref("");
const activeCase = ref(null);
const queryResult = ref(null);
const evalQuestions = ref([]);
const queryLoading = ref(false);
const caseLoading = ref(false);
const health = ref(null);
const sessions = ref([]);
const activeSessionId = ref("");
const chatMessages = ref([]);
const sessionLoading = ref(false);
const caseSidebarCollapsed = ref(false);
const standardCaseResults = ref(null);
const standardCaseLoading = ref(false);
const similarCaseResults = ref(null);
const similarCaseLoading = ref(false);

const loadHealth = async () => {
  health.value = await fetchHealth();
};

const loadCases = async () => {
  caseLoading.value = true;
  try {
    const response = await fetchDocumentKeys();
    caseIds.value = response.chunk_ids || [];
    caseCount.value = response.count ?? caseIds.value.length;
    if (!activeCaseId.value && caseIds.value.length > 0) {
      await handleSelect(caseIds.value[0]);
    }
  } finally {
    caseLoading.value = false;
  }
};

const loadEvalQuestions = async () => {
  const response = await fetchEvalQuestions();
  evalQuestions.value = response.questions || [];
};

const handleSelect = async (caseId) => {
  activeCaseId.value = caseId;
  activeCase.value = await fetchDocumentChunk(caseId);
};

const reloadLibraryState = async () => {
  await Promise.all([loadCases(), loadHealth()]);
};

const imagesFromHits = (hits = []) => {
  const seen = new Set();
  const images = [];
  for (const hit of hits) {
    for (const image of hit.images || []) {
      if (!image?.image_id || seen.has(image.image_id)) continue;
      seen.add(image.image_id);
      images.push(image);
      if (images.length >= 6) return images;
    }
  }
  return images;
};

const loadSessions = async () => {
  const response = await fetchSessions();
  sessions.value = response.sessions || [];
};

const loadSessionMessages = async (sessionId) => {
  const response = await fetchSessionMessages(sessionId);
  chatMessages.value = (response.messages || []).map((message) => ({
    ...message,
    local_id: `saved-${message.message_id}`,
  }));
};

const ensureActiveSession = async () => {
  await loadSessions();
  if (sessions.value.length > 0) {
    await handleSelectSession(sessions.value[0].session_id);
    return;
  }
  activeSessionId.value = "";
  chatMessages.value = [];
};

const makeSessionTitle = (text) => {
  const value = text.trim();
  if (!value) return "新会话";

  const dateWeather = value.match(
    /((?:\d{1,2}月)?\d{1,2}\s*[-~～至]\s*\d{1,2}\s*日|\d{1,2}\s*日)(.{0,8}?(?:雨雪|暴雪|沙尘|寒潮|大风|霜冻|降水|降雨|降雪|雷暴|冰雹|高温|低温|雾|霾))/
  );
  if (dateWeather) {
    return `${dateWeather[1]}${dateWeather[2]}`
      .replace(/\s+/g, "")
      .replace(/天气过程|过程|天气/g, "")
      .slice(0, 14);
  }

  const cleaned = value
    .replace(/请|帮我|一下|分析|说明|介绍|总结|这个|关于|天气过程|过程|天气|吗|？|\?/g, "")
    .replace(/[，,。.!！：:；;\s]+/g, "");
  return (cleaned || value).slice(0, 14);
};

const activeSession = () => sessions.value.find((session) => session.session_id === activeSessionId.value);

const renameNewSessionAfterFirstQuestion = async (questionText) => {
  const session = activeSession();
  if (!session || chatMessages.value.length > 0 || session.title !== "新会话") return;
  const title = makeSessionTitle(questionText);
  const updated = await updateSessionTitle(session.session_id, title);
  sessions.value = sessions.value.map((item) => (item.session_id === updated.session_id ? updated : item));
};

const handleCreateSession = async (title = "新会话") => {
  sessionLoading.value = true;
  try {
    const session = await createSession(title);
    await loadSessions();
    await handleSelectSession(session.session_id);
    return session;
  } finally {
    sessionLoading.value = false;
  }
};

const handleSelectSession = async (sessionId) => {
  if (!sessionId || sessionId === activeSessionId.value) return;
  sessionLoading.value = true;
  try {
    activeSessionId.value = sessionId;
    queryResult.value = null;
    await loadSessionMessages(sessionId);
  } finally {
    sessionLoading.value = false;
  }
};

const handleRefreshLibrary = async () => {
  caseLoading.value = true;
  try {
    await loadCases();
    ElMessage({
      message: "刷新成功：已重新获取向量节点 key",
      type: "success",
      duration: 2000,
    });
  } finally {
    caseLoading.value = false;
  }
};

const handleUploadMaterial = async (files) => {
  caseLoading.value = true;
  try {
    const response = await uploadMaterialsBatch(files);
    ElMessage.success(
      `批量上传成功：${response.uploaded_files?.length || 0} 个 PDF，新增入库 ${response.indexed} 个证据片段`
    );
    await reloadLibraryState();
  } finally {
    caseLoading.value = false;
  }
};

const handleStandardCaseSearch = async (params) => {
  standardCaseLoading.value = true;
  try {
    standardCaseResults.value = await fetchStandardCases(params || {});
  } catch (error) {
    ElMessage.error(`多维检索失败：${error.message}`);
  } finally {
    standardCaseLoading.value = false;
  }
};

const handleSimilarCaseSearch = async (params) => {
  similarCaseLoading.value = true;
  try {
    similarCaseResults.value = await fetchSimilarCases(params || {});
  } catch (error) {
    ElMessage.error(`相似匹配失败：${error.message}`);
  } finally {
    similarCaseLoading.value = false;
  }
};

const pushLocalMessage = (role, content, images = []) => {
  const message = {
    role,
    content,
    images,
    local_id: `local-${Date.now()}-${chatMessages.value.length}`,
  };
  chatMessages.value.push(message);
  return message;
};

const handleQuery = async (question) => {
  const text = question.trim();
  if (!text) {
    ElMessage.warning("请先输入问题");
    return;
  }
  if (!activeSessionId.value) {
    await handleCreateSession(makeSessionTitle(text));
  } else {
    await renameNewSessionAfterFirstQuestion(text);
  }

  pushLocalMessage("user", text);
  const assistantMessage = pushLocalMessage("assistant", "");
  queryLoading.value = true;

  try {
    await streamQuery(
      { question: text, top_k: 5, top_n: 3, session_id: activeSessionId.value },
      {
        onMetadata: (metadata) => {
          const hits = metadata.hits || [];
          queryResult.value = {
            question: metadata.question,
            answer: "",
            retrieval_mode: metadata.retrieval_mode,
            llm_used: false,
            llm_status: "streaming",
            hit_count: metadata.hit_count,
            hits,
          };
          assistantMessage.images = imagesFromHits(hits);
        },
        onDelta: (delta) => {
          assistantMessage.content += delta;
          if (queryResult.value) {
            queryResult.value.answer += delta;
          }
        },
        onDone: async (payload) => {
          if (queryResult.value) {
            queryResult.value.llm_used = Boolean(payload.llm_used);
            queryResult.value.llm_status = payload.llm_used ? "called" : "skipped_no_api_key";
          }
          await loadSessions();
        },
      }
    );
  } catch (error) {
    assistantMessage.content = `问答失败：${error.message}`;
    ElMessage.error(`问答失败：${error.message}`);
  } finally {
    queryLoading.value = false;
  }
};

onMounted(async () => {
  try {
    await Promise.all([loadHealth(), loadCases(), loadEvalQuestions(), ensureActiveSession()]);
  } catch (error) {
    ElMessage.error(`初始化失败：${error.message}`);
  }
});
</script>

<style scoped>
.app-shell {
  background: #f4f6f8;
  color: #17202a;
  height: 100vh;
  overflow: hidden;
}

.sidebar {
  background: #ffffff;
  border-right: 1px solid #dce1e7;
  height: 100vh;
  overflow: hidden;
  transition: width 0.18s ease;
}

.collapsed-case-button {
  align-items: center;
  background: #ffffff;
  border: 0;
  color: #184f90;
  cursor: pointer;
  display: flex;
  flex-direction: column;
  gap: 8px;
  height: 100%;
  justify-content: flex-start;
  padding-top: 22px;
  width: 100%;
}

.collapsed-case-button span {
  font-size: 13px;
  writing-mode: vertical-rl;
}

.session-column {
  flex: 0 0 260px;
  width: 260px;
}

.main {
  height: 100vh;
  min-height: 0;
  min-width: 0;
  overflow: hidden;
  padding: 0;
}
</style>
