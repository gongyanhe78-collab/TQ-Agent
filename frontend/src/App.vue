<template>
  <div v-if="restoring" class="auth-loading" role="status" aria-live="polite">
    <span class="auth-loading-spinner" aria-hidden="true" />
    <span>正在恢复登录状态</span>
  </div>

  <LoginScreen
    v-else-if="!isAuthenticated"
    :error="authError"
    :loading="authSubmitting"
    :notice="expired ? '登录已过期，请重新登录' : ''"
    @submit="handleLogin"
  />

  <el-container
    v-else
    class="app-shell"
    :class="{ 'session-collapsed': sessionPanelCollapsed }"
  >
    <SessionList
      class="session-column"
      :class="{ collapsed: sessionPanelCollapsed }"
      :sessions="sessions"
      :active-session-id="activeSessionId"
      :loading="sessionLoading"
      :collapsed="sessionPanelCollapsed"
      @new-session="handleCreateSession"
      @select-session="handleSelectSession"
      @toggle-collapse="sessionPanelCollapsed = !sessionPanelCollapsed"
    />

    <el-main class="main">
      <ChatPanel
        :messages="displayMessages"
        :loading="queryLoading"
        :session-id="activeSessionId"
        @query="handleQuery"
        @regenerate="handleRegenerate"
        @select-version="handleSelectVersion"
      />
    </el-main>

    <div class="account-control">
      <UserMenu :user="user" @logout="handleLogout" />
    </div>
  </el-container>
</template>

<script setup>
import { computed, onMounted, reactive, ref } from "vue";
import { ElMessage } from "element-plus";
import ChatPanel from "./components/ChatPanel.vue";
import SessionList from "./components/SessionList.vue";
import LoginScreen from "./components/auth/LoginScreen.vue";
import UserMenu from "./components/auth/UserMenu.vue";
import { useAuthSession } from "./components/auth/useAuthSession";
import { buildDisplayedMessages, resolveVersionGroup } from "./messageVersions";
import {
  createSession,
  fetchSessionMessages,
  fetchSessions,
  getApiErrorMessage,
  streamQuery,
  updateSessionTitle,
} from "./api";

const sessions = ref([]);
const activeSessionId = ref("");
const chatMessages = ref([]);
const sessionLoading = ref(false);
const queryLoading = ref(false);
// 每个回答组只记录当前查看下标，数据库中的各版本仍完整保留。
const selectedVersionByGroup = reactive({});
// 侧栏状态由页面统一维护，收起后主聊天区才能同步扩展而不是留下空白列。
const sessionPanelCollapsed = ref(false);
let activeStreamController = null;

const abortActiveStream = () => {
  if (!activeStreamController) return;
  activeStreamController.abort();
  activeStreamController = null;
  queryLoading.value = false;
};

// 认证失效时立即停止流式更新；上一用户内容仅在登录页后方保留，下一次登录前统一清除。
const handleAuthenticationExpired = () => {
  abortActiveStream();
  ElMessage.warning("登录已过期，请重新登录");
};

const {
  user,
  restoring,
  submitting: authSubmitting,
  error: authError,
  expired,
  isAuthenticated,
  restore,
  login,
  logout,
} = useAuthSession({ onExpired: handleAuthenticationExpired });

const resetUserState = () => {
  abortActiveStream();
  sessions.value = [];
  activeSessionId.value = "";
  // 反馈状态保存在各自消息对象中，清空消息会同步丢弃旧用户的反馈和回答版本选择。
  chatMessages.value = [];
  sessionLoading.value = false;
  for (const key of Object.keys(selectedVersionByGroup)) {
    delete selectedVersionByGroup[key];
  }
};

// 会话列表和消息都从当前认证用户的主会话库恢复，页面不维护额外的匿名会话状态。
const loadSessions = async () => {
  const response = await fetchSessions();
  sessions.value = response.sessions || [];
};

const loadSessionMessages = async (sessionId) => {
  const response = await fetchSessionMessages(sessionId);
  chatMessages.value = (response.messages || []).map((message) => {
    // Agent 展示数据保存在消息 metadata 中，恢复历史时提升到组件直接使用的字段。
    const metadata = message.metadata || {};
    return {
      ...message,
      agent_type: metadata.agent_type || "",
      agent_result: metadata.agent_result || {},
      images: metadata.images || [],
      image_display_mode: metadata.image_display_mode || "hidden",
      visuals: metadata.visuals || [],
      reports: metadata.reports || [],
      progress: metadata.progress || {},
      status: metadata.status || "completed",
      regenerated_from_message_id: metadata.regenerated_from_message_id || null,
      regeneration_instruction: metadata.regeneration_instruction || "",
      source_user_message_id: metadata.source_user_message_id || null,
      version_group_id: metadata.version_group_id || "",
      version: metadata.version || 1,
      local_id: `saved-${message.message_id}`,
    };
  });
  // 重新进入会话时默认展示每组最新版本。
  for (const key of Object.keys(selectedVersionByGroup)) delete selectedVersionByGroup[key];
};

// 同一原问题的所有助手版本折叠为一个显示槽，切换时整条回答及其操作状态同步变化。
const displayMessages = computed(() => buildDisplayedMessages(chatMessages.value, selectedVersionByGroup));

const ensureActiveSession = async () => {
  await loadSessions();
  if (sessions.value.length > 0) {
    await handleSelectSession(sessions.value[0].session_id);
    return;
  }
  activeSessionId.value = "";
  chatMessages.value = [];
};

const enterWorkspace = async () => {
  resetUserState();
  try {
    await ensureActiveSession();
  } catch (error) {
    if (user.value) {
      ElMessage.error(`聊天记录加载失败：${getApiErrorMessage(error)}`);
    }
  }
};

const handleLogin = async (credentials) => {
  try {
    const account = await login(credentials);
    if (!account) return;
    await enterWorkspace();
  } catch {
    // 登录错误由 LoginScreen 根据认证组合函数中的错误信息展示。
  }
};

const handleLogout = async () => {
  try {
    await logout();
  } catch (error) {
    ElMessage.error(`退出登录失败：${getApiErrorMessage(error)}`);
  } finally {
    resetUserState();
  }
};

const makeSessionTitle = (text) => {
  const value = text.trim();
  if (!value) return "新会话";
  const cleaned = value
    .replace(/请|帮我|一下|分析|说明|介绍|总结|这个|关于|吗|？|\?/g, "")
    .replace(/[，,。.!！：:；;\s]+/g, "");
  return (cleaned || value).slice(0, 14);
};

const activeSession = () => sessions.value.find((session) => session.session_id === activeSessionId.value);

const renameNewSessionAfterFirstQuestion = async (questionText) => {
  const session = activeSession();
  if (!session || chatMessages.value.length > 0 || session.title !== "新会话") return;
  const updated = await updateSessionTitle(session.session_id, makeSessionTitle(questionText));
  sessions.value = sessions.value.map((item) => (item.session_id === updated.session_id ? updated : item));
};

const handleCreateSession = async (title = "新会话") => {
  sessionLoading.value = true;
  try {
    const session = await createSession(title);
    await loadSessions();
    activeSessionId.value = session.session_id;
    chatMessages.value = [];
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
    await loadSessionMessages(sessionId);
  } finally {
    sessionLoading.value = false;
  }
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

const pushLocalMessage = (role, content) => {
  const message = {
    role,
    content,
    images: [],
    image_display_mode: "hidden",
    visuals: [],
    reports: [],
    agent_type: "",
    agent_result: {},
    progress: {},
    status: role === "assistant" ? "streaming" : "completed",
    regenerated_from_message_id: null,
    regeneration_instruction: "",
    source_user_message_id: null,
    version_group_id: "",
    version: 1,
    local_id: `local-${Date.now()}-${chatMessages.value.length}`,
  };
  chatMessages.value.push(message);
  // 数组写入后 Vue 会把对象包装为 Proxy；必须返回该代理对象，
  // 后续 SSE 的 delta 与进度更新才能立即触发页面重新渲染。
  return chatMessages.value[chatMessages.value.length - 1];
};

const handleStreamFailure = (assistantMessage, error) => {
  const hasReceivedContent = Boolean(String(assistantMessage.content || "").trim());
  assistantMessage.status = "interrupted";

  if (error?.name === "AbortError" || error?.code === "AUTH_EXPIRED" || !user.value) {
    if (!hasReceivedContent) assistantMessage.content = "回答已中断";
    return;
  }

  if (!hasReceivedContent) {
    assistantMessage.content = `问答失败：${getApiErrorMessage(error)}`;
  }
  ElMessage.warning(hasReceivedContent ? "回答已中断，已保留收到的内容" : assistantMessage.content);
};

// 把普通提问和重新生成统一接入同一套 SSE 状态更新，避免两条链路行为不一致。
const streamAssistantAnswer = async (text, assistantMessage, extraPayload = {}) => {
  queryLoading.value = true;
  const controller = new AbortController();
  activeStreamController = controller;
  try {
    await streamQuery(
      {
        question: text,
        top_k: 5,
        top_n: 3,
        session_id: activeSessionId.value,
        ...extraPayload,
      },
      {
        onMetadata: (metadata) => {
          // 图片、图表和报告统一挂载到助手消息，确保历史聊天可以按消息重放。
          assistantMessage.agent_type = metadata.agent_type || assistantMessage.agent_type;
          assistantMessage.agent_result = metadata.agent_result || assistantMessage.agent_result;
          assistantMessage.progress = metadata.progress || assistantMessage.progress;
          assistantMessage.status = metadata.status || assistantMessage.status;
          const metadataImages = metadata.images || [];
          assistantMessage.images = metadataImages.length
            ? metadataImages
            : imagesFromHits(metadata.hits || []);
          assistantMessage.image_display_mode = metadata.image_display_mode || assistantMessage.image_display_mode;
          assistantMessage.visuals = metadata.visuals || [];
          assistantMessage.reports = metadata.reports || [];
          // 正文结束后的轻量 metadata 会先于 done 到达，用它立即启用消息操作。
          assistantMessage.message_id = metadata.message_id || assistantMessage.message_id;
          assistantMessage.regenerated_from_message_id = metadata.regenerated_from_message_id
            || assistantMessage.regenerated_from_message_id;
          assistantMessage.regeneration_instruction = metadata.regeneration_instruction
            || assistantMessage.regeneration_instruction;
          assistantMessage.source_user_message_id = metadata.source_user_message_id
            || assistantMessage.source_user_message_id;
          assistantMessage.version_group_id = metadata.version_group_id || assistantMessage.version_group_id;
          assistantMessage.version = metadata.version || assistantMessage.version;
        },
        onDelta: (delta) => {
          assistantMessage.content += delta;
        },
        onDone: async (payload) => {
          assistantMessage.reports = payload.reports || assistantMessage.reports;
          assistantMessage.message_id = payload.message_id || assistantMessage.message_id;
          assistantMessage.status = payload.status || "completed";
          assistantMessage.regenerated_from_message_id = payload.regenerated_from_message_id || null;
          assistantMessage.regeneration_instruction = payload.regeneration_instruction || "";
          assistantMessage.source_user_message_id = payload.source_user_message_id || null;
          assistantMessage.version_group_id = payload.version_group_id || assistantMessage.version_group_id;
          assistantMessage.version = payload.version || 1;
          await loadSessions();
        },
      },
      { signal: controller.signal },
    );
  } catch (error) {
    handleStreamFailure(assistantMessage, error);
  } finally {
    if (activeStreamController === controller) activeStreamController = null;
    queryLoading.value = false;
  }
};

// 所有自然语言问题共用一个 SSE 入口，由后端按意图路由到主 RAG、多维检索或 Smart。
const handleQuery = async (question) => {
  const text = question.trim();
  if (!text || queryLoading.value || !user.value) return;

  try {
    if (!activeSessionId.value) {
      await handleCreateSession(makeSessionTitle(text));
    } else {
      await renameNewSessionAfterFirstQuestion(text);
    }
  } catch (error) {
    ElMessage.error(`会话创建失败：${getApiErrorMessage(error)}`);
    return;
  }

  pushLocalMessage("user", text);
  const assistantMessage = pushLocalMessage("assistant", "");
  await streamAssistantAnswer(text, assistantMessage);
};

const handleRegenerate = async (request) => {
  const message = request?.message || request;
  const instruction = String(request?.instruction || "").trim();
  if (queryLoading.value || !message?.message_id || !activeSessionId.value) return;
  const assistantMessage = pushLocalMessage("assistant", "");
  assistantMessage.regenerated_from_message_id = message.message_id;
  assistantMessage.version_group_id = message.version_group_id || resolveVersionGroup(
    message,
    new Map(chatMessages.value.filter((item) => item.message_id).map((item) => [item.message_id, item])),
  );
  assistantMessage.version = Number(message.version_navigation?.total || message.version || 1) + 1;
  // 新版本生成时回到该组最新答案；用户之后仍可通过 1 / N 导航查看旧版本。
  delete selectedVersionByGroup[assistantMessage.version_group_id];
  // 后端会按消息 ID 找回原问题；占位问题只用于满足统一请求协议。
  await streamAssistantAnswer("重新生成原回答", assistantMessage, {
    regenerate_from_message_id: message.message_id,
    regeneration_instruction: instruction,
  });
};

const handleSelectVersion = ({ groupId, index }) => {
  if (!groupId || !Number.isInteger(index)) return;
  selectedVersionByGroup[groupId] = index;
};

onMounted(async () => {
  const restored = await restore();
  if (restored) await enterWorkspace();
});
</script>

<style scoped>
.auth-loading {
  align-items: center;
  background: #fbfbfa;
  color: #62625d;
  display: flex;
  font-size: 13px;
  gap: 10px;
  height: 100vh;
  justify-content: center;
}

.auth-loading-spinner {
  animation: auth-loading-spin 0.9s linear infinite;
  border: 2px solid #d9d9d4;
  border-radius: 50%;
  border-top-color: #3f3f3b;
  height: 18px;
  width: 18px;
}

@keyframes auth-loading-spin {
  to {
    transform: rotate(360deg);
  }
}

.app-shell {
  background: #fbfbfa;
  color: #292927;
  height: 100vh;
  overflow: hidden;
  position: relative;
}

.session-column {
  flex: 0 0 264px;
  width: 264px;
}

.session-column.collapsed {
  flex-basis: 56px;
  width: 56px;
}

.main {
  height: 100vh;
  min-height: 0;
  min-width: 0;
  overflow: hidden;
  padding: 0;
}

.account-control {
  position: absolute;
  right: 16px;
  top: 9px;
  z-index: 20;
}

@media (max-width: 760px) {
  .app-shell {
    display: grid;
    grid-template-rows: 220px minmax(0, 1fr);
  }

  .app-shell.session-collapsed {
    grid-template-rows: 56px minmax(0, 1fr);
  }

  .session-column {
    height: 220px;
    width: 100%;
  }

  .session-column.collapsed {
    flex-basis: 56px;
    height: 56px;
    width: 100%;
  }

  .main {
    height: 100%;
  }

  .account-control {
    right: 12px;
    top: 8px;
  }
}
</style>
