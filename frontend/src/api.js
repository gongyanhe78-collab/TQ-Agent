import axios from "axios";

const normalizeOrigin = (value) => String(value || "").trim().replace(/\/+$/, "");
const browserApiOrigin = typeof window !== "undefined"
  ? normalizeOrigin(window.localStorage.getItem("risk-api-origin"))
  : "";
const configuredApiOrigin = normalizeOrigin(import.meta.env.VITE_API_ORIGIN);

// 生产环境固定使用同源相对地址；开发环境可通过 Vite 环境变量切到联调后端。
export const API_ORIGIN = import.meta.env.DEV
  ? configuredApiOrigin || browserApiOrigin || "http://127.0.0.1:8000"
  : "";
const API_BASE_URL = `${API_ORIGIN}/api`;

const clientOptions = {
  baseURL: API_BASE_URL,
  timeout: 60000,
  withCredentials: true,
};

const api = axios.create(clientOptions);
const authClient = axios.create(clientOptions);

// 访问令牌只保存在模块内存，页面刷新后依赖 HttpOnly 刷新 Cookie 恢复登录。
let accessToken = "";
let accessTokenExpiresAt = 0;
let refreshInFlight = null;
let authExpiredHandler = null;
let authExpiredNotified = false;

const storeAuthSession = (data) => {
  accessToken = String(data?.access_token || "");
  const expiresIn = Math.max(0, Number(data?.expires_in) || 0);
  accessTokenExpiresAt = accessToken && expiresIn
    ? Date.now() + expiresIn * 1000
    : 0;
  authExpiredNotified = false;
  return data;
};

export const clearAccessToken = () => {
  accessToken = "";
  accessTokenExpiresAt = 0;
  authExpiredNotified = false;
};

const notifyAuthExpired = () => {
  if (authExpiredNotified) return;
  authExpiredNotified = true;
  authExpiredHandler?.();
};

export const setAuthExpiredHandler = (handler) => {
  authExpiredHandler = typeof handler === "function" ? handler : null;
  return () => {
    if (authExpiredHandler === handler) authExpiredHandler = null;
  };
};

const refreshAccessToken = async () => {
  if (refreshInFlight) return refreshInFlight;
  refreshInFlight = authClient
    .post("/auth/refresh")
    .then(({ data }) => storeAuthSession(data))
    .finally(() => {
      refreshInFlight = null;
    });
  return refreshInFlight;
};

api.interceptors.request.use((config) => {
  config.headers = config.headers || {};
  if (accessToken) config.headers.Authorization = `Bearer ${accessToken}`;
  return config;
});

api.interceptors.response.use(
  (response) => response,
  async (error) => {
    const request = error.config || {};
    if (error.response?.status !== 401) return Promise.reject(error);

    if (request._authRetried) {
      clearAccessToken();
      notifyAuthExpired();
      return Promise.reject(error);
    }

    request._authRetried = true;
    try {
      await refreshAccessToken();
      return api(request);
    } catch (refreshError) {
      clearAccessToken();
      notifyAuthExpired();
      return Promise.reject(refreshError.response?.status === 401 ? error : refreshError);
    }
  },
);

const responseDetail = async (response) => {
  try {
    const payload = await response.json();
    return String(payload?.detail || payload?.message || "").trim();
  } catch {
    return "";
  }
};

const createResponseError = async (response) => {
  const detail = await responseDetail(response);
  const error = new Error(detail || `请求失败：${response.status}`);
  error.status = response.status;
  error.detail = detail;
  return error;
};

export const loginSession = async (username, password) => {
  const { data } = await authClient.post("/auth/login", { username, password });
  return storeAuthSession(data);
};

export const refreshSession = async () => refreshAccessToken();

export const logoutSession = async () => {
  try {
    const { data } = await authClient.post("/auth/logout");
    return data;
  } finally {
    clearAccessToken();
  }
};

// 统一把后端错误转为可展示文案，避免各页面自行猜测 401、403 和 404 的语义。
export const getApiErrorMessage = (error, fallback = "请求失败，请稍后重试") => {
  const status = Number(error?.status || error?.response?.status || 0);
  const detail = String(error?.detail || error?.response?.data?.detail || "").trim();
  if (status === 401) return detail || "登录已过期，请重新登录";
  if (status === 403) return detail || "没有权限执行此操作";
  if (status === 404) return detail || "资源不存在或无权访问";
  if (status >= 500) return detail || "服务暂时不可用，请稍后重试";
  return detail || String(error?.message || fallback);
};

export const fetchSessions = async () => (await api.get("/sessions")).data;
export const createSession = async (title) => (await api.post("/sessions", { title })).data;
export const updateSessionTitle = async (sessionId, title) =>
  (await api.patch(`/sessions/${sessionId}`, { title })).data;
export const getFeedbackClientId = () => {
  const key = "risk-feedback-client-id";
  let value = window.localStorage.getItem(key);
  if (!value) {
    value = `web-${crypto.randomUUID?.() || `${Date.now()}-${Math.random()}`}`;
    window.localStorage.setItem(key, value);
  }
  return value;
};
export const fetchSessionMessages = async (sessionId) =>
  (await api.get(`/sessions/${sessionId}/messages`, { params: { client_id: getFeedbackClientId() } })).data;
export const fetchSessionMessage = async (sessionId, messageId) =>
  (await api.get(`/sessions/${sessionId}/messages/${messageId}`)).data;
export const deleteSession = async (sessionId) => (await api.delete(`/sessions/${sessionId}`)).data;

export const exportMessagePdf = async (sessionId, messageId) =>
  (await api.post(`/sessions/${sessionId}/messages/${messageId}/export-pdf`)).data;
export const saveMessageFeedback = async (sessionId, messageId, payload) =>
  (await api.post(`/sessions/${sessionId}/messages/${messageId}/feedback`, payload)).data;
export const reportMessageIssue = async (sessionId, messageId, payload) =>
  (await api.post(`/sessions/${sessionId}/messages/${messageId}/report`, payload)).data;
export const shareMessage = async (sessionId, messageId) =>
  (await api.post(`/sessions/${sessionId}/messages/${messageId}/share`)).data;
export const rebuildKnowledgeBase = async () => (await api.post("/documents/index")).data;

// 资源地址可能来自持久化元数据，统一转换为认证客户端可复用的相对路径。
const toApiRequestPath = (value) => {
  const url = String(value || "").trim();
  if (/^https?:\/\//i.test(url)) return url;
  if (url.startsWith("/api/")) return url.slice(4);
  return url.startsWith("/") ? url : `/${url}`;
};

export const fetchAuthenticatedBlob = async (url) => {
  const response = await api.get(toApiRequestPath(url), { responseType: "blob" });
  return response.data;
};

export const fetchSmartCaseChunk = async (chunkId) =>
  (await api.get(`/smart-case-match/assets/chunks/${encodeURIComponent(chunkId)}`)).data;

export const downloadAuthenticatedFile = async (url, filename) => {
  const blob = await fetchAuthenticatedBlob(url);
  const objectUrl = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = objectUrl;
  link.download = filename || "download";
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(objectUrl);
};

const parseSseEvent = async (event, handlers) => {
  const dataLine = event.split("\n").find((line) => line.startsWith("data: "));
  if (!dataLine) return;
  const data = JSON.parse(dataLine.slice(6));
  if (data.type === "metadata") handlers.onMetadata?.(data);
  if (data.type === "delta") {
    handlers.onDelta?.(data.text || "");
    // 主动让出一个浏览器渲染帧，避免连续 SSE 小块都在同一轮微任务中合并显示。
    await new Promise((resolve) => setTimeout(resolve, 0));
  }
  if (data.type === "done") await handlers.onDone?.(data);
  if (data.type === "error") {
    const error = new Error(data.message || "回答生成失败");
    error.code = "STREAM_ERROR";
    throw error;
  }
};

// 流式请求自行处理 401 刷新，并携带 Cookie 与短期访问令牌，避免把 token 放进 URL。
export const streamQuery = async (payload, handlers = {}, options = {}) => {
  const controller = new AbortController();
  const externalSignal = options.signal;
  let tokenExpired = false;
  let expiryTimer = null;
  let retried = false;

  const armExpiryTimer = () => {
    clearTimeout(expiryTimer);
    if (!accessTokenExpiresAt) return;
    const remaining = accessTokenExpiresAt - Date.now();
    if (remaining <= 0) {
      tokenExpired = true;
      clearAccessToken();
      notifyAuthExpired();
      controller.abort();
      return;
    }
    expiryTimer = setTimeout(() => {
      tokenExpired = true;
      clearAccessToken();
      notifyAuthExpired();
      controller.abort();
    }, remaining);
  };

  const abortFromExternal = () => controller.abort(externalSignal?.reason);
  if (externalSignal) {
    if (externalSignal.aborted) abortFromExternal();
    else externalSignal.addEventListener("abort", abortFromExternal, { once: true });
  }

  const openStream = async () => {
    const headers = {
      Accept: "text/event-stream",
      "Content-Type": "application/json",
    };
    if (accessToken) headers.Authorization = `Bearer ${accessToken}`;
    return fetch(`${API_BASE_URL}/query/stream`, {
      method: "POST",
      headers,
      credentials: "include",
      body: JSON.stringify(payload),
      signal: controller.signal,
    });
  };

  try {
    if (accessToken && accessTokenExpiresAt && Date.now() >= accessTokenExpiresAt) {
      try {
        await refreshAccessToken();
      } catch {
        clearAccessToken();
        notifyAuthExpired();
        throw new Error("登录已过期，请重新登录");
      }
    }

    armExpiryTimer();
    let response = await openStream();
    if (response.status === 401 && !retried) {
      retried = true;
      try {
        await refreshAccessToken();
      } catch {
        clearAccessToken();
        notifyAuthExpired();
        throw new Error("登录已过期，请重新登录");
      }
      armExpiryTimer();
      response = await openStream();
    }

    if (!response.ok || !response.body) {
      if (response.status === 401) {
        clearAccessToken();
        notifyAuthExpired();
      }
      throw await createResponseError(response);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const events = buffer.split(/\r?\n\r?\n/);
      buffer = events.pop() || "";
      for (const event of events) {
        await parseSseEvent(event, handlers);
      }
    }

    buffer += decoder.decode();
    if (buffer.trim()) await parseSseEvent(buffer, handlers);
  } catch (error) {
    if (tokenExpired) {
      const authError = new Error("登录已过期，请重新登录");
      authError.code = "AUTH_EXPIRED";
      throw authError;
    }
    throw error;
  } finally {
    clearTimeout(expiryTimer);
    externalSignal?.removeEventListener("abort", abortFromExternal);
  }
};
