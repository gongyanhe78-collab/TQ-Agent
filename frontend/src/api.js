import axios from "axios";

// 默认连接标准后端端口；开发调试可用环境变量或浏览器本地配置临时切换。
const browserApiOrigin = typeof window !== "undefined" ? window.localStorage.getItem("risk-api-origin") : "";
export const API_ORIGIN = import.meta.env.VITE_API_ORIGIN || browserApiOrigin || "http://127.0.0.1:8000";
const API_BASE_URL = `${API_ORIGIN}/api`;

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 60000,
});

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

// 统一解析后端的 metadata/delta/done 事件，并将 Agent 差异屏蔽在聊天协议之后。
export const streamQuery = async (payload, handlers) => {
  const response = await fetch(`${API_BASE_URL}/query/stream`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok || !response.body) {
    throw new Error(`Stream query failed: ${response.status}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const events = buffer.split("\n\n");
    buffer = events.pop() || "";
    for (const event of events) {
      const line = event.split("\n").find((item) => item.startsWith("data: "));
      if (!line) continue;
      const data = JSON.parse(line.slice(6));
      if (data.type === "metadata") handlers.onMetadata?.(data);
      if (data.type === "delta") {
        handlers.onDelta?.(data.text || "");
        // 主动让出一个浏览器渲染帧，避免连续 SSE 小块都在同一轮微任务中合并显示。
        await new Promise((resolve) => setTimeout(resolve, 0));
      }
      if (data.type === "done") handlers.onDone?.(data);
      if (data.type === "error") throw new Error(data.message);
    }
  }
};
