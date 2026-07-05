import axios from "axios";

const API_BASE_URL = "http://127.0.0.1:8000/api";

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 60000,
});

export const fetchHealth = async () => (await api.get("/health")).data;
export const fetchCases = async () => (await api.get("/cases")).data;
export const fetchVectorKeys = async () => (await api.get("/vectors/keys")).data;
export const fetchCaseDetail = async (caseId) => (await api.get(`/cases/${caseId}`)).data;
export const fetchDocumentKeys = async () => (await api.get("/documents/keys")).data;
export const fetchDocumentChunk = async (chunkId) => (await api.get(`/documents/chunks/${chunkId}`)).data;
export const fetchStandardCases = async (params = {}) => (await api.get("/standard-cases", { params })).data;
export const fetchSimilarCases = async (params = {}) => (await api.get("/standard-cases/similar", { params })).data;
export const fetchEvalQuestions = async () => (await api.get("/eval/questions")).data;
export const runQuery = async (payload) => (await api.post("/query", payload)).data;
export const runExtraction = async () => (await api.post("/extract")).data;
export const runIndexing = async () => (await api.post("/index")).data;
export const refreshLibrary = async () => (await api.post("/refresh", null, { timeout: 600000 })).data;

export const uploadMaterial = async (file) =>
  (
    await api.post("/materials", file, {
      headers: {
        "Content-Type": "application/pdf",
        "X-Filename": encodeURIComponent(file.name),
      },
      timeout: 600000,
    })
  ).data;

// Upload multiple PDFs through FastAPI's standard UploadFile multipart endpoint.
export const uploadMaterialsBatch = async (files) => {
  const formData = new FormData();
  Array.from(files).forEach((file) => formData.append("files", file));
  return (await api.post("/materials/batch", formData, { timeout: 600000 })).data;
};

export const fetchSessions = async () => (await api.get("/sessions")).data;
export const createSession = async (title) => (await api.post("/sessions", { title })).data;
export const updateSessionTitle = async (sessionId, title) =>
  (await api.patch(`/sessions/${sessionId}`, { title })).data;
export const fetchSessionMessages = async (sessionId) =>
  (await api.get(`/sessions/${sessionId}/messages`)).data;
export const deleteSession = async (sessionId) => (await api.delete(`/sessions/${sessionId}`)).data;

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
      if (data.type === "delta") handlers.onDelta?.(data.text || "");
      if (data.type === "done") handlers.onDone?.(data);
      if (data.type === "error") throw new Error(data.message);
    }
  }
};
