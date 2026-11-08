import DOMPurify from "dompurify";
import { marked } from "marked";

marked.setOptions({
  breaks: true,
  gfm: true,
});

// 图片 ID 按固定命名格式识别，不依赖当前回答附带的图片列表是否包含该资源。
const IMAGE_ID_PATTERN = /^.+-page-\d{3}-(?:snapshot|image-\d{3})$/;

const escapeHtml = (value) =>
  String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");

// 回答采用 GFM Markdown 渲染，清洗 HTML 后再写入页面，兼顾表格展示与内容安全。
export const renderMarkdown = (content) => {
  const markdown = String(content || "")
    .replace(
      /(\*\*|__)?对应图片ID[:：]?(\*\*|__)?[:：]?/g,
      (_match, open = "", close = "") => `${open}对应图片${close}`
    )
    .replace(/`([^`\r\n]+)`/g, (source, rawImageId) => {
      const imageId = rawImageId.trim();
      if (!IMAGE_ID_PATTERN.test(imageId)) return source;
      const safeImageId = escapeHtml(imageId);
      return `<a href="#" class="markdown-image-id" data-image-id="${safeImageId}" role="button" aria-label="查看图片 ${safeImageId}"><code>${safeImageId}</code></a>`;
    });

  return DOMPurify.sanitize(marked.parse(markdown), {
    ADD_ATTR: ["data-image-id", "role", "aria-label"],
  });
};

// Smart 的完整文字答案仍会保存到会话，但页面只展示结构化结果，避免综合研判重复出现。
export const showTextContent = (message) =>
  Boolean(message?.content && message?.agent_type !== "smart_case_match");

// 后端 metadata 阶段使用 running，前端占位消息使用 streaming，两者都属于处理中状态。
export const isProcessingMessage = (message) =>
  Boolean(
    message?.role === "assistant" &&
    (message.status === "streaming" || message.status === "running")
  );

// 消息级 PDF 已经直接下载，历史 metadata 中即使保留记录也不应重新显示报告卡。
export const visibleReports = (message) =>
  (message.reports || []).filter((report) => report.report_mode !== "message_export");

// 只有已经落库且不再流式生成的回答才允许执行消息级操作。
export const showMessageActions = (message, sessionId) =>
  Boolean(
    sessionId &&
    message?.role === "assistant" &&
    message?.message_id &&
    message?.status === "completed"
  );

// 柱状图以当前结果中的最大值为基准计算相对宽度。
export const barPercent = (value, items = []) => {
  const maximum = Math.max(...items.map((item) => Number(item.value) || 0), 1);
  return Math.max(2, Math.round(((Number(value) || 0) / maximum) * 100));
};

// 通过事件委托捕获 v-html 中的图片 ID；未随回答返回的图片改由 ID 接口直接加载。
export const resolveMarkdownImage = (event, message) => {
  const link = event.target?.closest?.(".markdown-image-id[data-image-id]");
  if (!link) return null;
  event.preventDefault();
  const imageId = link.getAttribute("data-image-id");
  const image = (message.images || []).find((item) => String(item.image_id || "") === imageId);
  return {
    ...(image || {}),
    image_id: imageId,
    url: image?.url || `/api/image-evidence/${encodeURIComponent(imageId)}`,
  };
};
