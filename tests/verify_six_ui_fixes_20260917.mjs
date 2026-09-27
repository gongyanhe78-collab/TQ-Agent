import { chromium } from "../frontend/node_modules/playwright-core/index.mjs";
import fs from "node:fs";

const edgePath = process.env.EDGE_PATH || "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe";
const baseUrl = process.env.CHAT_UI_URL || "http://127.0.0.1:5173/";
const onePixelPng = Buffer.from(
  "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=",
  "base64",
);

const browser = await chromium.launch({ executablePath: edgePath, headless: true });
const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
let reportPdfRequests = 0;

// 使用发生问题时的三类消息构造稳定回归场景，不依赖云端模型响应。
const smartResult = {
  query_summary: {
    process_name: "分析2025年台风“白海豚”残余涡旋与副高共同影响晋东南的强降水过程，并作出预警",
    disaster_types: ["强降水"],
    affected_areas: ["晋东南"],
    circulation_description: "台风残余涡旋与副高共同影响",
  },
  matched_cases: [{
    case_id: "case-1", rank: 1, title: "7月2-3日分散性暴雨天气过程", retrieval_score: 0.82,
    date_range: "2025年7月2-3日", source_pdf: "FST2025-7.pdf", disaster_types: ["暴雨"],
    affected_areas: ["长治"], match_reasons: ["环流配置相近"], key_differences: ["落区不同"],
    key_references: [{ text: "在相近配置下应关注短时强降水。", evidence_chunk_ids: ["chunk-006"], evidence_image_refs: [{ image_id: "image-1", label: "图1" }] }],
    evidence_images: [{ image_id: "image-1", url: "/mock/image.png", caption: "环流形势图" }], supplemental_images: [],
  }],
  forecast_summary: {
    similarity_assessment: "当前过程与两个历史个例在环流配置上高度相似。",
    core_features: ["副热带高压边缘暖湿气流共同作用", "短时强降水特征明显"],
    main_risk: "山洪和中小河流洪水风险",
    confidence: 0.85,
    support_case_ids: ["case-1", "case-2"],
  },
  forecast_tips: [{ priority: 1, focus_object: "晋东南强降水", possible_bias: "可能低估累计雨量", suggested_action: "加强雷达监测", confidence: 0.9, consensus_level: "高共识" }],
};

const messages = [
  { message_id: 1, role: "assistant", content: "2025年5月至7月灾害分析。", metadata: { agent_type: "document_rag", images: [{ image_id: "rag-image", url: "/mock/image.png" }], image_display_mode: "hidden" } },
  { message_id: 2, role: "assistant", content: "已根据上一轮分析结果生成简易 PDF 汇总报告。", metadata: { agent_type: "conversation_report", images: [{ image_id: "report-image", url: "/mock/image.png" }], reports: [{ report_id: "report-1", title: "简易汇总报告", filename: "report.pdf", url: "/api/case-multidim/reports/report.pdf", thumbnail_url: "/api/case-multidim/report-thumbnails/report.thumbnail.png" }] } },
  { message_id: 3, role: "assistant", content: "Smart 结构化结果", metadata: { agent_type: "smart_case_match", agent_result: smartResult } },
];

await page.route("http://127.0.0.1:8000/api/**", async (route) => {
  const url = route.request().url();
  if (url.endsWith("/api/sessions")) return route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ sessions: [{ session_id: "ui-regression", title: "六项问题回归" }] }) });
  if (url.includes("/api/sessions/ui-regression/messages")) return route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ messages }) });
  if (url.includes("report-thumbnails") || url.includes("/mock/image.png")) return route.fulfill({ status: 200, contentType: "image/png", body: onePixelPng });
  if (url.endsWith("report.pdf")) {
    reportPdfRequests += 1;
    return route.fulfill({ status: 200, contentType: "application/pdf", body: Buffer.from("%PDF-1.4\n%%EOF") });
  }
  return route.fulfill({ status: 404, body: "not found" });
});

const result = {};
try {
  await page.goto(baseUrl, { waitUntil: "networkidle" });
  await page.locator(".smart-result").waitFor();
  result.unrequestedImageBlocks = await page.locator(".answer-image-references").count();
  result.reportCards = await page.locator(".pdf-card").count();
  result.reportThumbnailSrc = await page.locator(".pdf-thumbnail img").getAttribute("src");
  result.reportThumbnailLoaded = await page.locator(".pdf-thumbnail img").evaluate((image) => image.complete && image.naturalWidth > 0);
  result.smartTitle = await page.locator(".query-overview > strong").innerText();
  result.referenceBackground = await page.locator(".reference-item").evaluate((element) => getComputedStyle(element).backgroundColor);
  result.coreSummary = await page.locator(".core-summary").innerText();
  result.forecastTipCount = await page.locator(".forecast-tips > li").count();

  if (result.unrequestedImageBlocks !== 0) throw new Error("未请求图片时仍展示了图片区");
  if (result.reportCards !== 1) throw new Error("承接式报告卡片数量不正确");
  if (!result.reportThumbnailSrc?.includes("report.thumbnail.png")) throw new Error("报告卡片未使用持久化缩略图");
  if (!result.reportThumbnailLoaded) throw new Error("报告缩略图首次加载失败");
  if (!result.smartTitle.includes("白海豚")) throw new Error("Smart 完整过程摘要未显示");
  if (result.referenceBackground !== "rgba(0, 0, 0, 0)") throw new Error("参考经验仍存在独立背景色");
  if (!result.coreSummary.includes("核心结论摘要") || !result.coreSummary.includes("最大风险")) throw new Error("Smart 核心结论摘要缺失");
  if (result.forecastTipCount !== 1) throw new Error("综合研判明细未完整显示");
  if (reportPdfRequests !== 0) throw new Error("完整 PDF 在用户点击前被提前加载");
  await page.locator(".pdf-thumbnail-action").click();
  await page.locator(".pdf-dialog").waitFor();
  await page.waitForFunction(() => document.querySelector(".pdf-frame"));
  result.reportPdfRequestsAfterClick = reportPdfRequests;
  if (result.reportPdfRequestsAfterClick !== 1) throw new Error("点击查看后没有按需加载完整 PDF");
  await page.keyboard.press("Escape");

  // 刷新页面模拟重新打开会话，稳定 URL 应继续加载而不是出现 Blob/embed 黑屏。
  await page.reload({ waitUntil: "networkidle" });
  await page.locator(".pdf-thumbnail img").waitFor();
  result.reportThumbnailLoadedAfterReopen = await page.locator(".pdf-thumbnail img").evaluate((image) => image.complete && image.naturalWidth > 0);
  if (!result.reportThumbnailLoadedAfterReopen) throw new Error("重新打开会话后报告缩略图加载失败");

  await page.screenshot({ path: "tests/six-ui-fixes-20260917.png", fullPage: true });
  fs.writeFileSync("tests/six-ui-fixes-20260917.json", JSON.stringify(result, null, 2), "utf8");
  console.log(JSON.stringify(result, null, 2));
} finally {
  await browser.close();
}
