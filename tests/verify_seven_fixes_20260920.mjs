import assert from "node:assert/strict";
import fs from "node:fs";
import { chromium } from "../frontend/node_modules/playwright-core/index.mjs";

const edgePath = "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe";
const baseUrl = process.env.UI_BASE_URL || "http://localhost:5173/";
const browser = await chromium.launch({ executablePath: edgePath, headless: true });
const page = await browser.newPage({ viewport: { width: 1280, height: 900 }, acceptDownloads: true });
const onePixelPng = Buffer.from(
  "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Y9Z0L8AAAAASUVORK5CYII=",
  "base64",
);

const feedbackPayloads = [];
let regenerationPayload = null;
let exportCalls = 0;

const messages = [
  { message_id: 1, role: "user", content: "请分析2025年1月至7月最大降水，并给出对应过程" },
  {
    message_id: 2,
    role: "assistant",
    content: "版本一：最大降水为359.3毫米。",
    metadata: {
      agent_type: "document_rag", status: "completed", version: 1,
      version_group_id: "user-1", source_user_message_id: 1,
    },
  },
  {
    message_id: 3,
    role: "assistant",
    content: "版本二：最大降水为359.3毫米，发生于7月2日至3日，地点为阳高。",
    metadata: {
      agent_type: "document_rag", status: "completed", version: 2,
      version_group_id: "user-1", source_user_message_id: 1,
      regenerated_from_message_id: 2,
    },
  },
  { message_id: 4, role: "user", content: "生成正式多维报告" },
  {
    message_id: 5,
    role: "assistant",
    content: "正式多维报告已生成。",
    metadata: {
      agent_type: "case_multidim_search", status: "completed", version: 1,
      version_group_id: "user-4", source_user_message_id: 4,
      reports: [{
        report_id: "formal-report", report_mode: "formal", title: "正式多维分析报告",
        filename: "formal.pdf", url: "/api/case-multidim/reports/formal.pdf",
        thumbnail_url: "/api/case-multidim/report-thumbnails/formal.thumbnail.png",
      }],
    },
  },
];

await page.route("**/api/**", async (route) => {
  const request = route.request();
  const url = request.url();
  const method = request.method();
  if (url.endsWith("/api/sessions") && method === "GET") {
    return route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({
      sessions: [{ session_id: "ui-20260920", title: "七项优化验收" }],
    }) });
  }
  if (url.includes("/api/sessions/ui-20260920/messages?") && method === "GET") {
    return route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ messages }) });
  }
  if (/\/feedback$/.test(url) && method === "POST") {
    const payload = request.postDataJSON();
    feedbackPayloads.push(payload);
    return route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({
      rating: payload.rating, reason: payload.reason || "",
    }) });
  }
  if (url.endsWith("/api/query/stream") && method === "POST") {
    regenerationPayload = request.postDataJSON();
    const events = [
      { type: "metadata", agent_type: "document_rag", status: "completed", message_id: 6, version: 3, version_group_id: "user-1" },
      { type: "delta", text: "版本三：已按反馈补充关键证据。" },
      { type: "done", status: "completed", agent_type: "document_rag", message_id: 6, version: 3, version_group_id: "user-1" },
    ];
    const body = events.map((event) => `data: ${JSON.stringify(event)}\n\n`).join("");
    return route.fulfill({ status: 200, contentType: "text/event-stream; charset=utf-8", body });
  }
  if (/\/messages\/6\/export-pdf$/.test(url) && method === "POST") {
    exportCalls += 1;
    // 保留短暂等待，供浏览器验证右下角生成状态确实可见。
    await new Promise((resolve) => setTimeout(resolve, 700));
    return route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({
      report_id: "message-export", report_mode: "message_export", download_only: true,
      filename: "message-answer.pdf", url: "/api/case-multidim/reports/message-answer.pdf",
    }) });
  }
  if (url.endsWith("/api/case-multidim/reports/message-answer.pdf")) {
    return route.fulfill({
      status: 200,
      headers: { "Content-Type": "application/pdf", "Content-Disposition": "attachment; filename=message-answer.pdf" },
      body: Buffer.from("%PDF-1.4\n%%EOF"),
    });
  }
  if (url.includes("report-thumbnails")) {
    return route.fulfill({ status: 200, contentType: "image/png", body: onePixelPng });
  }
  if (url.endsWith("/api/case-multidim/reports/formal.pdf")) {
    return route.fulfill({ status: 200, contentType: "application/pdf", body: Buffer.from("%PDF-1.4\n%%EOF") });
  }
  return route.fulfill({ status: 404, body: "not found" });
});

const result = {};
try {
  await page.goto(baseUrl, { waitUntil: "networkidle" });
  const firstAnswer = page.locator(".message.assistant").first();
  await firstAnswer.getByText("版本二：最大降水为359.3毫米，发生于7月2日至3日，地点为阳高。").waitFor();

  // 同一原问题默认只显示最新回答，并可在一个显示槽内切换 1 / 2。
  result.visibleVersionGroups = await page.locator(".version-switcher").count();
  result.initialVersionLabel = await firstAnswer.locator(".version-label").innerText();
  assert.equal(result.initialVersionLabel.trim(), "2 / 2");
  assert.equal(await page.getByText("版本一：最大降水为359.3毫米。").count(), 0);
  await firstAnswer.getByRole("button", { name: "查看上一版本" }).click();
  await page.getByText("版本一：最大降水为359.3毫米。").waitFor();
  assert.equal((await page.locator(".message.assistant").first().locator(".version-label").innerText()).trim(), "1 / 2");
  await page.locator(".message.assistant").first().getByRole("button", { name: "查看下一版本" }).click();
  await page.getByText("版本二：最大降水为359.3毫米，发生于7月2日至3日，地点为阳高。").waitFor();

  result.followUpCount = await page.locator(".follow-up-suggestions button").count();
  result.followUpPromptCount = await page.getByText("接下来你可以继续了解：", { exact: true }).count();
  assert.equal(result.followUpCount, 0);
  assert.equal(result.followUpPromptCount, 0);

  // 负反馈先提交原因，再检查深色选中态和刷新前的互斥状态。
  const improvementButton = page.locator(".message.assistant").first().getByRole("button", { name: "需要改进" });
  await improvementButton.click();
  await page.getByText("遗漏关键信息", { exact: true }).click();
  await page.locator(".feedback-description textarea").fill("缺少关键证据来源");
  await page.getByRole("button", { name: "仅提交反馈" }).click();
  await page.locator('.action-button.active[aria-label="需要改进"]').waitFor();
  result.feedbackActiveColor = await improvementButton.evaluate((element) => getComputedStyle(element).backgroundColor);
  assert.equal(result.feedbackActiveColor, "rgb(41, 41, 39)");

  // 已选中时先取消，再重新打开并按同一原因生成新版本。
  await improvementButton.click();
  await improvementButton.click();
  await page.getByText("遗漏关键信息", { exact: true }).click();
  await page.locator(".feedback-description textarea").fill("补充观测站与时间证据");
  await page.getByRole("button", { name: "按此原因重新生成" }).click();
  await page.waitForTimeout(800);
  if (await page.getByText("版本三：已按反馈补充关键证据。").count() === 0) {
    const diagnostic = {
      regenerationPayload,
      bodyText: await page.locator("body").innerText(),
    };
    fs.writeFileSync("tests/seven-fixes-failure-20260920.json", JSON.stringify(diagnostic, null, 2), "utf8");
    await page.screenshot({ path: "tests/seven-fixes-failure-20260920.png", fullPage: true });
    throw new Error("按反馈重新生成后没有显示新版本，诊断已写入 tests/seven-fixes-failure-20260920.json");
  }
  result.regeneratedVersionLabel = (await page.locator(".message.assistant").first().locator(".version-label").innerText()).trim();
  assert.equal(result.regeneratedVersionLabel, "3 / 3");
  assert.equal(regenerationPayload.regenerate_from_message_id, 3);
  assert.match(regenerationPayload.regeneration_instruction, /遗漏关键信息/);

  // 消息级 PDF 只下载，右下角显示旋转状态；正式报告卡仍保留且数量不增加。
  const reportCardsBefore = await page.locator(".pdf-card").count();
  assert.equal(reportCardsBefore, 1);
  await page.evaluate(() => {
    window.__messagePdfDownload = null;
    document.addEventListener("click", (event) => {
      const anchor = event.target instanceof HTMLAnchorElement ? event.target : null;
      if (!anchor?.download) return;
      window.__messagePdfDownload = { href: anchor.href, filename: anchor.download };
      event.preventDefault();
    }, true);
  });
  await page.locator(".message.assistant").first().getByRole("button", { name: "更多操作" }).click();
  await page.locator(".el-dropdown-menu__item:visible").filter({ hasText: "导出到 PDF" }).last().click();
  const downloadStatus = page.locator(".pdf-download-status");
  await downloadStatus.waitFor();
  const statusBox = await downloadStatus.boundingBox();
  const viewport = page.viewportSize();
  assert(statusBox && viewport);
  result.pdfStatusRightGap = Math.round(viewport.width - statusBox.x - statusBox.width);
  result.pdfStatusBottomGap = Math.round(viewport.height - statusBox.y - statusBox.height);
  assert(result.pdfStatusRightGap <= 30 && result.pdfStatusBottomGap <= 30);
  await page.waitForFunction(() => Boolean(window.__messagePdfDownload));
  const download = await page.evaluate(() => window.__messagePdfDownload);
  result.downloadFilename = download.filename;
  result.downloadHref = download.href;
  assert.equal(result.downloadFilename, "message-answer.pdf");
  assert.match(result.downloadHref, /\/api\/case-multidim\/reports\/message-answer\.pdf$/);
  await downloadStatus.waitFor({ state: "hidden" });
  result.reportCardsAfter = await page.locator(".pdf-card").count();
  assert.equal(result.reportCardsAfter, 1);
  assert.equal(exportCalls, 1);

  await page.screenshot({ path: "tests/seven-fixes-desktop-20260920.png", fullPage: true });
  await page.setViewportSize({ width: 390, height: 844 });
  await page.waitForTimeout(100);
  result.mobileOverflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
  assert(result.mobileOverflow <= 1);
  await page.screenshot({ path: "tests/seven-fixes-mobile-20260920.png", fullPage: true });

  result.feedbackPayloads = feedbackPayloads;
  result.regenerationPayload = regenerationPayload;
  result.exportCalls = exportCalls;
  fs.writeFileSync("tests/seven-fixes-20260920.json", JSON.stringify(result, null, 2), "utf8");
  console.log(JSON.stringify(result, null, 2));
} finally {
  await browser.close();
}
