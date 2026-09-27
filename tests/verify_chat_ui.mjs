import { chromium } from "../frontend/node_modules/playwright-core/index.mjs";

const edgePath = "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe";
const baseUrl = process.env.CHAT_UI_URL || "http://127.0.0.1:5173/";

// 使用本机 Edge 执行真实布局检查，避免测试时额外下载浏览器。
const browser = await chromium.launch({
  executablePath: edgePath,
  headless: true,
  args: ["--disable-gpu", "--disable-features=Vulkan,WebGPU"],
});

const results = {};
try {
  const desktopPage = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
  await desktopPage.goto(baseUrl, { waitUntil: "networkidle" });
  await desktopPage.locator(".message.assistant").last().waitFor();

  const markdownTable = desktopPage.locator(".markdown-body table").first();
  results.markdownTableCount = await desktopPage.locator(".markdown-body table").count();
  if (!results.markdownTableCount) {
    throw new Error("历史回答中未找到已经渲染的 Markdown 表格");
  }

  results.tableTag = await markdownTable.evaluate((element) => element.tagName);
  results.assistantStyle = await desktopPage.locator(".message.assistant .bubble").last().evaluate((element) => {
    const style = getComputedStyle(element);
    return { backgroundColor: style.backgroundColor, borderStyle: style.borderStyle };
  });
  results.userStyle = await desktopPage.locator(".message.user .bubble").last().evaluate((element) => {
    const style = getComputedStyle(element);
    return { backgroundColor: style.backgroundColor, color: style.color };
  });

  await markdownTable.scrollIntoViewIfNeeded();
  await desktopPage.screenshot({ path: "tests/chat-ui-table-desktop.png", fullPage: false });

  const mobilePage = await browser.newPage({ viewport: { width: 390, height: 844 } });
  await mobilePage.goto(baseUrl, { waitUntil: "networkidle" });
  await mobilePage.locator(".composer").waitFor();
  results.mobile = await mobilePage.evaluate(() => {
    const sendButton = document.querySelector(".composer .el-button");
    const buttonRect = sendButton?.getBoundingClientRect();
    return {
      bodyScrollWidth: document.body.scrollWidth,
      viewportWidth: window.innerWidth,
      sendButtonRight: buttonRect?.right || 0,
      sendButtonVisible: Boolean(buttonRect && buttonRect.left >= 0 && buttonRect.right <= window.innerWidth),
    };
  });
  await mobilePage.screenshot({ path: "tests/chat-ui-mobile.png", fullPage: false });

  if (results.tableTag !== "TABLE") throw new Error("Markdown 表格未生成语义化 table 元素");
  if (results.assistantStyle.backgroundColor !== "rgba(0, 0, 0, 0)") throw new Error("助手回答仍有气泡背景");
  if (results.assistantStyle.borderStyle !== "none") throw new Error("助手回答仍有气泡边框");
  if (!results.mobile.sendButtonVisible || results.mobile.bodyScrollWidth > results.mobile.viewportWidth) {
    throw new Error("移动端存在横向溢出或发送按钮不可见");
  }

  console.log(JSON.stringify(results, null, 2));
} finally {
  await browser.close();
}
