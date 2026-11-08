# 天气主题界面点缀设计规格

> 日期：2026-11-08
> 状态：已确认（登录页选 C「轻缀微调」，聊天页按轻缀落点组合）
> 关联计划：`docs/superpowers/plans/2026-11-08-weather-themed-ui.md`

## 1. 背景与目标

当前登录页与聊天页是极简灰白风格（#fbfbfa 米白背景、#2d2d2b 深灰按钮），整体朴素。用户希望参考 README.md 中的天气 emoji（🌦️🌧️☀️❄️ 等）与 Element Plus 的图标风格，给登录页和聊天页加入天气元素，使其更生动但不破坏现有极简风格。

**目标：** 采用「轻缀微调」风格——保持现有米白极简配色与组件样式，仅在几处细节点缀天气 emoji 与极淡渐变，不改变任何功能与 SSE 协议。

## 2. 设计确认（用户已批准）

### 登录页（方案 C）
- 卡片顶部新增一行天气 emoji 装饰：`🌦️ 🌤️ ⛅ 🌧️`（`aria-hidden="true"`，纯装饰）。
- 页面背景从纯 `#fbfbfa` 改为极淡的天际渐变：`linear-gradient(180deg, #fbfbfa 0%, #f4f8fb 100%)`。
- 其余（深灰 brand-mark「气象」、标题、表单、按钮）保持不变。

### 聊天页
1. **顶栏**：标题前加 `🌦️`，右侧新增当前日期显示（按小时选 emoji：白天 `☀️`，夜晚 `🌙`），格式如 `☀️ 11月8日 · 星期日`。
2. **空会话欢迎区**（新元素）：当消息列表为空且未加载时显示——
   - 大号天气 emoji 组合（🌦️ 🌤️ ⛅）
   - 标题「天气与气象灾害，问我吧」、副标题「可查询山西省历史个例、统计报告，也可问实时天气」
   - 三个示例问题胶囊按钮，点击后**将纯文字填入输入框**（emoji 仅作按钮装饰，不进入问题文本）：
     - `❄️ 分析2025年山西寒潮天气过程`
     - `🌧️ 检索2025年3月暴雨个例`
     - `☀️ 今天重庆天气怎么样`
3. **消息区背景**：改为极淡的天际渐变 `linear-gradient(180deg, #fbfbfa 0%, #f6f9fc 60%, #fbfbfa 100%)`。
4. **会话侧栏**：标题「聊天记录」前加 `💬`。

## 3. 组件改动

### `frontend/src/components/auth/LoginScreen.vue`
- 模板：`.login-panel` 内、`.brand-mark` 上方新增 `<div class="weather-decor" aria-hidden="true">🌦️ 🌤️ ⛅ 🌧️</div>`。
- 样式：`.weather-decor`（字体大小、字距、颜色），`.login-page` 背景改渐变。

### `frontend/src/components/ChatPanel.vue`
- 顶栏：`<h1>` 前加天气 span；右侧新增日期 span（`computed` 生成）。
- 消息区：`v-if="!messages.length && !loading"` 显示欢迎区，否则渲染 `ChatMessages`。
- 欢迎区：三个示例按钮，`@click` 调用 `composerRef.value.setQuestion(text)`。
- 样式：`.topbar-weather`、`.topbar-date`、`.welcome`、`.suggestion`、消息区渐变背景。

### `frontend/src/components/chat/ChatComposer.vue`
- 新增 `setQuestion(text)` 方法并 `defineExpose({ setQuestion })`，供 ChatPanel 填入示例问题。

### `frontend/src/components/SessionList.vue`
- 标题改为 `💬 聊天记录`。

## 4. 不变项与约束

- 所有文档、注释使用中文。
- 不改动后端、SSE 协议、消息渲染逻辑、Element Plus 组件行为。
- 示例问题填入输入框时**不含 emoji 前缀**，避免干扰后端意图识别。
- 视觉验证方式：`npm run build` 构建通过 + 浏览器截图人工确认。

## 5. 验证

1. `npm run build` 构建成功（frontend）。
2. 浏览器验证：登录页天气装饰与渐变；聊天页顶栏日期、空态欢迎区、示例问题填入、消息区渐变。
3. 后端不受影响，无需改动。
