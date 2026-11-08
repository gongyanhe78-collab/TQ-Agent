# 天气主题界面点缀实现计划

> **执行提示：** 本计划按任务逐步实现，每个任务以 `npm run build` 作为验证关卡并独立提交。前端无单测框架，采用"构建 + 浏览器视觉确认"验证。
> **目标：** 按已批准的「轻缀微调」风格，给登录页与聊天页加入天气 emoji 与极淡渐变点缀，不改功能与 SSE 协议。
> **架构：** 纯前端样式与模板改动（Vue 3 + Element Plus），涉及 4 个组件：LoginScreen、ChatPanel、ChatComposer、SessionList。示例问题通过 ChatComposer 暴露的 `setQuestion` 填入输入框。
> **技术栈：** Vue 3.5 / Element Plus 2.8 / Vite。
> **规格：** `docs/superpowers/specs/2026-11-08-weather-themed-ui-design.md`

## 全局约束

- 所有文档、注释使用中文。
- 不修改后端与 SSE 协议；示例问题填入输入框时不含 emoji 前缀。
- 验证命令：`cd frontend && npm run build`；Docker 更新：`docker compose build frontend && docker compose up -d frontend`。
- 工作分支：`codex/phase1-enterprise-transition`。

---

### Task 1：登录页天气点缀

**Files:**
- Modify: `frontend/src/components/auth/LoginScreen.vue`

**Interfaces:**
- Produces: 登录卡片顶部天气 emoji 装饰行 + 极淡天际渐变背景（纯视觉，无接口）。

- [ ] **Step 1: 修改模板**

在 `LoginScreen.vue` 模板中、`<div class="brand-mark" ...>` 之前插入：

```html
      <div class="weather-decor" aria-hidden="true">🌦️ 🌤️ ⛅ 🌧️</div>
```

- [ ] **Step 2: 修改样式**

`.login-page` 背景改为：

```css
.login-page {
  align-items: center;
  background: linear-gradient(180deg, #fbfbfa 0%, #f4f8fb 100%);
  color: #292927;
  display: flex;
  justify-content: center;
  min-height: 100%;
  padding: 28px 20px;
}
```

在 `.brand-mark` 样式前新增：

```css
.weather-decor {
  color: #8f9aa8;
  font-size: 20px;
  letter-spacing: 5px;
  line-height: 1;
  margin-bottom: 14px;
  text-align: center;
}
```

- [ ] **Step 3: 构建验证**

Run: `cd frontend && npm run build`
Expected: 构建成功

- [ ] **Step 4: 提交**

```bash
git add frontend/src/components/auth/LoginScreen.vue
git commit -m "feat: 登录页添加天气点缀与天际渐变"
```

---

### Task 2：聊天页顶栏天气与消息区渐变

**Files:**
- Modify: `frontend/src/components/ChatPanel.vue`

**Interfaces:**
- Produces: 顶栏标题前天气 emoji、右侧日期（computed `todayLabel`）、消息区极淡渐变（纯视觉）。

- [ ] **Step 1: 修改模板**

顶栏 `<header class="topbar">` 内改为：

```html
    <header class="topbar">
      <div class="topbar-title">
        <span class="topbar-weather" aria-hidden="true">🌦️</span>
        <h1>气象灾害个例问答助手</h1>
      </div>
      <span class="topbar-date" aria-hidden="true">{{ todayLabel }}</span>
    </header>
```

- [ ] **Step 2: 修改脚本**

`<script setup>` 中新增 `computed` 并生成日期标签（放在 `const emit = defineEmits(...)` 之后）：

```js
import { computed, nextTick, ref, watch } from "vue";

// 顶栏右侧显示当天日期，并按时段切换天气 emoji。
const todayLabel = computed(() => {
  const now = new Date();
  const week = ["日", "一", "二", "三", "四", "五", "六"][now.getDay()];
  const hour = now.getHours();
  const icon = hour >= 6 && hour < 18 ? "☀️" : "🌙";
  return `${icon} ${now.getMonth() + 1}月${now.getDate()}日 · 星期${week}`;
});
```

注意：若现有 `import` 行不含 `computed`，把 `computed` 加入 import。

- [ ] **Step 3: 修改样式**

`.topbar` 改为两端对齐，并新增 `.topbar-title`、`.topbar-weather`、`.topbar-date`：

```css
.topbar {
  align-items: center;
  background: rgba(251, 251, 250, 0.94);
  border-bottom: 1px solid #ededeb;
  display: flex;
  justify-content: space-between;
  padding: 0 max(24px, calc((100% - 800px) / 2));
}

.topbar-title {
  align-items: center;
  display: flex;
  gap: 7px;
}

.topbar-weather {
  font-size: 16px;
  line-height: 1;
}

.topbar-date {
  color: #8f8f89;
  font-size: 12px;
}
```

`.messages` 背景改为：

```css
.messages {
  background: linear-gradient(180deg, #fbfbfa 0%, #f6f9fc 60%, #fbfbfa 100%);
  min-height: 0;
  overflow-y: auto;
  padding: 30px max(24px, calc((100% - 800px) / 2)) 44px;
  scrollbar-color: #d8d8d4 transparent;
  scrollbar-width: thin;
}
```

- [ ] **Step 4: 构建验证**

Run: `cd frontend && npm run build`
Expected: 构建成功

- [ ] **Step 5: 提交**

```bash
git add frontend/src/components/ChatPanel.vue
git commit -m "feat: 聊天页顶栏加入天气与日期显示"
```

---

### Task 3：空会话欢迎区与示例问题填入

**Files:**
- Modify: `frontend/src/components/ChatPanel.vue`
- Modify: `frontend/src/components/chat/ChatComposer.vue`

**Interfaces:**
- Consumes: `ChatComposer.setQuestion(text)`（Task 3 内新增）
- Produces: 空态欢迎区（天气 emoji + 欢迎语 + 示例问题按钮，点击填入输入框）

- [ ] **Step 1: ChatComposer 暴露 setQuestion**

`frontend/src/components/chat/ChatComposer.vue` 的 `<script setup>` 末尾（`const submit = ...` 之后）新增：

```js
// 供 ChatPanel 的空态示例问题把纯文字填入输入框，emoji 只作按钮装饰。
const setQuestion = (text) => {
  question.value = String(text || "");
};
defineExpose({ setQuestion });
```

- [ ] **Step 2: ChatPanel 模板增加欢迎区与引用**

`ChatPanel.vue` 模板的消息区改为（保持滚动容器不变）：

```html
    <div ref="messageList" class="messages">
      <div v-if="!messages.length && !loading" class="welcome">
        <div class="welcome-emoji" aria-hidden="true">🌦️ 🌤️ ⛅</div>
        <h2 class="welcome-title">天气与气象灾害，问我吧</h2>
        <p class="welcome-subtitle">可查询山西省历史个例、统计报告，也可问实时天气</p>
        <div class="welcome-suggestions">
          <button
            v-for="item in suggestions"
            :key="item.text"
            class="suggestion"
            type="button"
            @click="fillSuggestion(item.text)"
          >
            <span aria-hidden="true">{{ item.icon }}</span>
            {{ item.text }}
          </button>
        </div>
      </div>
      <ChatMessages
        v-else
        :messages="messages"
        :session-id="sessionId"
        @preview-image="openImagePreview"
        @regenerate="emit('regenerate', $event)"
        @select-version="emit('select-version', $event)"
      />
    </div>
```

同时把 `ChatComposer` 加上 ref：

```html
    <ChatComposer ref="composerRef" :loading="loading" @submit="emit('query', $event)" />
```

- [ ] **Step 3: ChatPanel 脚本新增建议列表与填入方法**

```js
const composerRef = ref(null);

// 空态示例问题：emoji 仅作按钮装饰，填入输入框时只保留纯文字。
const suggestions = [
  { icon: "❄️", text: "分析2025年山西寒潮天气过程" },
  { icon: "🌧️", text: "检索2025年3月暴雨个例" },
  { icon: "☀️", text: "今天重庆天气怎么样" },
];

const fillSuggestion = (text) => {
  if (composerRef.value) {
    composerRef.value.setQuestion(text);
  }
};
```

- [ ] **Step 4: ChatPanel 样式新增欢迎区**

```css
.welcome {
  align-items: center;
  display: flex;
  flex-direction: column;
  justify-content: center;
  min-height: 100%;
  padding: 24px 0 60px;
  text-align: center;
}

.welcome-emoji {
  font-size: 44px;
  letter-spacing: 8px;
  line-height: 1;
}

.welcome-title {
  color: #292927;
  font-size: 18px;
  font-weight: 650;
  margin: 18px 0 0;
}

.welcome-subtitle {
  color: #74746e;
  font-size: 13px;
  margin: 8px 0 0;
}

.welcome-suggestions {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  justify-content: center;
  margin-top: 20px;
  max-width: 560px;
}

.suggestion {
  background: #ffffff;
  border: 1px solid #deded9;
  border-radius: 16px;
  color: #555550;
  cursor: pointer;
  font-size: 13px;
  padding: 8px 14px;
}

.suggestion:hover,
.suggestion:focus {
  border-color: #b8b8b1;
  color: #292927;
}

.suggestion span {
  margin-right: 5px;
}
```

- [ ] **Step 5: 构建验证**

Run: `cd frontend && npm run build`
Expected: 构建成功

- [ ] **Step 6: 提交**

```bash
git add frontend/src/components/ChatPanel.vue frontend/src/components/chat/ChatComposer.vue
git commit -m "feat: 聊天页空态欢迎区与示例问题填入"
```

---

### Task 4：会话侧栏标题点缀

**Files:**
- Modify: `frontend/src/components/SessionList.vue`

- [ ] **Step 1: 修改标题**

`SessionList.vue` 模板中 `<h2 v-if="!collapsed">聊天记录</h2>` 改为：

```html
      <h2 v-if="!collapsed">💬 聊天记录</h2>
```

- [ ] **Step 2: 构建验证**

Run: `cd frontend && npm run build`
Expected: 构建成功

- [ ] **Step 3: 提交**

```bash
git add frontend/src/components/SessionList.vue
git commit -m "feat: 会话侧栏标题加入天气点缀"
```

---

### Task 5：整体验证、Docker 更新与收尾

**Files:** 无新增（验证与部署）

- [ ] **Step 1: 前端整体构建**

Run: `cd frontend && npm run build`
Expected: 构建成功

- [ ] **Step 2: 浏览器视觉验证**

在已登录页面检查：登录页天气装饰与渐变；聊天页顶栏日期、空态欢迎区、示例问题点击填入输入框、消息区渐变。若发现样式问题，回退对应 Task 修复。

- [ ] **Step 3: Docker 更新前端**

```bash
docker compose build frontend
docker compose up -d frontend
docker compose ps
```

Expected: `frontend` 容器重建并 healthy。

- [ ] **Step 4: 最终提交确认**

Run: `git status --short`
Expected: 无未提交改动（Task 1-4 已逐个提交）。
