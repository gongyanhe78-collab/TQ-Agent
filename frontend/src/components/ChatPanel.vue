<template>
  <section class="chat-page">
    <header class="topbar">
      <div class="topbar-title">
        <span class="topbar-weather" aria-hidden="true">🌦️</span>
        <h1>气象灾害个例问答助手</h1>
      </div>
      <span class="topbar-date" aria-hidden="true">{{ todayLabel }}</span>
    </header>

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

    <ChatComposer ref="composerRef" :loading="loading" @submit="emit('query', $event)" />

    <EvidencePreview
      v-model:open="imagePreviewOpen"
      :image="imagePreview"
    />
  </section>
</template>

<script setup>
import { computed, nextTick, ref, watch } from "vue";
import ChatComposer from "./chat/ChatComposer.vue";
import ChatMessages from "./chat/ChatMessages.vue";
import EvidencePreview from "./chat/EvidencePreview.vue";

const props = defineProps({
  messages: {
    type: Array,
    default: () => [],
  },
  loading: {
    type: Boolean,
    default: false,
  },
  sessionId: {
    type: String,
    default: "",
  },
});

const emit = defineEmits(["query", "regenerate", "select-version"]);

// 顶栏右侧显示当天日期，并按时段切换天气 emoji。
const todayLabel = computed(() => {
  const now = new Date();
  const week = ["日", "一", "二", "三", "四", "五", "六"][now.getDay()];
  const hour = now.getHours();
  const icon = hour >= 6 && hour < 18 ? "☀️" : "🌙";
  return `${icon} ${now.getMonth() + 1}月${now.getDate()}日 · 星期${week}`;
});

const messageList = ref(null);
const imagePreviewOpen = ref(false);
const imagePreview = ref(null);
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

const openImagePreview = (image) => {
  imagePreview.value = image;
  imagePreviewOpen.value = true;
};

// 新消息和流式增量到达时保持视图停留在最新内容。
watch(
  () => props.messages
    .map((message) => `${message.local_id || message.message_id}:${message.content}:${message.progress?.message || ""}`)
    .join("|"),
  async () => {
    await nextTick();
    if (messageList.value) {
      messageList.value.scrollTop = messageList.value.scrollHeight;
    }
  },
);
</script>

<style scoped>
.chat-page {
  background: #fbfbfa;
  display: grid;
  grid-template-rows: 52px minmax(0, 1fr) auto;
  height: 100%;
  min-height: 0;
}

.topbar {
  align-items: center;
  background: rgba(251, 251, 250, 0.94);
  border-bottom: 1px solid #ededeb;
  display: flex;
  justify-content: space-between;
  padding: 0 max(24px, calc((100% - 800px) / 2));
}

h1 {
  color: #555552;
  font-size: 14px;
  font-weight: 600;
  letter-spacing: 0;
  margin: 0;
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

.messages {
  background: linear-gradient(180deg, #fbfbfa 0%, #f6f9fc 60%, #fbfbfa 100%);
  min-height: 0;
  overflow-y: auto;
  padding: 30px max(24px, calc((100% - 800px) / 2)) 44px;
  scrollbar-color: #d8d8d4 transparent;
  scrollbar-width: thin;
}

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

@media (max-width: 760px) {
  .chat-page {
    grid-template-rows: 48px minmax(0, 1fr) auto;
  }

  .topbar,
  .messages {
    padding-left: 16px;
    padding-right: 16px;
  }

  .messages {
    padding-bottom: 28px;
  }
}
</style>
