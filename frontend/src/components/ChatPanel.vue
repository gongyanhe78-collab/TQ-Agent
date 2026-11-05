<template>
  <section class="chat-page">
    <header class="topbar">
      <h1>气象灾害个例问答助手</h1>
    </header>

    <div ref="messageList" class="messages">
      <ChatMessages
        :messages="messages"
        :session-id="sessionId"
        @preview-image="openImagePreview"
        @regenerate="emit('regenerate', $event)"
        @select-version="emit('select-version', $event)"
      />
    </div>

    <ChatComposer :loading="loading" @submit="emit('query', $event)" />

    <EvidencePreview
      v-model:open="imagePreviewOpen"
      :image="imagePreview"
    />
  </section>
</template>

<script setup>
import { nextTick, ref, watch } from "vue";
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
const messageList = ref(null);
const imagePreviewOpen = ref(false);
const imagePreview = ref(null);

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
  padding: 0 max(24px, calc((100% - 800px) / 2));
}

h1 {
  color: #555552;
  font-size: 14px;
  font-weight: 600;
  letter-spacing: 0;
  margin: 0;
}

.messages {
  min-height: 0;
  overflow-y: auto;
  padding: 30px max(24px, calc((100% - 800px) / 2)) 44px;
  scrollbar-color: #d8d8d4 transparent;
  scrollbar-width: thin;
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
