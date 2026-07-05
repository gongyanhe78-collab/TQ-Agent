<template>
  <aside class="session-panel">
    <header class="session-header">
      <div>
        <p class="eyebrow">Sessions</p>
        <h2>聊天记录</h2>
      </div>
      <el-button type="primary" :icon="Plus" :loading="loading" @click="$emit('new-session')">
        新建
      </el-button>
    </header>

    <el-scrollbar class="session-scroll">
      <button
        v-for="session in sessions"
        :key="session.session_id"
        class="session-item"
        :class="{ active: session.session_id === activeSessionId }"
        @click="$emit('select-session', session.session_id)"
      >
        <span class="session-title">{{ session.title || "新会话" }}</span>
        <span class="session-time">{{ formatTime(session.updated_at) }}</span>
      </button>
      <el-empty v-if="!sessions.length" description="暂无聊天记录" />
    </el-scrollbar>
  </aside>
</template>

<script setup>
import { Plus } from "@element-plus/icons-vue";

defineProps({
  sessions: {
    type: Array,
    default: () => [],
  },
  activeSessionId: {
    type: String,
    default: "",
  },
  loading: {
    type: Boolean,
    default: false,
  },
});

defineEmits(["new-session", "select-session"]);

// Render a compact timestamp so the session list stays readable in a narrow column.
const formatTime = (value) => {
  if (!value) return "";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "";
  return date.toLocaleString("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
};
</script>

<style scoped>
.session-panel {
  background: #fbfcfd;
  border-right: 1px solid #dce1e7;
  display: flex;
  flex-direction: column;
  height: 100vh;
  min-height: 0;
  overflow: hidden;
  padding: 18px 14px;
}

.session-header {
  align-items: flex-start;
  display: flex;
  gap: 12px;
  justify-content: space-between;
}

.eyebrow {
  color: #607083;
  font-size: 12px;
  margin: 0 0 4px;
  text-transform: uppercase;
}

h2 {
  color: #17202a;
  font-size: 20px;
  font-weight: 650;
  line-height: 1.2;
  margin: 0;
}

.session-scroll {
  flex: 1;
  margin-top: 16px;
  min-height: 0;
  overflow: hidden;
}

.session-item {
  background: transparent;
  border: 1px solid transparent;
  border-radius: 8px;
  color: #243242;
  cursor: pointer;
  display: grid;
  gap: 6px;
  margin-bottom: 8px;
  padding: 11px 10px;
  text-align: left;
  width: 100%;
}

.session-item:hover {
  background: #f1f4f7;
}

.session-item.active {
  background: #eaf2ff;
  border-color: #9ec5fe;
}

.session-title {
  font-size: 14px;
  font-weight: 600;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.session-time {
  color: #718096;
  font-size: 12px;
}
</style>
