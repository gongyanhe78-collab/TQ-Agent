<template>
  <aside class="session-panel" :class="{ collapsed }">
    <header class="session-header">
      <h2 v-if="!collapsed">聊天记录</h2>
      <el-button
        class="collapse-button"
        text
        :icon="collapsed ? Expand : Fold"
        :aria-label="collapsed ? '展开聊天记录' : '收起聊天记录'"
        :title="collapsed ? '展开聊天记录' : '收起聊天记录'"
        @click="$emit('toggle-collapse')"
      />
      <el-button v-if="!collapsed" class="new-session-button" :icon="Plus" :loading="loading" @click="$emit('new-session')">
        新建
      </el-button>
    </header>

    <el-scrollbar v-if="!collapsed" class="session-scroll">
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
import { Expand, Fold, Plus } from "@element-plus/icons-vue";

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
  collapsed: {
    type: Boolean,
    default: false,
  },
});

defineEmits(["new-session", "select-session", "toggle-collapse"]);

// 使用紧凑时间格式，保证窄栏中的聊天记录易于浏览。
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
  background: #f3f3f1;
  border-right: 1px solid #e4e4e0;
  display: flex;
  flex-direction: column;
  box-sizing: border-box;
  height: 100%;
  min-height: 0;
  overflow: hidden;
  padding: 15px 12px;
}

.session-panel.collapsed {
  align-items: center;
  padding: 12px 6px;
  width: 56px;
}

@media (max-width: 760px) {
  .session-panel.collapsed {
    border-bottom: 1px solid #dce1e7;
    border-right: 0;
    height: 56px;
    width: 100%;
  }
}

.session-header {
  align-items: flex-start;
  display: flex;
  gap: 12px;
  justify-content: space-between;
}

.session-panel.collapsed .session-header {
  align-items: center;
  display: flex;
  justify-content: center;
  width: 100%;
}

.collapse-button {
  color: #5f5f5a;
  flex: 0 0 auto;
}

h2 {
  color: #30302e;
  font-size: 16px;
  font-weight: 600;
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
  border-radius: 7px;
  color: #343431;
  cursor: pointer;
  display: grid;
  gap: 4px;
  margin-bottom: 4px;
  padding: 9px 10px;
  text-align: left;
  width: 100%;
}

.session-item:hover {
  background: #e9e9e6;
}

.session-item.active {
  background: #e4e4e0;
  border-color: #d9d9d4;
}

.session-title {
  font-size: 14px;
  font-weight: 600;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.session-time {
  color: #85857f;
  font-size: 12px;
}

.new-session-button {
  background: #ffffff;
  border-color: #d9d9d4;
  color: #343431;
}

.new-session-button:hover,
.new-session-button:focus {
  background: #e9e9e6;
  border-color: #c8c8c2;
  color: #20201e;
}
</style>
