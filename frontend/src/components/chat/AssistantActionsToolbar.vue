<template>
  <div class="message-actions">
    <div
      v-if="message.version_navigation?.total > 1"
      class="version-switcher"
      aria-label="回答版本切换"
    >
      <button
        class="version-button"
        type="button"
        aria-label="查看上一版本"
        :disabled="!message.version_navigation.can_previous"
        @click="selectVersion(-1)"
      >
        <el-icon><ArrowLeft /></el-icon>
      </button>
      <span class="version-label"
        >{{ message.version_navigation.current }} / {{ message.version_navigation.total }}</span
      >
      <button
        class="version-button"
        type="button"
        aria-label="查看下一版本"
        :disabled="!message.version_navigation.can_next"
        @click="selectVersion(1)"
      >
        <el-icon><ArrowRight /></el-icon>
      </button>
    </div>

    <el-tooltip content="复制响应" placement="bottom">
      <button class="action-button" type="button" aria-label="复制响应" @click="emit('copy')">
        <el-icon><CopyDocument /></el-icon>
      </button>
    </el-tooltip>
    <el-tooltip content="创建共享链接" placement="bottom">
      <button class="action-button" type="button" aria-label="创建共享链接" @click="emit('share')">
        <el-icon><Share /></el-icon>
      </button>
    </el-tooltip>
    <el-tooltip content="喜欢这个回答" placement="bottom">
      <button
        class="action-button"
        :class="{ active: feedback === 'love' }"
        type="button"
        aria-label="喜欢这个回答"
        :aria-pressed="feedback === 'love'"
        @click="emit('toggle-love')"
      >
        <el-icon><Star /></el-icon>
      </button>
    </el-tooltip>
    <el-tooltip content="需要改进" placement="bottom">
      <button
        class="action-button"
        :class="{ active: feedback === 'needs_improvement' }"
        type="button"
        aria-label="需要改进"
        :aria-pressed="feedback === 'needs_improvement'"
        @click="emit('improve')"
      >
        <el-icon><Warning /></el-icon>
      </button>
    </el-tooltip>
    <el-tooltip content="重新生成" placement="bottom">
      <button
        class="action-button"
        type="button"
        aria-label="重新生成"
        @click="emit('regenerate', message)"
      >
        <el-icon><RefreshRight /></el-icon>
      </button>
    </el-tooltip>

    <el-dropdown trigger="click" @command="emit('more-command', $event)">
      <button class="action-button" type="button" aria-label="更多操作">
        <el-icon><MoreFilled /></el-icon>
      </button>
      <template #dropdown>
        <el-dropdown-menu>
          <el-dropdown-item command="issue">
            <el-icon><ChatDotSquare /></el-icon>报告问题
          </el-dropdown-item>
          <el-dropdown-item command="pdf" :disabled="pdfExporting">
            <el-icon><Download /></el-icon>{{ pdfExporting ? "正在导出" : "导出到 PDF" }}
          </el-dropdown-item>
        </el-dropdown-menu>
      </template>
    </el-dropdown>
  </div>
</template>

<script setup>
import {
  ArrowLeft,
  ArrowRight,
  ChatDotSquare,
  CopyDocument,
  Download,
  MoreFilled,
  RefreshRight,
  Share,
  Star,
  Warning,
} from "@element-plus/icons-vue";

defineProps({
  message: {
    type: Object,
    required: true,
  },
  feedback: {
    type: String,
    default: "none",
  },
  pdfExporting: {
    type: Boolean,
    default: false,
  },
});

const emit = defineEmits([
  "copy",
  "share",
  "toggle-love",
  "improve",
  "regenerate",
  "select-version",
  "more-command",
]);

const selectVersion = (offset) => emit("select-version", offset);
</script>

<style scoped>
.message-actions {
  align-items: center;
  display: flex;
  gap: 2px;
  margin-top: 12px;
  min-height: 30px;
}

.action-button {
  align-items: center;
  background: transparent;
  border: 0;
  border-radius: 5px;
  color: #777771;
  cursor: pointer;
  display: inline-flex;
  height: 30px;
  justify-content: center;
  padding: 0;
  width: 30px;
}

.action-button:hover {
  background: #ecece8;
  color: #292927;
}

.action-button.active {
  background: #292927;
  color: #ffffff;
  box-shadow: inset 0 0 0 1px #292927;
}

.action-button.active:hover {
  background: #171715;
  color: #ffffff;
}

.version-switcher {
  align-items: center;
  display: inline-flex;
  gap: 2px;
  margin-right: 6px;
}

.version-button {
  align-items: center;
  background: transparent;
  border: 0;
  border-radius: 4px;
  color: #555552;
  cursor: pointer;
  display: inline-flex;
  height: 26px;
  justify-content: center;
  padding: 0;
  width: 26px;
}

.version-button:hover:not(:disabled) {
  background: #ecece8;
}

.version-button:disabled {
  color: #c5c5c0;
  cursor: default;
}

.version-label {
  color: #777771;
  font-size: 11px;
  min-width: 34px;
  text-align: center;
}
</style>
