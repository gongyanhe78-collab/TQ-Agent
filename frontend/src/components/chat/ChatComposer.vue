<template>
  <form class="composer" @submit.prevent="submit">
    <el-input
      v-model="question"
      type="textarea"
      :rows="3"
      resize="none"
      placeholder="输入问题"
      @keydown.ctrl.enter.prevent="submit"
    />
    <el-button type="primary" native-type="submit" :icon="Promotion" :loading="loading">发送</el-button>
  </form>
</template>

<script setup>
import { ref } from "vue";
import { Promotion } from "@element-plus/icons-vue";

const props = defineProps({
  loading: {
    type: Boolean,
    default: false,
  },
});

const emit = defineEmits(["submit"]);
const question = ref("");

// 输入框只负责提交自然语言，具体使用哪条检索链路由后端决定。
const submit = () => {
  const text = question.value.trim();
  if (!text || props.loading) return;
  emit("submit", text);
  question.value = "";
};

// 供 ChatPanel 的空态示例问题把纯文字填入输入框，emoji 只作按钮装饰。
const setQuestion = (text) => {
  question.value = String(text || "");
};
defineExpose({ setQuestion });
</script>

<style scoped>
.composer {
  align-items: flex-end;
  background: linear-gradient(to top, #fbfbfa 82%, rgba(251, 251, 250, 0));
  display: grid;
  gap: 10px;
  grid-template-columns: minmax(0, 1fr) auto;
  padding: 20px max(24px, calc((100% - 800px) / 2)) 22px;
}

.composer :deep(.el-textarea__inner) {
  background: #ffffff;
  border-radius: 16px;
  box-shadow: 0 0 0 1px #deded9 inset, 0 5px 18px rgba(35, 35, 32, 0.06);
  color: #292927;
  font-family: inherit;
  min-height: 78px !important;
  padding: 13px 15px;
}

.composer :deep(.el-textarea) {
  min-width: 0;
}

.composer :deep(.el-textarea__inner:focus) {
  box-shadow: 0 0 0 1px #999991 inset, 0 5px 18px rgba(35, 35, 32, 0.06);
}

.composer :deep(.el-button--primary) {
  background: #2d2d2b;
  border-color: #2d2d2b;
  border-radius: 14px;
  height: 44px;
  min-width: 44px;
}

.composer :deep(.el-button--primary:hover) {
  background: #454542;
  border-color: #454542;
}

@media (max-width: 760px) {
  .composer {
    grid-template-columns: minmax(0, 1fr) 44px;
    padding: 20px 16px 14px;
  }

  /* 手机端只保留发送图标，确保按钮始终位于输入框右侧且不会溢出屏幕。 */
  .composer :deep(.el-button--primary) {
    font-size: 0;
    min-width: 44px;
    padding: 0;
    width: 44px;
  }

  .composer :deep(.el-button--primary .el-icon) {
    font-size: 16px;
    margin-right: 0;
  }
}
</style>
