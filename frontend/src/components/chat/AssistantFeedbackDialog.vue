<template>
  <el-dialog
    v-model="visible"
    title="这条回答哪里需要改进？"
    width="min(460px, calc(100vw - 32px))"
    :close-on-click-modal="false"
    append-to-body
  >
    <el-radio-group v-model="category" class="feedback-reasons">
      <el-radio-button
        v-for="option in improvementOptions"
        :key="option.value"
        :value="option.value"
      >
        {{ option.label }}
      </el-radio-button>
    </el-radio-group>
    <el-input
      v-model="description"
      type="textarea"
      :rows="3"
      maxlength="300"
      show-word-limit
      placeholder="可补充具体问题，例如遗漏了哪项事实或哪部分证据不匹配"
      class="feedback-description"
    />
    <template #footer>
      <div class="feedback-dialog-actions">
        <el-button @click="visible = false">取消</el-button>
        <el-button :loading="submitting" @click="emit('submit', false)"> 仅提交反馈 </el-button>
        <el-button type="primary" :loading="submitting" @click="emit('submit', true)">
          按此原因重新生成
        </el-button>
      </div>
    </template>
  </el-dialog>
</template>

<script setup>
const visible = defineModel("visible", { type: Boolean, default: false });
const category = defineModel("category", { type: String, default: "off_topic" });
const description = defineModel("description", { type: String, default: "" });

defineProps({
  submitting: {
    type: Boolean,
    default: false,
  },
});

const emit = defineEmits(["submit"]);

const improvementOptions = [
  { value: "off_topic", label: "答非所问" },
  { value: "fact_error", label: "事实错误" },
  { value: "missing_information", label: "遗漏关键信息" },
  { value: "evidence_mismatch", label: "证据不匹配" },
  { value: "layout", label: "排版问题" },
  { value: "other", label: "其他" },
];
</script>

<style scoped>
.feedback-reasons {
  display: grid;
  gap: 8px;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  width: 100%;
}

.feedback-reasons :deep(.el-radio-button),
.feedback-reasons :deep(.el-radio-button__inner) {
  width: 100%;
}

.feedback-description {
  margin-top: 16px;
}

.feedback-dialog-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  justify-content: flex-end;
}

@media (max-width: 520px) {
  .feedback-dialog-actions {
    align-items: stretch;
    flex-direction: column-reverse;
  }

  .feedback-dialog-actions .el-button {
    margin-left: 0;
    width: 100%;
  }
}
</style>
