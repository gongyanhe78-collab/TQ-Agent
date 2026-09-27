<template>
  <section class="processing-status" aria-live="polite" aria-label="回答处理进度">
    <div
      v-for="(step, index) in visibleSteps"
      :key="step.label"
      class="processing-step"
      :class="stepState(index)"
    >
      <span class="processing-icon" aria-hidden="true">
        <el-icon v-if="stepState(index) === 'active'" class="processing-spinner">
          <Loading />
        </el-icon>
        <el-icon v-else-if="stepState(index) === 'completed'" class="processing-check">
          <CircleCheck />
        </el-icon>
        <el-icon v-else>
          <component :is="step.icon" />
        </el-icon>
      </span>
      <span class="processing-label">{{ step.label }}</span>
    </div>
  </section>
</template>

<script setup>
import { computed } from "vue";
import { CircleCheck, Compass, Loading, Opportunity, Search } from "@element-plus/icons-vue";

const props = defineProps({
  progress: {
    type: Object,
    default: () => ({}),
  },
});

// 五个阶段保持稳定顺序，避免后端不同 Agent 的细粒度 stage 让页面跳来跳去。
const steps = [
  { label: "正在理解问题", icon: Search },
  { label: "正在检索相关证据", icon: Compass },
  { label: "正在整理多个灾害过程", icon: Opportunity },
  { label: "正在等待回答模型", icon: Opportunity },
  { label: "已开始输出", icon: Opportunity },
];

const normalizedMessage = computed(() => String(props.progress?.message || ""));
const currentIndex = computed(() => {
  const stage = String(props.progress?.stage || "understanding").toLowerCase();
  const message = normalizedMessage.value;

  if (stage === "streaming" || stage === "persist" || message.includes("已开始")) return 4;
  if (
    message.includes("整理")
    || message.includes("分析个例")
    || message.includes("整理结构化")
    || stage === "analysis"
    || stage === "pdf_export"
  ) return 2;
  if (message.includes("等待模型") || message.includes("首段回答") || stage === "generation") return 3;
  if (
    stage === "retrieval"
    || stage === "retrieved"
    || stage === "structured_search"
    || stage === "condition_parse"
    || stage === "search"
    || message.includes("检索")
  ) return 1;
  return 0;
});

// 只渲染已经到达的阶段，避免用户提前看到尚未开始的处理步骤。
const visibleSteps = computed(() => steps.slice(0, currentIndex.value + 1));

const stepState = (index) => {
  if (index < currentIndex.value) return "completed";
  if (index === currentIndex.value) return "active";
  return "pending";
};
</script>

<style scoped>
.processing-status {
  display: grid;
  gap: 10px;
  margin: 2px 0 18px;
  max-width: 420px;
  padding: 2px 0;
}

.processing-step {
  align-items: center;
  color: #a1a19c;
  display: flex;
  font-size: 13px;
  gap: 10px;
  line-height: 1.35;
  min-height: 18px;
  transition: color 160ms ease, opacity 160ms ease;
}

.processing-step.active {
  color: #696963;
}

.processing-step.completed {
  color: #85857f;
}

.processing-step.pending {
  opacity: 0.62;
}

.processing-icon {
  align-items: center;
  color: #8d8d87;
  display: inline-flex;
  flex: 0 0 15px;
  height: 15px;
  justify-content: center;
  width: 15px;
}

.processing-step.active .processing-icon {
  color: #70706a;
}

.processing-step.completed .processing-icon {
  color: #9b9b95;
}

.processing-spinner {
  animation: processing-spin 1s linear infinite;
}

.processing-check {
  font-size: 14px;
}

.processing-label {
  white-space: nowrap;
}

@keyframes processing-spin {
  to {
    transform: rotate(360deg);
  }
}

@media (prefers-reduced-motion: reduce) {
  .processing-spinner {
    animation: none;
  }
}
</style>
