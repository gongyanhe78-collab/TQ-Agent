<template>
  <section class="case-panel">
    <header class="panel-header">
      <div>
        <p class="eyebrow">ChromaDB</p>
        <h1>向量节点</h1>
      </div>
      <el-button circle :icon="Fold" @click="$emit('toggle-collapse')" />
    </header>

    <div class="stats">
      <span>{{ caseCount }} 个 key</span>
      <span>collection: weather_document_chunks</span>
    </div>

    <div class="actions">
      <el-button :icon="Refresh" :loading="loading" @click="$emit('refresh-library')">刷新</el-button>
      <el-button type="primary" :icon="DocumentAdd" :loading="loading" @click="openFilePicker">
        批量上传
      </el-button>
      <input
        ref="fileInput"
        class="file-input"
        type="file"
        accept="application/pdf,.pdf"
        multiple
        @change="handleFileChange"
      />
    </div>

    <el-input v-model="keyword" class="search" placeholder="筛选 key" clearable :prefix-icon="Search" />

    <el-scrollbar class="nodes">
      <button
        v-for="caseId in filteredCaseIds"
        :key="caseId"
        class="node"
        :class="{ active: caseId === activeCaseId }"
        @click="$emit('select', caseId)"
      >
        <span class="dot" />
        <span class="key">{{ caseId }}</span>
      </button>
      <el-empty v-if="!filteredCaseIds.length" description="暂无向量节点" />
    </el-scrollbar>
  </section>
</template>

<script setup>
import { computed, ref } from "vue";
import { DocumentAdd, Fold, Refresh, Search } from "@element-plus/icons-vue";

const props = defineProps({
  caseIds: {
    type: Array,
    default: () => [],
  },
  activeCaseId: {
    type: String,
    default: "",
  },
  loading: {
    type: Boolean,
    default: false,
  },
  caseCount: {
    type: Number,
    default: 0,
  },
});

const emit = defineEmits(["select", "refresh-library", "upload-material", "toggle-collapse"]);

const keyword = ref("");
const fileInput = ref(null);
const filteredCaseIds = computed(() => {
  const value = keyword.value.trim().toLowerCase();
  if (!value) return props.caseIds;
  return props.caseIds.filter((caseId) => caseId.toLowerCase().includes(value));
});

// Open the hidden file picker so the user can choose one or many PDFs.
const openFilePicker = () => {
  fileInput.value?.click();
};

// Pass selected PDF files upward; App.vue performs upload, extraction, and indexing.
const handleFileChange = (event) => {
  const files = Array.from(event.target.files || []);
  event.target.value = "";
  if (!files.length) return;
  emit("upload-material", files);
};
</script>

<style scoped>
.case-panel {
  display: flex;
  flex-direction: column;
  height: 100%;
  min-height: 0;
  overflow: hidden;
  padding: 18px;
}

.panel-header {
  align-items: flex-start;
  display: flex;
  justify-content: space-between;
}

.eyebrow {
  color: #607083;
  font-size: 12px;
  margin: 0 0 4px;
  text-transform: uppercase;
}

h1 {
  font-size: 22px;
  font-weight: 650;
  line-height: 1.2;
  margin: 0;
}

.stats {
  color: #607083;
  display: flex;
  flex-direction: column;
  font-size: 13px;
  gap: 4px;
  margin-top: 14px;
}

.actions {
  display: grid;
  gap: 10px;
  grid-template-columns: 1fr 1fr;
  margin-top: 18px;
}

.file-input {
  display: none;
}

.search {
  margin-top: 14px;
}

.nodes {
  flex: 1;
  margin-top: 14px;
  min-height: 0;
  overflow: hidden;
}

.node {
  align-items: center;
  background: transparent;
  border: 1px solid transparent;
  border-radius: 6px;
  color: #243242;
  cursor: pointer;
  display: flex;
  gap: 10px;
  margin-bottom: 6px;
  padding: 10px;
  text-align: left;
  width: 100%;
}

.node:hover {
  background: #f1f4f7;
}

.node.active {
  background: #eaf2ff;
  border-color: #9ec5fe;
  color: #0b4f9c;
}

.dot {
  background: #39a86b;
  border-radius: 50%;
  flex: 0 0 8px;
  height: 8px;
  width: 8px;
}

.key {
  font-family: Consolas, "Courier New", monospace;
  font-size: 13px;
  line-height: 1.35;
  overflow-wrap: anywhere;
}
</style>
