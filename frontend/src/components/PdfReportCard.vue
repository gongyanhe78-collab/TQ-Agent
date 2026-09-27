<template>
  <section class="pdf-card">
    <header class="pdf-head">
      <span class="pdf-icon">PDF</span>
      <div class="pdf-title">
        <strong>{{ report.title || "气象灾害个例分析报告" }}</strong>
        <span>{{ report.filename }}</span>
      </div>
      <el-button :icon="Download" circle title="下载 PDF" @click.stop="download" />
    </header>
    <div class="pdf-thumbnail">
      <img
        v-if="!thumbnailError"
        :src="thumbnailUrl"
        alt="PDF 报告第一页缩略图"
        @error="thumbnailError = true"
      />
      <div v-else class="pdf-placeholder">PDF 报告已生成</div>
      <button
        class="pdf-thumbnail-action"
        type="button"
        title="查看完整报告"
        :disabled="!reportUrl"
        @click="openPreview"
      >
        <span>查看完整报告</span>
      </button>
    </div>
  </section>

  <el-dialog v-model="previewOpen" class="pdf-dialog" width="min(1100px, 94vw)" top="3vh">
    <template #header>
      <strong>{{ report.title || report.filename }}</strong>
    </template>
    <iframe
      v-if="previewUrl"
      class="pdf-frame"
      :src="`${previewUrl}#toolbar=1&navpanes=0&view=FitH`"
      title="PDF 报告预览"
    />
    <div v-else class="pdf-dialog-placeholder">{{ previewError || (previewLoading ? "正在加载完整报告..." : "报告预览尚未加载") }}</div>
  </el-dialog>
</template>

<script setup>
import { computed, onBeforeUnmount, ref, watch } from "vue";
import { Download } from "@element-plus/icons-vue";
import { API_ORIGIN } from "../api";

const props = defineProps({
  report: { type: Object, required: true },
});

const previewOpen = ref(false);
const previewUrl = ref("");
const previewError = ref("");
const previewLoading = ref(false);
const thumbnailError = ref(false);
let previewRequestId = 0;

const reportUrl = computed(() => {
  const value = props.report.url || "";
  if (/^https?:\/\//i.test(value)) return value;
  return `${API_ORIGIN}${value.startsWith("/") ? value : `/${value}`}`;
});

const thumbnailUrl = computed(() => {
  const explicit = props.report.thumbnail_url || "";
  if (explicit) return /^https?:\/\//i.test(explicit) ? explicit : `${API_ORIGIN}${explicit.startsWith("/") ? explicit : `/${explicit}`}`;
  const filename = String(props.report.filename || "").replace(/\.pdf$/i, ".thumbnail.png");
  return filename ? `${API_ORIGIN}/api/case-multidim/report-thumbnails/${encodeURIComponent(filename)}` : "";
});

// 释放浏览器为 PDF 预览创建的临时地址，避免切换会话后持续占用内存。
const releasePreviewUrl = () => {
  if (!previewUrl.value) return;
  URL.revokeObjectURL(previewUrl.value);
  previewUrl.value = "";
};

// 报告切换时只重置状态，完整 PDF 在用户点击后按需加载。
watch(
  reportUrl,
  () => {
    previewRequestId += 1;
    previewOpen.value = false;
    previewError.value = "";
    previewLoading.value = false;
    thumbnailError.value = false;
    releasePreviewUrl();
  },
  { immediate: true },
);

// 只有用户点击查看时才加载完整 PDF，列表缩略图始终使用稳定的 PNG 地址。
const openPreview = async () => {
  if (!reportUrl.value) return;
  previewOpen.value = true;
  if (previewUrl.value || previewLoading.value) return;
  const requestId = ++previewRequestId;
  previewError.value = "";
  previewLoading.value = true;
  try {
    // 报告接口按附件返回，转成 Blob 后在弹窗预览，不会误触发浏览器下载。
    const response = await fetch(reportUrl.value);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const blob = await response.blob();
    if (requestId !== previewRequestId) return;
    previewUrl.value = URL.createObjectURL(blob);
  } catch (error) {
    if (requestId !== previewRequestId) return;
    previewError.value = `报告预览加载失败：${error instanceof Error ? error.message : "未知错误"}`;
  } finally {
    if (requestId === previewRequestId) previewLoading.value = false;
  }
};

onBeforeUnmount(() => {
  previewRequestId += 1;
  releasePreviewUrl();
});

// 使用浏览器原生下载能力，文件内容与缩略预览保持一致。
const download = () => {
  const link = document.createElement("a");
  link.href = reportUrl.value;
  link.download = props.report.filename || "case-report.pdf";
  link.click();
};
</script>

<style scoped>
.pdf-card { background: #fff; border: 1px solid #deded9; border-radius: 7px; margin-top: 14px; overflow: hidden; }
.pdf-head { align-items: center; border-bottom: 1px solid #e4e4e0; display: flex; gap: 10px; min-height: 58px; padding: 10px 12px; }
.pdf-icon { background: #b42318; border-radius: 4px; color: #fff; display: grid; flex: none; font-size: 11px; font-weight: 700; height: 34px; place-items: center; width: 34px; }
.pdf-title { display: grid; flex: 1; min-width: 0; }
.pdf-title strong, .pdf-title span { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.pdf-title strong { color: #292927; font-size: 14px; }
.pdf-title span { color: #777771; font-size: 11px; }
.pdf-thumbnail { background: #ecece8; height: 310px; overflow: hidden; position: relative; width: 100%; }
.pdf-thumbnail img { display: block; height: 100%; object-fit: contain; object-position: top center; width: 100%; }
.pdf-placeholder { align-items: center; color: #777771; display: flex; height: 100%; justify-content: center; padding: 20px; text-align: center; }
.pdf-thumbnail-action { background: transparent; border: 0; cursor: pointer; inset: 0; padding: 0; position: absolute; width: 100%; }
.pdf-thumbnail-action:disabled { cursor: default; }
.pdf-thumbnail-action span { background: rgba(45, 45, 43, .94); bottom: 12px; color: #fff; font-size: 12px; left: 50%; padding: 7px 12px; position: absolute; transform: translateX(-50%); }
.pdf-thumbnail-action:disabled span { display: none; }
.pdf-frame { border: 0; height: 82vh; width: 100%; }
.pdf-dialog-placeholder { align-items: center; color: #777771; display: flex; height: 60vh; justify-content: center; padding: 20px; }
:global(.pdf-dialog .el-dialog__body) { padding: 0; }
@media (max-width: 760px) { .pdf-thumbnail { height: 240px; } .pdf-frame { height: 78vh; } }
</style>
