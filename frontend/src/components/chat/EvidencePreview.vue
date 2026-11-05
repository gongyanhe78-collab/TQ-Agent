<template>
  <el-dialog
    :model-value="open"
    class="image-preview-dialog"
    title="图片预览"
    width="min(960px, 92vw)"
    top="5vh"
    append-to-body
    @update:model-value="emit('update:open', $event)"
  >
    <div v-if="image" class="image-preview-content">
      <img
        :key="image.image_id"
        :src="assetUrl(image.url)"
        :alt="image.caption || image.image_id"
        @error="hideBrokenImage"
      />
      <p v-if="image.caption">{{ image.caption }}</p>
    </div>
  </el-dialog>
</template>

<script setup>
import { API_ORIGIN } from "../../api";

defineProps({
  open: {
    type: Boolean,
    default: false,
  },
  image: {
    type: Object,
    default: null,
  },
});

const emit = defineEmits(["update:open"]);

const assetUrl = (url) => {
  if (!url || /^https?:\/\//i.test(url)) return url || "";
  return `${API_ORIGIN}${url.startsWith("/") ? url : `/${url}`}`;
};

// 图片资源失效时隐藏预览元素，证据按钮仍保留以便识别对应资源。
const hideBrokenImage = (event) => {
  event.currentTarget.style.display = "none";
};
</script>

<style scoped>
:global(.image-preview-dialog .image-preview-content) {
  display: grid;
  gap: 12px;
  justify-items: center;
}

:global(.image-preview-dialog .image-preview-content img) {
  max-height: 76vh;
  max-width: 100%;
  object-fit: contain;
}

:global(.image-preview-dialog .image-preview-content p) {
  color: #656560;
  margin: 0;
  text-align: center;
}
</style>
