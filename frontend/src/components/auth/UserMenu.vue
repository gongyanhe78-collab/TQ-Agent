<template>
  <el-dropdown trigger="click" placement="bottom-end" @command="handleCommand">
    <button class="user-trigger" type="button" aria-label="打开账号菜单">
      <span class="user-avatar" aria-hidden="true">{{ initial }}</span>
      <span class="user-name">{{ username }}</span>
      <el-icon class="user-arrow" aria-hidden="true"><ArrowDown /></el-icon>
    </button>
    <template #dropdown>
      <el-dropdown-menu>
        <el-dropdown-item
          v-if="isAdmin"
          command="rebuild"
          :disabled="rebuilding"
          :icon="rebuilding ? Loading : Refresh"
        >
          {{ rebuilding ? "正在重建知识库" : "重建知识库" }}
        </el-dropdown-item>
        <el-dropdown-item command="logout" :icon="SwitchButton" :divided="isAdmin">
          退出登录
        </el-dropdown-item>
      </el-dropdown-menu>
    </template>
  </el-dropdown>
</template>

<script setup>
import { computed, ref } from "vue";
import { ElMessage, ElMessageBox } from "element-plus";
import { ArrowDown, Loading, Refresh, SwitchButton } from "@element-plus/icons-vue";

import { getApiErrorMessage, rebuildKnowledgeBase } from "../../api";

const props = defineProps({
  user: {
    type: Object,
    default: () => ({}),
  },
});

const emit = defineEmits(["logout"]);
const rebuilding = ref(false);
const username = computed(() => String(props.user?.username || "当前用户"));
const isAdmin = computed(() => props.user?.role === "admin");
const initial = computed(() => username.value.slice(0, 1).toUpperCase());

const rebuildKnowledge = async () => {
  try {
    await ElMessageBox.confirm(
      "重建会刷新现有文档向量和图片证据，需要管理员权限。是否继续？",
      "重建知识库",
      {
        confirmButtonText: "开始重建",
        cancelButtonText: "取消",
        type: "warning",
      }
    );
  } catch {
    return;
  }

  rebuilding.value = true;
  try {
    await rebuildKnowledgeBase();
    ElMessage.success("知识库重建完成");
  } catch (error) {
    ElMessage.error(`知识库重建失败：${getApiErrorMessage(error)}`);
  } finally {
    rebuilding.value = false;
  }
};

const handleCommand = (command) => {
  if (command === "rebuild") rebuildKnowledge();
  if (command === "logout") emit("logout");
};
</script>

<style scoped>
.user-trigger {
  align-items: center;
  background: rgba(251, 251, 250, 0.94);
  border: 1px solid #deded9;
  border-radius: 7px;
  color: #3b3b38;
  cursor: pointer;
  display: flex;
  gap: 8px;
  height: 34px;
  max-width: 190px;
  padding: 3px 8px 3px 4px;
}

.user-trigger:hover,
.user-trigger:focus-visible {
  background: #f0f0ed;
  border-color: #cbcbc5;
  outline: none;
}

.user-avatar {
  align-items: center;
  background: #343432;
  border-radius: 5px;
  color: #ffffff;
  display: flex;
  flex: 0 0 26px;
  font-size: 12px;
  font-weight: 650;
  height: 26px;
  justify-content: center;
}

.user-name {
  font-size: 13px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.user-arrow {
  color: #85857f;
  flex: 0 0 auto;
  font-size: 12px;
}

@media (max-width: 760px) {
  .user-trigger {
    max-width: 44px;
    padding-right: 7px;
  }

  .user-name,
  .user-arrow {
    display: none;
  }
}
</style>
