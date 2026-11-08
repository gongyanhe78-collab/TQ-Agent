<template>
  <main class="login-page">
    <section class="login-panel" aria-labelledby="login-title">
      <div class="weather-decor" aria-hidden="true">🌦️ 🌤️ ⛅ 🌧️</div>
      <div class="brand-mark" aria-hidden="true">气象</div>
      <div class="login-heading">
        <h1 id="login-title">气象灾害个例问答助手</h1>
        <p>使用本地账号登录</p>
      </div>

      <div v-if="notice || errorMessage" class="login-notice" role="alert">
        {{ errorMessage || notice }}
      </div>

      <form class="login-form" @submit.prevent="submit">
        <label class="field-label" for="username">用户名</label>
        <el-input
          id="username"
          v-model="username"
          :disabled="loading"
          :prefix-icon="User"
          autocomplete="username"
          placeholder="请输入用户名"
          size="large"
          @input="validationError = ''"
        />

        <label class="field-label" for="password">密码</label>
        <el-input
          id="password"
          v-model="password"
          :disabled="loading"
          :prefix-icon="Lock"
          autocomplete="current-password"
          placeholder="请输入密码"
          show-password
          size="large"
          type="password"
          @input="validationError = ''"
        />

        <el-button
          class="login-button"
          :loading="loading"
          native-type="submit"
          size="large"
          type="primary"
        >
          登录
        </el-button>
      </form>
    </section>
  </main>
</template>

<script setup>
import { computed, ref } from "vue";
import { Lock, User } from "@element-plus/icons-vue";

const props = defineProps({
  loading: {
    type: Boolean,
    default: false,
  },
  error: {
    type: String,
    default: "",
  },
  notice: {
    type: String,
    default: "",
  },
});

const emit = defineEmits(["submit"]);
const username = ref("");
const password = ref("");
const validationError = ref("");
const errorMessage = computed(() => validationError.value || props.error);

const submit = () => {
  if (props.loading) return;
  const normalizedUsername = username.value.trim();
  if (!normalizedUsername || !password.value) {
    validationError.value = "请输入用户名和密码";
    return;
  }
  emit("submit", {
    username: normalizedUsername,
    password: password.value,
  });
};
</script>

<style scoped>
.login-page {
  align-items: center;
  background: linear-gradient(180deg, #fbfbfa 0%, #f4f8fb 100%);
  color: #292927;
  display: flex;
  justify-content: center;
  min-height: 100%;
  padding: 28px 20px;
}

.login-panel {
  background: #f4f4f2;
  border: 1px solid #deded9;
  border-radius: 8px;
  box-sizing: border-box;
  max-width: 380px;
  padding: 30px;
  width: 100%;
}

.weather-decor {
  color: #8f9aa8;
  font-size: 20px;
  letter-spacing: 5px;
  line-height: 1;
  margin-bottom: 14px;
  text-align: center;
}

.brand-mark {
  align-items: center;
  background: #343432;
  border-radius: 7px;
  color: #ffffff;
  display: flex;
  font-size: 13px;
  font-weight: 650;
  height: 38px;
  justify-content: center;
  letter-spacing: 1px;
  width: 38px;
}

.login-heading {
  margin: 18px 0 24px;
}

h1 {
  color: #292927;
  font-size: 20px;
  font-weight: 650;
  letter-spacing: 0;
  line-height: 1.35;
  margin: 0;
}

.login-heading p {
  color: #74746e;
  font-size: 13px;
  margin: 7px 0 0;
}

.login-notice {
  background: #fff7e8;
  border: 1px solid #ead8b4;
  border-radius: 6px;
  color: #76551e;
  font-size: 13px;
  line-height: 1.5;
  margin-bottom: 16px;
  padding: 9px 11px;
}

.login-form {
  display: grid;
  gap: 9px;
}

.field-label {
  color: #555550;
  font-size: 13px;
  font-weight: 600;
  margin-top: 3px;
}

.login-form :deep(.el-input__wrapper) {
  background: #ffffff;
  border-radius: 7px;
  box-shadow: 0 0 0 1px #d9d9d4 inset;
}

.login-form :deep(.el-input__wrapper.is-focus) {
  box-shadow: 0 0 0 1px #77776f inset;
}

.login-button {
  background: #2d2d2b;
  border-color: #2d2d2b;
  border-radius: 7px;
  margin-top: 12px;
  width: 100%;
}

.login-button:hover,
.login-button:focus {
  background: #454542;
  border-color: #454542;
}

@media (max-width: 480px) {
  .login-panel {
    padding: 24px 20px;
  }

  h1 {
    font-size: 18px;
  }
}
</style>
