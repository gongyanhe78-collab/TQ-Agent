import { computed, onBeforeUnmount, ref } from "vue";

import {
  clearAccessToken,
  getApiErrorMessage,
  loginSession,
  logoutSession,
  refreshSession,
  setAuthExpiredHandler,
} from "../../api";

// 认证状态只维护当前用户与恢复状态，业务数据清理由页面统一执行。
export const useAuthSession = ({ onExpired } = {}) => {
  const user = ref(null);
  const restoring = ref(true);
  const submitting = ref(false);
  const error = ref("");
  const expired = ref(false);

  const isAuthenticated = computed(() => Boolean(user.value));
  const isAdmin = computed(() => user.value?.role === "admin");

  const releaseExpiredHandler = setAuthExpiredHandler(() => {
    if (!user.value) return;
    user.value = null;
    expired.value = true;
    error.value = "登录已过期，请重新登录";
    onExpired?.();
  });

  const restore = async () => {
    restoring.value = true;
    error.value = "";
    try {
      const session = await refreshSession();
      user.value = session?.user || null;
      expired.value = false;
      return Boolean(user.value);
    } catch (restoreError) {
      user.value = null;
      // 首次访问没有刷新 Cookie 时后端返回 401，这属于未登录而非异常。
      if (restoreError?.response?.status !== 401) {
        error.value = getApiErrorMessage(restoreError, "无法恢复登录状态");
      }
      return false;
    } finally {
      restoring.value = false;
    }
  };

  const login = async (credentials) => {
    submitting.value = true;
    error.value = "";
    try {
      const session = await loginSession(credentials.username, credentials.password);
      user.value = session?.user || null;
      expired.value = false;
      return user.value;
    } catch (loginError) {
      error.value = getApiErrorMessage(loginError, "登录失败");
      throw loginError;
    } finally {
      submitting.value = false;
    }
  };

  const logout = async () => {
    try {
      await logoutSession();
    } finally {
      clearAccessToken();
      user.value = null;
      expired.value = false;
      error.value = "";
    }
  };

  onBeforeUnmount(releaseExpiredHandler);

  return {
    user,
    restoring,
    submitting,
    error,
    expired,
    isAuthenticated,
    isAdmin,
    restore,
    login,
    logout,
  };
};
