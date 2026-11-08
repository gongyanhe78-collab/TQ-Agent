// 让现有前端在容器部署时复用当前站点的 Nginx 同源代理。
try {
  window.localStorage.setItem("risk-api-origin", window.location.origin);
} catch (error) {
  console.warn("无法写入本地 API 地址配置，将继续使用前端默认值。", error);
}
