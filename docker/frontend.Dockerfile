# 前端构建阶段使用锁定版本 Node，安装依赖后生成静态产物。
FROM public.ecr.aws/docker/library/node:22.11.0-alpine3.20 AS builder

WORKDIR /app

COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund

COPY frontend/ ./
RUN npm run build

# 运行阶段只保留 Nginx、静态文件和同源代理配置。
FROM public.ecr.aws/docker/library/nginx:1.27.3-alpine AS runtime

COPY docker/nginx.conf /etc/nginx/conf.d/default.conf
COPY docker/risk-runtime-config.js /usr/share/nginx/html/risk-runtime-config.js
COPY docker/frontend-entrypoint.sh /usr/local/bin/frontend-entrypoint.sh
COPY --from=builder /app/dist/ /usr/share/nginx/html/

RUN chmod 0555 /usr/local/bin/frontend-entrypoint.sh \
    && chmod 0444 /usr/share/nginx/html/risk-runtime-config.js

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD wget --quiet --tries=1 --spider http://127.0.0.1/healthz || exit 1

ENTRYPOINT ["/usr/local/bin/frontend-entrypoint.sh"]
CMD ["nginx", "-g", "daemon off;"]
