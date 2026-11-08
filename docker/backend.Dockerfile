# 系统包默认使用国内镜像；可通过构建参数覆盖，避免外网软件源造成不可复现的构建失败。
ARG DEBIAN_MIRROR=mirrors.aliyun.com

# 构建阶段只负责安装 Python 依赖，避免把编译工具链带入运行镜像。
FROM public.ecr.aws/docker/library/python:3.12.7-slim-bookworm AS builder

ARG DEBIAN_MIRROR

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_ROOT_USER_ACTION=ignore

RUN sed -i "s|http://deb.debian.org|https://${DEBIAN_MIRROR}|g" \
        /etc/apt/sources.list.d/debian.sources \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY requirements.txt ./
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install -r requirements.txt

# 运行阶段只保留健康检查和轻量 PDF 解析所需的系统依赖。
FROM public.ecr.aws/docker/library/python:3.12.7-slim-bookworm AS runtime

ARG DEBIAN_MIRROR

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1 \
    PATH="/opt/venv/bin:${PATH}" \
    PYTHONPATH=/app \
    HOME=/home/appuser \
    TZ=Asia/Shanghai

RUN sed -i "s|http://deb.debian.org|https://${DEBIAN_MIRROR}|g" \
        /etc/apt/sources.list.d/debian.sources \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 appuser \
    && useradd --uid 10001 --gid 10001 --create-home appuser

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY --chown=appuser:appuser alembic.ini ./alembic.ini
COPY --chown=appuser:appuser backend ./backend
COPY --chown=appuser:appuser scripts ./scripts
COPY --chown=appuser:appuser docker ./docker

RUN mkdir -p \
        /app/data \
        /app/data/document_index \
        /app/data/document_images \
        /app/data/numba_cache \
        /app/backend/app/agents/case_multidim_search/runtime \
    && chown -R appuser:appuser \
        /app/data \
        /app/backend/app/agents/case_multidim_search/runtime

USER appuser:appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=45s --retries=5 \
    CMD curl --fail --silent --show-error http://127.0.0.1:8000/api/health || exit 1

CMD ["python", "-m", "uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
