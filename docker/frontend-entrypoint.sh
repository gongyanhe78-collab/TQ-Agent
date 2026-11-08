#!/bin/sh
set -eu

# 生产容器通过同源地址访问 API；该配置只影响当前浏览器会话。
index_file="/usr/share/nginx/html/index.html"
if ! grep -q "risk-runtime-config.js" "$index_file"; then
    sed -i 's#</head>#<script src="/risk-runtime-config.js"></script></head>#' "$index_file"
fi

exec "$@"
