# 系统架构与请求链路

> 本文档用于面试讲解：先看整体架构，再看一条真实请求的完整链路，最后说明
> 模块化单体与单机部署的边界。所有文档与注释使用中文。

## 1. 总体架构

```mermaid
flowchart LR
    subgraph 客户端
        U[浏览器]
    end

    subgraph 单机 Docker Compose
        F[前端 Nginx 静态资源]
        A[API FastAPI]
        R[Agent Runtime 与业务服务]
        P[(PostgreSQL 权威数据)]
        RD[(Redis 短期缓存与限流)]
        V[(Chroma / Milvus 向量索引)]
    end

    M[外部模型 DashScope]

    U -->|HTTPS 8080| F
    F -->|JSON / SSE| A
    A --> R
    R -->|会话/消息/运行/分享| P
    R -->|短期上下文/限流| RD
    R -->|检索与记忆| V
    R -->|Chat/Embedding/Rerank| M
    P -.->|备份恢复| V
```

关键点：

- 前端是 Nginx 托管的静态构建产物，无业务状态。
- API 与 Agent Runtime 在同一进程内以模块化单体组织，避免微服务网络开销。
- PostgreSQL 是唯一权威数据源；Redis 与向量索引均可从权威数据重建。
- 外部模型只承担生成与检索重排，不保存业务事实。

## 2. 一条请求的完整链路

```mermaid
sequenceDiagram
    participant U as 浏览器
    participant F as 前端
    participant A as API
    participant S as 会话存储
    participant AG as Agent Runtime
    participant P as PostgreSQL
    participant RD as Redis
    participant V as 向量库
    participant M as 外部模型

    U->>F: 打开页面
    F->>A: 登录/刷新令牌
    A->>P: 校验用户并读取会话
    A-->>F: 返回会话列表

    U->>F: 输入问题并发送
    F->>A: POST /api/query（带幂等键）
    A->>AG: 意图路由（本地知识库 or 开放域）
    AG->>P: 加载会话上下文与用户归属
    AG->>RD: 读取短期上下文（未命中回退 P）
    AG->>V: 向量检索候选证据
    AG->>M: 生成回答（必要时联网搜索）
    AG->>P: 持久化用户消息、回答与运行记录
    AG-->>A: 返回证据与回答
    A-->>F: SSE 流式回答 + request_id/trace_id
    F-->>U: 渲染回答与证据来源
```

## 3. 边界与取舍

- 同一进程内的模块化单体：会话、意图、检索、生成按服务类分层，但共享一个
  FastAPI 生命周期与数据库连接池，适合单机演示与开发迭代。
- 降级边界：Redis 故障不影响历史会话，向量库故障有结构化检索回退，
  模型故障返回已核验证据或明确无答案，不会伪造结论。
- 可观测性：每个请求都有 `request_id` 与 `trace_id`，`/metrics` 暴露
  请求量、错误量、模型调用成功率、缓存/向量回退次数。
