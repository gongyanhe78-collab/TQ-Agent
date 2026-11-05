# Smart_Case_Match 对接接口文档

本文档用于其他项目接入 `Smart_Case_Match` 相似个例智能匹配能力。文档描述当前目录对应的 HTTP 接口、请求字段、SSE 流式事件、返回数据和证据资源加载方式。

## 1. 接口概览

默认服务地址：

```text
http://127.0.0.1:8000
```

实际部署时请将主机和端口替换为目标服务地址。接口前缀固定为：

```text
/api/smart-case-match
```

| 方法 | 路径 | 用途 |
|---|---|---|
| `GET` | `/api/smart-case-match/health` | 检查本地个例库、Embedding、Rerank、LLM 状态 |
| `POST` | `/api/smart-case-match/conversation/stream` | 推荐：提交自然语言并通过 SSE 获取阶段结果 |
| `POST` | `/api/smart-case-match/search` | 提交结构化条件并同步获取完整结果 |
| `GET` | `/api/smart-case-match/progress/{progress_id}` | 查询进度快照，适合非 SSE 监控 |
| `GET` | `/api/smart-case-match/assets/chunks/{chunk_id}` | 加载文字证据正文 |
| `GET` | `/api/smart-case-match/assets/images/{image_id}` | 加载图片证据文件 |
| `GET` | `/api/smart-case-match/conversation` | 打开内置自然语言页面，仅用于人工访问或联调 |

当前接口不要求调用方直接访问聊天模型、Embedding 或 Rerank 服务。对接方只需要调用本模块接口。

## 2. 推荐对接：自然语言流式接口

### 2.1 请求

```http
POST /api/smart-case-match/conversation/stream HTTP/1.1
Content-Type: application/json
Accept: text/event-stream
Cache-Control: no-cache
```

请求体：

```json
{
  "message": "2026年1月12日至15日，山西出现强寒潮雨雪过程。北中部有大到暴雪，局部积雪15到22厘米，伴有6到8级偏北风和9级以上阵风，需重点关注暴雪落区、道路结冰和设施农业防冻。",
  "progress_id": "weather_console_20260820_001"
}
```

字段说明：

| 字段 | 类型 | 必填 | 限制与说明 |
|---|---|---|---|
| `message` | string | 是 | 2～6000 字。输入过程日期、灾种、区域、实况、环流、强度和重点关注事项。 |
| `progress_id` | string | 否 | 最长 80 字。调用方自己的关联编号，不作为任务主键。 |

服务端会生成真正的 `run_id`，后续所有 SSE 事件均携带该编号。日期会由规则层结合 `Asia/Shanghai` 业务日期最终确定；调用方不应依赖模型自行推断的日期。

### 2.2 SSE 消息格式

每条消息遵循标准 SSE 格式：

```text
event: 事件名
data: JSON对象

```

建议使用 `event` 字段分发事件，使用 `data` 字段解析 JSON，不要按消息到达顺序猜测阶段类型。

### 2.3 事件顺序

正常流程通常按以下顺序发送：

```text
accepted
  -> parsed_query
  -> heartbeat / stage（可能交替出现）
  -> matched_cases
  -> forecast
  -> completed
```

发生异常时会发送 `error`，客户端应停止等待 `completed`。客户端断开连接时，服务端会取消任务并写入取消状态审计记录。

### 2.4 事件示例

#### accepted

```text
event: accepted
data: {"run_id":"smart_case_chat_8f1d4c9e","stage":"parse_natural_query","message":"正在理解天气过程描述","percent":2}

```

#### parsed_query

```json
{
  "run_id": "smart_case_chat_8f1d4c9e",
  "message": "过程信息识别完成，正在检索历史个例",
  "percent": 5,
  "parse_status": "called",
  "query": {
    "process_name": "2026年1月12日至15日强寒潮雨雪过程",
    "start_date": "2026-01-12",
    "end_date": "2026-01-15",
    "date": "",
    "disaster_types": ["雨雪", "降雪", "暴雪", "寒潮", "大风"],
    "affected_areas": ["全省", "北部", "中部", "南部"],
    "observation_description": "北中部有大到暴雪，局部积雪15到22厘米。",
    "circulation_description": "强冷空气南下，前期有一定水汽。",
    "intensity_description": "降温12到18℃，阵风9级以上。",
    "metric_descriptions": {
      "积雪深度": "15到22厘米",
      "降温幅度": "12到18℃",
      "最大阵风": "9级以上"
    },
    "raw_query": "原始自然语言过程描述"
  }
}
```

`query` 是系统规则化后的结果，适合对接方在界面上展示“系统理解的查询条件”。其中 `date_expression`、`date_evidence` 等模型内部日期表达不会进入稳定接口。

#### stage

```json
{
  "run_id": "smart_case_chat_8f1d4c9e",
  "stage": "rank_and_select",
  "message": "候选排序完成，正在提取个例参考",
  "percent": 55
}
```

`stage` 事件用于显示进度，不包含完整业务正文。调用方应允许未知的 `stage` 值，以兼容后续增加处理节点。

#### heartbeat

```json
{
  "run_id": "smart_case_chat_8f1d4c9e",
  "message": "模型仍在处理中",
  "percent": 55,
  "stage": "extract_case_references"
}
```

心跳不代表阶段完成，只说明连接仍然有效。客户端不应据此重复发起请求。

#### matched_cases

该事件在最终选中的个例完成参考经验、匹配理由、关键差异和证据整理后发送。页面不会先显示未完成的个例卡片。

```json
{
  "run_id": "smart_case_chat_8f1d4c9e",
  "phase": "referenced",
  "query_summary": {
    "date_range": "2026-01-12 至 2026-01-15",
    "disaster_types": ["雨雪", "暴雪", "寒潮"],
    "affected_areas": ["全省", "北部", "中部", "南部"]
  },
  "matched_cases": [
    {
      "case_id": "case-2025-01-23",
      "title": "1月23-26日雨雪寒潮大风天气过程",
      "date_range": "2025年1月23-26日",
      "disaster_types": ["雨雪", "暴雪", "寒潮", "大风"],
      "affected_areas": ["全省", "北部", "中部", "南部"],
      "retrieval_score": 0.90,
      "score_breakdown": {
        "area": 1.0,
        "disaster": 0.93,
        "temporal": 1.0,
        "mechanism": 0.9,
        "intensity": 0.8,
        "evolution": 0.9
      },
      "match_reasons": [
        "双方均受强冷空气影响，北中部降雪先增强后扩展，过程演变方向相近。"
      ],
      "reference_points": [
        {
          "text": "在强寒潮配合前期水汽的配置下，降雪范围可由北中部逐步扩大，强降雪区需结合积雪深度判断。",
          "evidence_chunk_ids": ["case-2025-01-23-chunk-027"]
        }
      ],
      "similarities": ["强冷空气南下与前期水汽共同作用"],
      "differences": ["历史过程持续时间更长，当前过程积雪深度更高。"],
      "evidence_images": [
        {
          "image_id": "case-2025-01-23-fig-001",
          "display_no": 1,
          "caption": "过程降雪和积雪深度分布",
          "url": "/api/smart-case-match/assets/images/case-2025-01-23-fig-001"
        }
      ]
    }
  ],
  "warnings": []
}
```

说明：

- `case_id` 和 `evidence_chunk_ids` 是内部关联 ID，必须原样保存，供后续证据接口使用。
- 业务正文应使用 `title`、`date_range` 等名称，不应把 `FST...`、`chunk...` 等内部标识展示给最终用户。
- `score_breakdown` 的维度由灾种配置动态决定，调用方不要假定所有灾种都有同样的字段。
- `reference_points` 是可迁移经验，不等同于当前过程预报结论。

#### forecast

```json
{
  "run_id": "smart_case_chat_8f1d4c9e",
  "forecast_summary": {
    "similarity_assessment": "当前过程与历史寒潮雨雪过程在强冷空气南下和降雪演变上相近；但当前积雪深度更高，不能直接照搬历史量级。",
    "core_features": [
      "北中部降雪先增强后扩展，积雪深度是当前重点比较指标。"
    ],
    "main_risk": "暴雪落区和积雪深度偏差可能扩大道路结冰及设施农业防冻风险。",
    "confidence": 0.90,
    "support_case_ids": ["case-2025-01-23"]
  },
  "forecast_tips": [
    {
      "tip_id": "tip-1",
      "priority": 1,
      "focus_object": "北中部暴雪落区和积雪深度",
      "possible_bias": "局地最大积雪深度可能偏小",
      "suggested_action": "优先核查雷达、自动站和积雪实况，及时订正暴雪落区及量级。",
      "support_case_ids": ["case-2025-01-23"],
      "evidence_chunk_ids": ["case-2025-01-23-chunk-027"],
      "consensus_level": "单个例提示",
      "confidence": 0.90
    }
  ],
  "warnings": []
}
```

`forecast_summary` 是跨个例综合结果，`forecast_tips` 是面向业务人员的行动提示。提示中的证据 ID 经过服务端校验，调用方应保留但不直接展示给普通用户。

#### completed

`completed` 返回完整的 `SmartCaseMatchResponse`，包含 `run_id`、`status`、`query_summary`、`matched_cases`、`forecast_summary`、`forecast_tips`、`warnings` 和 `audit`。调用方可以用该事件作为最终提交完成标志，并以此事件中的数据覆盖本地临时状态。

#### error

```json
{
  "run_id": "smart_case_chat_8f1d4c9e",
  "message": "相似个例匹配失败：具体错误信息"
}
```

`error` 事件出现时不要继续等待 `completed`。建议保留 `run_id`，便于结合服务端日志和审计文件排查。

### 2.5 JavaScript 调用示例

浏览器原生 `EventSource` 不支持 POST 请求体，因此推荐使用 `fetch()` 读取 SSE 流：

```javascript
async function runSmartCaseMatch(message, onEvent) {
  const response = await fetch("/api/smart-case-match/conversation/stream", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Accept": "text/event-stream"
    },
    body: JSON.stringify({
      message,
      progress_id: `integration_${Date.now()}`
    })
  });

  if (!response.ok || !response.body) {
    throw new Error(`HTTP ${response.status}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  while (true) {
    const {value, done} = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), {stream: !done});
    const messages = buffer.split("\n\n");
    buffer = messages.pop() || "";

    for (const messageText of messages) {
      const eventLine = messageText.split("\n").find(line => line.startsWith("event: "));
      const dataLine = messageText.split("\n").find(line => line.startsWith("data: "));
      if (!eventLine || !dataLine) continue;
      onEvent(eventLine.slice(7), JSON.parse(dataLine.slice(6)));
    }
    if (done) break;
  }
}
```

调用方应至少处理 `accepted`、`parsed_query`、`stage`、`heartbeat`、`matched_cases`、`forecast`、`completed` 和 `error`。未知事件应记录并忽略，不能导致整个连接解析失败。

## 3. 结构化同步接口

当调用方已经有结构化天气过程字段，不需要先经过自然语言解析时，可以调用：

```http
POST /api/smart-case-match/search HTTP/1.1
Content-Type: application/json
Accept: application/json
```

请求示例：

```json
{
  "process_name": "晋南持续性强降水过程",
  "start_date": "2026-07-10",
  "end_date": "2026-07-12",
  "date": "",
  "disaster_types": ["暴雨", "短时强降水"],
  "affected_areas": ["临汾", "运城"],
  "observation_description": "晋南出现持续性降水，局地小时雨强较大。",
  "circulation_description": "副高边缘暖湿气流与低层切变共同影响。",
  "intensity_description": "局地小时雨强达到暴雨量级。",
  "metric_descriptions": {
    "累计降水量": "过程累计量级较大",
    "最大小时雨强": "局地小时雨强较大"
  },
  "raw_query": "重点关注强降水落区和小时雨强。",
  "progress_id": "integration_sync_001",
  "top_n": 3,
  "diversity_mode": "moderate",
  "include_images": true
}
```

字段限制：

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---:|---|
| `process_name` | string | `""` | 过程名称，最长 120 字。 |
| `start_date` / `end_date` | string | `""` | 过程起止日期。 |
| `date` | string | `""` | 日期补充表达。 |
| `disaster_types` | string[] | `[]` | 灾种列表。 |
| `affected_areas` | string[] | `[]` | 影响地市或区域。 |
| `observation_description` | string | `""` | 实况描述，最长 6000 字。 |
| `circulation_description` | string | `""` | 环流描述，最长 6000 字。 |
| `intensity_description` | string | `""` | 强度和持续时间，最长 3000 字。 |
| `metric_descriptions` | object | `{}` | 指标名称到原文描述的映射。 |
| `raw_query` | string | `""` | 补充自然语言，最长 6000 字。 |
| `progress_id` | string | `""` | 进度关联编号，最长 80 字。 |
| `top_n` | integer | `3` | 返回个例数量，范围 3～5。 |
| `diversity_mode` | string | `moderate` | `off` 或 `moderate`。 |
| `include_images` | boolean | `true` | 是否返回图片证据元数据。 |

至少需要提供灾种、影响区域、过程日期或 `raw_query` 中的一项，否则返回 HTTP `422`。

同步接口返回完整 JSON，结构与 SSE 的 `completed` 事件一致：

```json
{
  "run_id": "smart_case_search_8f1d4c9e",
  "status": "completed",
  "query_summary": {},
  "matched_cases": [],
  "forecast_summary": {},
  "forecast_tips": [],
  "warnings": [],
  "audit": {}
}
```

## 4. 健康检查与进度

### 4.1 健康检查

```bash
curl http://127.0.0.1:8000/api/smart-case-match/health
```

健康检查结果用于判断本地资料和模型客户端是否就绪。调用方不应把模型不可用直接等同于接口不可用；业务接口可能仍能返回结构化匹配或规则兜底结果。

### 4.2 进度查询

```bash
curl http://127.0.0.1:8000/api/smart-case-match/progress/smart_case_chat_8f1d4c9e
```

返回轻量进度快照，不包含完整业务正文和模型输入。示例：

```json
{
  "progress_id": "smart_case_chat_8f1d4c9e",
  "status": "running",
  "stage": "extract_case_references",
  "message": "正在提取历史个例参考",
  "percent": 76
}
```

流式对接优先使用 SSE 的 `stage` 和 `heartbeat`，只有无法保持 SSE 连接时才建议轮询该接口。

## 5. 证据资源接口

SSE 和同步响应只返回证据元数据，不把全部 chunk 正文和图片二进制直接放入主响应。

### 5.1 文字证据

```bash
curl http://127.0.0.1:8000/api/smart-case-match/assets/chunks/case-2025-01-23-chunk-027
```

返回：

```json
{
  "chunk_id": "case-2025-01-23-chunk-027",
  "chunk_no": 27,
  "source_pdf": "case-2025-01.pdf",
  "content": "历史个例正文片段……"
}
```

### 5.2 图片证据

```text
GET /api/smart-case-match/assets/images/{image_id}
```

接口直接返回图片文件。调用方应使用响应中的 `image_id` 或 `url`，不要拼接本地磁盘路径。

## 6. HTTP 状态码与错误处理

| 状态码 | 含义 | 对接方处理 |
|---:|---|---|
| `200` | 请求成功，或 SSE 已建立 | 解析响应或继续读取事件。 |
| `404` | 进度、文字证据或图片不存在 | 提示资源不可用，不要无限重试。 |
| `422` | 请求字段校验失败 | 修正请求字段后重新提交。 |
| `500` | 服务端执行异常 | 记录 `run_id` 或请求编号，按退避策略重试。 |

流式过程中出现 `error` 事件时，HTTP 连接可能已经成功建立，因此不能只根据 HTTP 状态码判断业务是否完成。

`status` 字段含义：

- `completed`：完整流程正常完成；
- `degraded`：部分模型或资料环节不可用，但仍返回可追溯结果；
- `failed`：流程未能形成有效结果。

调用方应同时检查 `status` 和 `warnings`，不要只判断 HTTP `200`。

## 7. 对接注意事项

1. 推荐优先调用 `/conversation/stream`，可以及时展示阶段进度，并在最终 `completed` 到达后统一落库。
2. 不要把 `matched_cases` 到达视为整个任务结束；综合研判会在后续 `forecast` 事件中返回。
3. 不要依赖固定的阶段百分比或固定的 `stage` 名称，前端应按事件字段兼容未知阶段。
4. `case_id`、`chunk_id`、`image_id` 仅用于程序关联，业务页面应展示个例标题、日期和证据说明。
5. `score_breakdown`、`metric_scores` 和 `dimension_scores` 会随灾种配置变化，必须按键动态渲染。
6. 文字和图片证据采用按需加载，建议在用户点击证据时再请求资源接口。
7. 服务端内部使用连接池和 Agent 单例；对接方不需要自行管理模型连接池。
8. 请求可能持续较长时间，HTTP 客户端、反向代理和网关应允许 SSE 长连接并关闭响应缓冲。
9. 断线重连可能创建新的匹配任务。调用方应使用自己的 `progress_id` 做关联，并避免无条件重复提交造成重复计算。
10. 运行审计日志由服务端写入 `Smart_Case_Match/logs/{日期}/{run_id}.json`，对接方如需排查候选排序、模型调用和证据链，应保存 `run_id`。

## 8. Python 服务间调用示例

结构化同步调用示例：

```python
import requests


payload = {
    "process_name": "晋南持续性强降水过程",
    "start_date": "2026-07-10",
    "end_date": "2026-07-12",
    "disaster_types": ["暴雨", "短时强降水"],
    "affected_areas": ["临汾", "运城"],
    "observation_description": "晋南出现持续性降水，局地小时雨强较大。",
    "circulation_description": "副高边缘暖湿气流与低层切变共同影响。",
    "metric_descriptions": {"最大小时雨强": "局地小时雨强较大"},
    "top_n": 3,
    "include_images": True,
}

response = requests.post(
    "http://127.0.0.1:8000/api/smart-case-match/search",
    json=payload,
    timeout=300,
)
response.raise_for_status()
result = response.json()
print(result["run_id"], result["status"])
```

如果对接项目需要自然语言过程理解和阶段进度，应使用支持 SSE 的 HTTP 客户端调用 `/conversation/stream`，不要把流式响应当作一次普通 JSON 响应读取。

## 9. 本地模块调用（同一 Python 项目）

如果其他功能与本目录位于同一个 Python 项目，也可以注入客户端直接使用 Agent，但这属于进程内集成，不是 HTTP 接口：

```python
from pathlib import Path

from backend.app.services.agent.Smart_Case_Match.agent import SmartCaseMatchAgent
from backend.app.services.agent.Smart_Case_Match.schemas import SmartCaseMatchRequest


agent = SmartCaseMatchAgent(
    data_dir=Path("backend/app/services/agent/data"),
    audit_log_dir=Path("backend/app/services/agent/Smart_Case_Match/logs"),
)

try:
    result = agent.run(SmartCaseMatchRequest(
        process_name="晋南持续性强降水过程",
        start_date="2026-07-10",
        end_date="2026-07-12",
        disaster_types=["暴雨"],
        affected_areas=["临汾", "运城"],
    ))
finally:
    # Agent 持有 HTTP 客户端和线程池，进程退出或模块卸载时必须释放资源。
    agent.close()
```

跨项目部署时优先使用 HTTP 接口；只有共享 Python 运行环境且能够明确管理 `agent.close()` 生命周期时，才使用进程内调用。
