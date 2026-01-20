# Skills 模块接口规范（后端）

版本：1.0.0

## 1. 目标
- 提供标准化的“技能（Skill）”注册、发现与调用接口，用于把后端能力以可组合的方式暴露给上层系统。
- 保持现有 API 完全兼容；Skills 作为新增 API 端点提供。

## 2. 基础概念
- Skill：具备唯一 `id`、版本号与可调用函数的能力单元。
- Registry：Skill 的注册表，负责列举与按 `id` 查找。

## 3. API 基础约定
### 3.1 Content-Type
- 请求/响应：`application/json; charset=utf-8`

### 3.2 统一调用请求结构（SkillInvokeRequest）
```json
{
  "request_id": "string",
  "params": {},
  "timeout_ms": 1000
}
```

字段说明：
- `request_id`：调用方生成的请求标识，用于链路追踪与幂等对齐。
- `params`：技能参数对象，按技能自定义。
- `timeout_ms`：最佳努力超时（后端在执行结束后做超时判定并返回 TIMEOUT）。

### 3.3 统一调用响应结构（SkillInvokeResponse）
```json
{
  "request_id": "string",
  "ok": true,
  "result": {},
  "error": null
}
```

字段说明：
- `ok=true` 时：`result` 必须存在，`error` 为空。
- `ok=false` 时：`error` 必须存在，`result` 为空。

## 4. 错误码定义（SkillError.code）
| code | 含义 | 典型场景 |
|---|---|---|
| INVALID_REQUEST | 请求格式不合法 | 缺少必填字段、类型不匹配 |
| SKILL_NOT_FOUND | 技能不存在 | skill_id 未注册 |
| SKILL_ERROR | 技能执行失败 | 技能内部抛异常 |
| TIMEOUT | 超时 | 执行耗时超过 timeout_ms |
| INTERNAL_ERROR | 内部错误 | 未预期错误 |

错误结构：
```json
{
  "code": "SKILL_ERROR",
  "message": "string",
  "details": {}
}
```

## 5. 现有内置技能
- `echo`：回显参数，用于联调测试。
- `gpu_status`：返回 GPU 可用性、CUDA/cuDNN 版本与监控指标。
- `video_pipeline_status`：返回视频处理生产者-消费者流水线的状态与线程健康信息。

