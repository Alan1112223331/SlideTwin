# SlideTwin

基于 Docling 和 OpenAI 兼容 Chat Completions 接口的课件 PDF 翻译程序，当前版本 **0.4.0**。支持中文译稿、中英对照 PDF 和 Docling 原始提取结果；当前支持 PDF，原生 PPTX 编辑尚未实现。

本项目独立管理环境、配置、缓存、测试与输出，不修改原课件。真实凭据、课件和运行产物不进入版本管理。GitHub 首次发布标签 `release0.1` 保留程序 0.3.2 的快照。

## Docker 与 HTTP API

部署和调用方式见 [DOCKER.md](DOCKER.md)。准备本地 `.env` 的服务访问令牌和模型密钥后运行 `docker compose up -d --build`，在 `/docs` 查看交互接口。

上传 PDF 到 `POST /v1/jobs`，随后查询任务并下载 `chinese`、`bilingual` 或 `docling` 结果。中文和中英各提供 PDF / JSON / Markdown，Docling 提供原版 JSON / Markdown。仅提取 Docling 不需要翻译 API Key。

## 运行

先按下方“安装与测试”创建环境和本地配置，再在 SlideTwin 目录运行：

```powershell
.\.venv\Scripts\python.exe -m slidetwin translate "D:\路径\课件.pdf" `
  --config config.local.toml `
  --work output/my-course `
  --out output/pdf/my-course-bilingual.pdf
```

完整运行默认包含全部页面。测试选页可加 `--pages 1,3-6`；模型仍获得完整文档的文字语境。扫描课件最好完整解析，未选页的上下文只能从原生文字层取得。`--no-preview` 可关闭对照图生成。

重跑同一命令会复用匹配输入、配置和提示版本的完成记录。0.3.0 改变了翻译策略，因此旧版本译文缓存会失效；验证过的 Docling 对象树仍可复用。

0.3.1 / 0.3.2 保持 0.3.0 的翻译缓存兼容，不需要重译已完成页面。连接超时、重试与退避设置不影响译文缓存身份。

## 模型与程序的边界

模型负责翻译、统一术语和复核意义。程序负责提取、请求调度、结果定位、字体、颜色、缩进、换行及 PDF 排版。

**程序不因英文残留、数字变化、日期表达、译文长度或强调短语不匹配而重试。** 这些旧规则保留为 `PROGRAM REVIEW HINTS`，供固定的一轮复核参考。提示声明可能误报，复核模型有权不采纳。最终模型输出有可定位的非空文本，就保留该文本。

仍可重试的情况是：连接/服务错误、空响应、无法映射到目标 ID，或缺少目标。有效目标立即保存，只补缺失目标。`>>>END>>>` 这类可恢复的分隔符错误直接解析。截断响应中已有的目标也保留。原文强调短语作为子 ID 一并翻译；无法定位对应译文时使用句子的基础样式并记录警告，绝不为了匹配样式要求模型重译或缩短句子。

默认处理顺序：

1. Docling 解析结构，结合原生 PDF 字形坐标、图内文字 OCR 得到内容与版面。
2. 模型读取全文，准备术语指南。超过预算的超长文档才分段并发，保留全部源文。
3. 按 token 预算尽量合并页面进行初译。
4. 每组进行一次意义复核：原文语境 + 初译 + 程序提示，其余组无需等待。
5. 程序排版、验证并输出全部页面。模型不参与排版决定。

一份能装进一个请求的课件，通常是 **3 次模型调用**：术语准备、初译、复核。关闭术语准备或复核会减少调用，默认以质量为先，均开启。

## 单模型异步并发与 fallback

示例配置的主模型是 `Qwen/Qwen3-VL-30B-A3B-Instruct`，备用为 `Qwen/Qwen3-VL-32B-Instruct`。正常请求全部使用主模型；只有主模型连接失败、超时、限流或可恢复服务错误时切换备用。主模型短暂冷却后重新优先使用。旧 `worker_models` 仅作为有序 fallback 的兼容别名，不再轮询分流。

例如允许 8 个在途请求，先启动 8 个；其中一个完成，立刻填入第 9 个，不等另外 7 个。后页可以先完成，完成记录立即写入检查点。复核等待对应初译；术语指南先准备，但辅助准备失败不阻止正文继续翻译。

普通超时、空响应和响应协议损坏先在同一主模型重试，单次请求异常不会立刻将整个主模型停用两分钟。重复失败后使用配置的 fallback；明确的 5xx 服务错误或 429 限流可以提前切换。指数退避带少量随机延迟，等待时释放调用槽位。429 遵守服务端 `Retry-After`（秒数或 HTTP 日期）；等待超过逻辑请求总时限时记录失败，不提前冲击限流接口。401 / 403 和确定的配置错误不会盲目重复调用。

流式响应中已到达的目标继续保留，只补缺失 ID。token 用量字段、请求日志写入或连接清理失败都不使正常译文作废；无法读取用量时，限流器保守保留已预留额度。语言语义提示仍仅交给复核模型，不触发程序重译。

程序按滚动 60 秒窗口同时约束请求数和 token 预算，允许额度内的一起启动，不再人为均匀间隔发送。完成后按服务端用量释放多余预留额度。RPM 按配置原值使用，TPM 乘以 `token_rate_utilization = 0.8`。429 会退避，不会在 80% 基础上反复再乘 80%。

```toml
[provider]
model = "Qwen/Qwen3-VL-30B-A3B-Instruct"
fallback_models = ["Qwen/Qwen3-VL-32B-Instruct"]
concurrency = 0                 # 随已配置 RPM 自动设置；额度未知时先用 8 个槽位
requests_per_minute = 0         # 填控制台真实 RPM；0 表示未知，不是无限额度
tokens_per_minute = 0           # 填控制台真实 TPM 原值，勿提前乘 80%
token_rate_utilization = 0.8
rate_limit_source = "unknown-account-quota"
retries = 3                     # 每次逻辑调用的总尝试次数，包含首次，不是额外再试 3 次
primary_attempts = 2            # 普通瞬时错误优先在主模型尝试的次数
retry_base_delay_seconds = 1
retry_max_delay_seconds = 15
read_timeout_seconds = 120      # 等待响应头或连续无响应数据的超时
request_deadline_seconds = 1800 # 单次逻辑调用的总时限，包含队列、重试与退避
```

请在 `config.local.toml` 填写服务控制台的实际 RPM 和 TPM。示例配置中的 0 表示额度尚未填写；程序不会通过 `/v1/models` 猜测账户额度。也可设 `concurrency = 8` 限制连接数。

硅基流动的额度按账户、按模型设置，付费模型随账户等级变化，见 [官方限流说明](https://api-docs.siliconflow.cn/docs/userguide/faqs/rate-limit-and-upgradation)。限流器维护当前共享客户端的记录，无法知道其他应用的用量。不要启动多个独立进程叠加额度。

## 上下文与图片预算

[硅基流动主模型规格](https://www.siliconflow.com/models/qwen3-vl-30b-a3b-instruct)公布 262K 上下文与 262K 最大输出。配置使用 262,144 token，请求预算为 **209,715 token（80%）**。实际 `max_tokens` 动态设置。

```toml
[provider]
context_window_tokens = 262144
context_utilization = 0.8
model_max_output_tokens = 262144
max_output_tokens = 131072 # 已验证端点接受；实际请求按目标长度缩小
tokenizer_file = ".local/tokenizers/qwen3-vl/tokenizer.json"

[translation]
batch_mode = "auto"
context_mode = "adaptive"
output_expansion_ratio = 1.5
max_context_characters = 0
review = true
glossary = true
image_policy = "ocr_only"
image_neighbors = 0
```

预算计入系统提示、文档语境、术语、目标 ID、图片、JSON schema（如启用）、复核草稿、提示及预计输出。选择尽可能多的连续页，**没有固定 4 页或最多 8 页的限制**。实际复核草稿超出预估时，仅按容量拆复核组，不重做初译。已配置的 TPM 比上下文小，批量也会受 TPM 80% 预算约束。

80% 是输入加输出的规划上限，不能保证实际返回后恰好达到 80%；输出长度及图片 token 由服务端决定。小课件全放进去仍远小于上限时，不填充无用内容。`batch-plan.json` 和请求日志记录估算、实际用量与图片页码。

`adaptive` 在全文能合理装入时直接给全文；更长文档给完整课程目录、术语指南、完整目标页及相邻页、关联远处页。`document` 强制每次全文；`hierarchical` 强制分层语境。不静默裁切目标原文。`batch_mode = "fixed"` 配合 `pages_per_request` 仅供显式指定页数使用。

只有使用了 OCR 结果的页面附原页图片，原生文字页面不附图。`vision = false` 或 `image_policy = "never"` 可禁用图片。可使用 [Qwen 官方 tokenizer](https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct/blob/main/tokenizer.json)，新环境可下载并设置路径；空路径使用保守估算。图片估算保留余量，实际 usage 单独记录。切换模型时，相应调整上下文、输出上限和 tokenizer；fallback 必须能容纳同样的请求预算。

## 通用接口与不支持 JSON 的模型

支持 OpenAI Chat Completions 消息格式的服务均可配置，无需 tool calling 或 OpenAI SDK。只读取用户配置的端点，不向其他服务发送密钥。

| `protocol` | 模型输出 |
| --- | --- |
| `auto`（默认） | `<<<ID>>>译文<<<END>>>`；缺失 ID 经有限补充仍无法解析时，退回单目标纯文本 |
| `tagged` | 带 ID 的普通文本，无 JSON、无工具调用 |
| `plain` | 一个目标的译文，仍提供全文/分层语境和完整目标页，会增加调用次数 |
| `json` | JSON 对象映射 ID 到译文 |
| `json_schema` | 严格 JSON schema |

程序只验证取得可定位的非空内容，不做语言门禁。`plain` 中模型直接写数值，能与原字面值精确对应时恢复程序内部标记；不能对应时保留模型输出，不为此重试。`extra_body` 支持服务特有参数，`token_parameter` 可为 `max_tokens` 或 `max_completion_tokens`，`temperature = "omit"` 可不发送 temperature。纯文本模型设置 `vision = false`。

## 输出与失败交付

重试耗尽后仍在 `--out` 输出完整选中页的待检查译稿，使用已有模型内容，不以原文替代已经取得的译文。确实没有取得内容才标“未取得译文”。**单块排版失败不再触发整页纯文字重排**：成功放置的译文和原页图形保留，只把放不下的文字列在该译文页下方的补充区域。正常页面保持原尺寸，不统一加警告横幅。待检查输出附 `.issues.json`，旧输出先备份再原子替换。如果某个 OCR 图中文字无法安全擦除，会明确记录 `source_pixels_retained`，译文仍列入局部补充区域。

| 故障 | 0.3.1 处理范围 |
| --- | --- |
| 术语准备某段失败 | 保留成功术语，带课件上下文继续翻译；失败记入 `preparation_warnings` |
| 单页超上下文或 TPM 预算 | 单独记录该页失败，其它页继续；已匹配的完成缓存仍可使用 |
| 缓存 JSON 损坏 | 该缓存视为未命中，其它缓存继续用；新版本逐页完成记录可由匹配的总账恢复 |
| 组请求被 400 / 413 / 422 拒绝 | 请求内容问题进行一次逐页恢复，改变请求大小；鉴权、已识别的模型/参数配置错误不盲目逐页重试 |
| 一页补跑失败 | 已命中缓存的同组其它页仍记为完成 |
| 复核拆分后某半失败 | 已完成的一半写入检查点，仅补未完成内容 |
| 纯文本模式强调短语失败 | 主句和其它已完成目标立即保存 |
| 单块预排版或渲染失败 | 隔离该文字块，保留其它译文、插图和表格 |
| 部分 Docling 转换或补充 OCR 失败 | 使用已取得的对象和原生文字，记录提取警告；不把不完整提取永久当作成功缓存 |
| PDF 预览失败 | PDF 已先发布；预览问题单独记录 |

400 错误现在记录服务端的结构化错误说明（脱敏、限长），便于区分实际原因。输入损坏、输入运行中被修改、磁盘/权限故障、Docling 完全不可用等基础故障仍可能阻止导出。程序不会编造没有获得的模型译文。

程序验收检查页数、尺寸、原文页像素、编辑范围外图形、字形与文字是否可提取，不保证译文意义正确。`automated_checks_passed` 表示程序检查通过；`completed_with_warnings` 表示已输出且有问题。视觉检查状态单独记录。

| 工作文件 | 内容 |
| --- | --- |
| `docling-document.json` / `document.json` | 解析对象、文字、字形坐标与语境 |
| `ocr-provenance.json` | 新解析中 Docling OCR 的页级来源信息 |
| `batch-plan.json` | 动态组页、上下文预算、OCR 图片页 |
| `request-timings.jsonl` | 实际模型、排队/首字/总耗时、图片数、token 用量 |
| `translations/` | 术语、分组初译/复核、逐页缓存 |
| `translation-events.json` | 语言提示、缺失 ID 补充、容量拆分 |
| `translation-candidates.json` | 模型返回的可识别内容 |
| `translation-ledger.json` | 已完成页面、译文、用量 |
| `layout-plan.json` | 字体、位置、排版警告 |
| `qa.json` / `run.json` | 程序检查与状态 |
| `preview/pair-*.png` | 原文/译文对照图 |

## 安装与测试

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
Copy-Item config.example.toml config.local.toml
```

Python 3.11+。Docling 初次运行下载解析/OCR 模型，之后复用缓存。中文字体自动尝试 Windows 等线、Linux Noto Sans CJK、macOS 苹方，也可配置字体路径。对照图优先用 Poppler，缺失时用 PyMuPDF。

凭据使用 `SLIDETWIN_API_KEY` 环境变量或 `api_key_file`。若使用 `apikey.txt`，仅在本地保存密钥，配置引用该文件。也可使用：

```powershell
.\.venv\Scripts\python.exe scripts/configure_local.py C:/private/apikey.txt `
  --base-url https://api.siliconflow.cn/v1 --model Qwen/Qwen3-VL-30B-A3B-Instruct
.\.venv\Scripts\python.exe -m slidetwin probe --config config.local.toml
.\.venv\Scripts\python.exe -m pytest -q
```

离线测试不读取真实密钥。实时测试会把课件内容和 OCR 页图片发送给配置的模型服务。

离线测试和匿名化端到端结果见 [VALIDATION.md](VALIDATION.md)，发布记录见 [CHANGELOG.md](CHANGELOG.md)。

## 当前限制与隐私

首次完整样品已完成导出，但仍发现孤字换行、浅色标签漏译、译文与保留公式或图形标记碰撞。自动 QA 通过不代表高保真验收通过，使用前应检查对照图。

仓库只收录程序、合成测试、示例配置和脱敏文档。API Key、本地配置、环境、课件、输出 PDF、页面图片、模型响应、日志、缓存和个人测试脚本均不随发布上传。测试使用的姓名和课程编号是虚构示例。

`scripts/refresh_verified_output.py` 和 `scripts/finish_async_four_courses.py` 是旧批次维护工具，仅为保留回归测试及兼容本地记录而收录，依赖使用者自行提供的本地清单；新文件请使用上面的 `slidetwin translate` 命令。
