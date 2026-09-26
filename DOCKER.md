# Docker 部署与 HTTP API

本服务接收 PDF，异步生成中文、中英对照和 Docling 原始提取结果。镜像使用 Linux CPU、Python 3.12，自带中文字体和 Poppler；无需在宿主机安装 Docling。当前镜像标签为 `slidetwin:0.4.1`，通过本项目 Dockerfile 本地构建。

## 启动

1. 将 `.env.example` 复制为 `.env`。
2. 在 `.env` 设置 `SLIDETWIN_API_TOKEN`，至少 16 个字符，建议随机生成。它用于调用本服务，与模型 API Key 不同。
3. 如需翻译，设置 `SLIDETWIN_API_KEY` 为模型服务密钥。仅提取 Docling 时可留空。
4. 将 `config.example.toml` 复制为 `config.local.toml`，按模型服务填写模型、上下文限制、RPM / TPM。在 `.env` 增加 `SLIDETWIN_CONFIG_FILE=./config.local.toml`。本机 Windows 字体、tokenizer 或密钥路径不能直接供 Linux 容器使用，未另行挂载时将相应路径留空，使用容器字体与环境变量密钥。
5. 先构建镜像（不会启动服务）：

```powershell
docker compose build
```

需要使用服务时手动启动：

```powershell
docker compose up -d --no-build
docker compose ps
```

容器配置为不自动重启。关闭 Docker Desktop 后再次打开，SlideTwin 不会随之启动；使用完可运行 `docker compose stop`。重建现有服务时，请先停止正在运行的容器，再执行 `docker compose build` 和 `docker compose up -d --no-build`。

默认地址 `http://127.0.0.1:8000`，交互接口文档 `http://127.0.0.1:8000/docs`。点击 Authorize 输入本服务的 API Token 后调用任务接口。`/healthz` 和接口说明不需要鉴权，其余接口需要 `Authorization: Bearer <token>`。

首次实际提取会下载 Docling 模型，可能比后续任务慢。模型缓存在 `slidetwin-models` 数据卷，上传文件、任务、检查点和输出保存在 `slidetwin-data`；普通重启不会丢失。`docker compose down -v` 会删除这些卷，日常停止请勿加 `-v`。

默认绑定本机回环地址。如需由其它设备访问，可设置 `SLIDETWIN_BIND_ADDRESS` 并在服务前配置 HTTPS 反向代理。服务使用同一个访问令牌，适用于单用户或受信任内部服务，不提供多租户权限隔离。

## 一次上传，获取全部结果

在安装了 SlideTwin 依赖的客户端目录运行：

```powershell
.\.venv\Scripts\python.exe scripts/api_client.py "input.pdf" --out output/api-result
```

脚本读取本地 `.env` 的 `SLIDETWIN_API_TOKEN`，提交任务、轮询并下载已生成的文件，包括失败情况下仍可用的结果。可加 `--url https://your-server.example` 指向远端服务。

也可直接调用 HTTP 接口。在 PowerShell 中使用 `curl.exe`，将 `JOB_ID` 换成上传返回的 `id`：

```powershell
curl.exe -H "Authorization: Bearer $env:SLIDETWIN_API_TOKEN" `
  -F "file=@input.pdf" -F "outputs=chinese,bilingual,docling" `
  http://127.0.0.1:8000/v1/jobs

curl.exe -H "Authorization: Bearer $env:SLIDETWIN_API_TOKEN" `
  http://127.0.0.1:8000/v1/jobs/JOB_ID

curl.exe -H "Authorization: Bearer $env:SLIDETWIN_API_TOKEN" `
  -o chinese.pdf http://127.0.0.1:8000/v1/jobs/JOB_ID/artifacts/chinese.pdf
```

注意：Docker Compose 会读取 `.env`，但 PowerShell 不会自动把其中变量加入 `$env:`。使用 curl 前应设置当前终端的令牌环境变量；Python 客户端会自动读取 `.env`。

提交返回 HTTP 202，任务状态依次为 `queued`、`running`、`completed` / `completed_with_warnings` / `failed`。查询响应的 `artifacts` 给出下载地址、字节数及 SHA-256。尚未生成的文件不在列表中。Docling 原始提取完成后即发布对应文件，无需等待翻译结束。

## 输出格式

| outputs 参数 | 下载文件 | 含义 |
| --- | --- | --- |
| `chinese` | `chinese.pdf`、`chinese.json`、`chinese.md` | 仅译文页 PDF，及按页 / 文字块组织的中文内容 |
| `bilingual` | `bilingual.pdf`、`bilingual.json`、`bilingual.md` | 原页 → 译文页交替的 PDF，及原文 / 译文配对内容 |
| `docling` | `docling.json`、`docling.md` | Docling 直接导出的原语言对象树及 Markdown，无翻译、无 SlideTwin 文本规则改写 |
| 所有任务 | `report.json` | 各类输出是否完整、警告和错误类别 |

中文和中英两份 PDF 共用一次翻译，中文版本直接选取中英 PDF 中的译文页，不再次调用模型或重新排版。术语、公式、变量和无法翻译的图像内文字可能仍保留原样；中文模式表示只含译文页面，不保证每个字形都是中文。

成品 PDF 始终保持原页面尺寸，不附加排版诊断框、内部编号或页底补充区域。无法放置的译文保存在 `report.json` 的 `page_issues[].unplaced_translations`（含页码对应信息、块 ID、原位置和完整译文），并继续出现在中文 / 中英 JSON 与 Markdown 中；任务仍报告 `completed_with_warnings`，不会把未放置的内容当作已完成排版。

`chinese.json` / `bilingual.json` 使用 `slidetwin.translation.v1` schema，包含源页码、页面尺寸、块 ID、坐标、角色和译文；后者额外包含原文。它们是 SlideTwin 的结果格式。`docling.json` 才是未经 SlideTwin 改写的 Docling schema。Markdown 用于读取和下游处理，不承诺还原 PDF 布局。

如指定 `pages=1,3-5`，PDF 和文本结果仅输出这些页；Docling 对象树中的页号按所选子文档从 1 开始，原页对应关系见任务的 `selected_pages`。JSON 中原始输入名称统一为服务内部名称，不保留客户端文件路径。

仅提取的示例：

```powershell
.\.venv\Scripts\python.exe scripts/api_client.py input.pdf --outputs docling --out output/extracted
```

## 任务恢复及清理

| 接口 | 用途 |
| --- | --- |
| `POST /v1/jobs` | multipart 上传 `file`；可选 `outputs`、`pages` |
| `GET /v1/jobs/{id}` | 查询状态与已发布文件 |
| `GET /v1/jobs/{id}/artifacts/{name}` | 下载白名单中的结果文件 |
| `POST /v1/jobs/{id}/retry` | 重试失败或带警告的任务，复用仍匹配的检查点 |
| `DELETE /v1/jobs/{id}` | 删除已结束任务的数据；运行中的任务返回 409 |

任务状态原子写入数据卷。服务重启会重新排队未完成任务，复用匹配的提取与翻译检查点。突然中断时当前请求尚未保存的模型输出可能需要重调，不能保证对外部 API 的恰好一次调用。

任务内部各组翻译仍按配置进行异步并发。默认一次运行一个文档任务，避免多文档同时挤占模型额度或内存；`SLIDETWIN_JOB_WORKERS` 可调整文档并行数，各文档目前独立计算模型限额，需要相应分摊账户 RPM / TPM。完成的文档会立即释放进程槽位。一个数据卷只能由一个 API 进程管理，不能直接启动多个 Uvicorn worker 共用它。

| 环境变量 | 默认值 |
| --- | --- |
| `SLIDETWIN_JOB_WORKERS` | 1 |
| `SLIDETWIN_MAX_PENDING` | 16，运行与排队任务合计 |
| `SLIDETWIN_MAX_UPLOAD_MB` | 100 |
| `SLIDETWIN_MAX_PAGES` | 500，按输入文档总页数 |
| `SLIDETWIN_JOB_TIMEOUT` | 7200 秒 |
| `SLIDETWIN_PORT` | 8000 |

任务不自动过期，使用 DELETE 清理即可。错误响应不会返回服务内部路径或原始异常，详细诊断留在数据卷中对应任务的 `worker.log`，不提供下载接口。

## 构建边界与验证

`.dockerignore` 默认拒绝全部文件，仅允许运行源码、依赖声明、构建脚本和示例配置进入构建上下文；真实密钥、本地配置、课件、Git 历史和输出不进入镜像。容器以非 root 用户运行。也支持通过运行时挂载文件设置 `SLIDETWIN_API_KEY_FILE`、`SLIDETWIN_API_TOKEN_FILE`，避免将密钥直接写在命令行。

镜像从固定提交下载简体中文 Noto Sans SC 字体并验证 SHA-256，生成静态 TrueType 字重。构建时检查中文渲染后的文字提取，避免将标准汉字映射为兼容字形；字体许可证随镜像保留。这是字体映射处理，不修改模型译文。

`requirements-docker.txt` 固定了测试镜像中的 Linux 依赖。镜像使用 CPU PyTorch，不包含 CUDA。运行环境需能下载公开模型权重并访问配置的翻译端点；离线部署需预先准备模型缓存。原版 Docling 提取机制与模型缓存说明见 [Docling 官方文档](https://docling-project.github.io/docling/usage/advanced_options/)；服务进程生命周期参考 [FastAPI 容器部署文档](https://fastapi.tiangolo.com/deployment/docker/)。

本功能增加 HTTP、持久化任务和多格式导出，保留已有翻译 / 排版逻辑。此前已知的浅色标签、公式与文本碰撞等质量限制仍需视觉检查，不因容器化自动消失。
