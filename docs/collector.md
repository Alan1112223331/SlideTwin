# 后台任务收集与状态判断

`collect_jobs.py` 只查询、下载、保存日志，不提交或重试翻译任务，也不启动 Docker。容器仍使用 `restart: "no"`，需要手动启动。

先在运行目录准备 `manifest.json`，其中每个 `samples` 元素至少含有已有任务的 `slug`、`job_id`，可附 `name`、`pages`。旧的八份课件清单可以直接使用，不会重新上传。

```json
{
  "samples": [
    {"slug": "course-01", "job_id": "existing-server-job-id", "pages": 90}
  ]
}
```

在项目目录启动独立后台进程。密钥通过本地 `.env` 或环境变量读取，不放在命令行里：

```powershell
.venv\Scripts\python.exe scripts\collect_jobs.py start `
  --run-dir output\my-run --env-file .env `
  --container slidetwin-slidetwin-1
```

后台 supervisor 与收集进程脱离启动终端/Codex 会话，正常关闭聊天窗口不会结束它们。收集进程退出时 supervisor 会重新拉起，已下载文件先核对 SHA-256，不重复提交任务。断开的服务日志从实际已保存的最后时间补齐；日志快照是短进程，没有遗留的 `docker logs --follow` 子进程。每个运行目录都有独占锁，不能同时启动两个正规的收集器。

查询时使用动态状态命令，不要仅看旧 `summary.json` 中的 `running`：

```powershell
.venv\Scripts\python.exe scripts\collect_jobs.py status --run-dir output\my-run
```

- `finished_and_archived == documents`：所有任务终止且产品、日志归档完成。`failed` 表示至少一个服务任务失败；`completed_with_warnings` 表示任务完成但有警告。
- `monitoring.collector: "alive"`：独立心跳新鲜。`unresponsive` 表示默认超过 30 秒没有心跳，任务状态只能视为最后一次观察，不代表服务任务停止。
- `monitoring.supervisor` 同样判断后台守护进程。若两者失联，服务中的翻译可能仍继续；重新执行同一条 `start` 命令即可继续收集。
- 正常全部结束后，心跳状态为 `finished`，收集器和 supervisor 自动退出，Docker 服务不会因此停止。

心跳每 5 秒更新，独立于下载/归档，长文件传输不应被误报为收集器失联。`health.json` 保存 supervisor 最近的判断，`status` 命令每次都按当前时间重新计算，避免守护进程本身停止后继续显示旧的存活状态。

产物：`summary.json`、`collector.json`、`supervisor.json`、`health.json`、`service.log`、`events.jsonl`；每份任务的 `jobs/<slug>/artifacts/`、`logs/`、`server-job/`。不指定 `--container` 时只能保存远程 API 公开的产品，状态会明确标为 `published_artifacts_only`，不能宣称已收集服务端私有日志。

也可以前台运行用于诊断：

```powershell
.venv\Scripts\python.exe scripts\collect_jobs.py run --run-dir output\my-run --env-file .env --container slidetwin-slidetwin-1
```

`start` 不依赖自动重启系统服务，也不注册计划任务。电脑关机、Docker 停止、服务凭据失效会影响收集或翻译；重新开启后仍可从同一运行目录恢复。运行目录应放在被 Git 忽略的 `output/` 下，包含原课件和私人诊断资料。

## 请求排队与模型超时

`provider.request_deadline_seconds` 限制实际 HTTP 调用及重试/退避消耗的预算。等待并发槽位、RPM 或 TPM 的时间另计，排队不会把一个尚未发出的模型调用判为 HTTP 超时。

`provider.queue_timeout_seconds = 0` 默认不额外限制本地排队，整个服务任务仍有 `SLIDETWIN_JOB_TIMEOUT` 上限。可以设为正数提供独立队列预算；达到该预算会标为 `queue_timeout`，不会尝试通过换模型规避同一个账号的配额。主模型和 fallback 共享账号限流器。

请求日志分别记录 `pool_queue_seconds`、`queue_seconds`、`network_seconds`、`total_seconds`。`http_status` 只有请求收到响应头时才存在；`queue_timeout` 的 `network_seconds` 为 0。断流时已收到的模型文字仍交给原有协议层保留。
