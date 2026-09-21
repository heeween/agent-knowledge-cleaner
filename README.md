# Agent Knowledge Cleaner

面向 CRM 聊天导出记录的**离线、增量知识生产管线**：把客服企微群聊记录清洗成「可发布、可审计、可直接挂载 RAG」的知识库。整条链路强调不可变审计链、发布闸与人工复核，任何一条进入知识库的回答都可以追溯回聊天原文。

> English version: [README.en.md](README.en.md)

## 界面预览

流水线已迁移至 yj-kb 服务端，通过 Web 界面操作（「知识库·从零重建」「知识库·增量更新」两个主入口）：

![知识库流水线：增量更新 / 从零重建](docs/screenshots/pipeline-ui-incremental.png)

知识入库前的最后一道闸是人工复核：逐条勾选要导出的条目（publish 默认全选，manual_review 可捞回），填版本号后入库发布：

![人工复核页](docs/screenshots/pipeline-ui-review.png)

## 它解决什么问题

- **脏数据进，干净知识出**：聊天记录里大量寒暄、个案、时效内容无法直接当知识用。管线用多级漏斗（硬过滤 → 精确/向量去重 → 与存量查重 → LLM 准入 → 发布闸）只留下自包含、无隐私、可复用的问答对。
- **答案只允许逐字引用客服原文**：LLM 抽取阶段不允许改写回答，从源头保证知识可溯源。
- **发布可回滚**：每个版本是一个不可变目录（`releases/<版本>/`），带 SHA-256 清单；发布是原子切换 + 热加载，失败自动回退。
- **审计链不可变**：历史发布条目的修订链记录在 `registry/`，源文件变更从不静默改写已发布的知识。

## 工作流程

完整链路共 8 步（对应服务端界面中的流程卡片）：

| 步骤 | 产物 | 说明 |
|---|---|---|
| 1. 变更检测 | 变更文件清单 | 对 markdowns 目录做内容哈希比对，识别新增/修改/删除的聊天文件 |
| 2. 解析 + 切片 | `candidates.jsonl` | 按【时间戳】格式解析消息，按时间间隔/说话人切换/长度上限切出候选问题片段 |
| 3. LLM 抽取 | `qa_extracted.jsonl` | 抽取「客户问题 + 客服回答」知识对，回答只允许逐字引用客服原文 |
| 4. 过滤 + 去重 | `funnel_audit.jsonl` | 硬过滤低置信/低价值/过短/过期项；精确去重 + 向量近似去重 + 与现有知识库查重 |
| 5. LLM 准入 | `judgments.jsonl` | 判定问答是否对齐、可复用、非个案 |
| 6. 发布闸 | `publishability.jsonl` | 独立判定可发布性：答案自包含、无客户隐私、非时效内容 |
| 7. 人工复核 | `export_preview.jsonl` | 页面勾选要导出的条目 |
| 8. 入库 + 发布 | `releases/<版本目录>/` | 写入知识登记表 → 自动对账服务器修订 → 生成向量与发布包 → 校验 → 原子切换 + 热加载 |

干跑（dry run）只执行第 1~2 步，用于核对变更文件与切片数量，不调用 LLM、不入库、不发布。

## 目录结构

```
pipeline.py            # 本地管线 CLI（init / ingest / review / publish / validate-release / rollback）
incremental_kb/        # 增量入库核心库（确定性分析器，不联网）
scripts/01~68          # 旧全量链路脚本（冻结，仅作审计参考，勿改）
scripts/69~76          # 增量链路（funnel v2）：过滤、抽取、去重、准入、发布闸、导出、对账闸
scripts/build_release_embeddings.py   # 向量挂载（离线缓存优先，--generate-missing 补生成）
scripts/release.py                    # 一键编排：发布 → 向量挂载 → 远端预检/同步/热切换
scripts/sync_release.py               # 向 yj-kb 主机暂存版本、校验、原子切换 current、热加载、回滚
registry/              # 消毒后的条目身份/修订契约（Git 跟踪）
releases/              # 不可变发布产物（消毒后的 QA 数据，Git 跟踪；向量文件除外）
docs/                  # 消费契约等文档 + 截图
data/ output/ .state/  # 原始数据、中间产物、本地 SQLite 登记表（不提交）
```

## 快速开始

```bash
pip install -r requirements.txt
python pipeline.py init
python pipeline.py ingest /path/to/managed-chat-directory --dry-run   # 干跑，只看会解析出什么
python pipeline.py ingest /path/to/managed-chat-directory
python pipeline.py review                                             # 处理待复核修订
python pipeline.py review --approve REV-... --note "verified against source"
python pipeline.py publish --version 1.0.1
python pipeline.py validate-release releases/1.0.1
```

普通 ingest 使用确定性分析器，不访问网络。外部 LLM 分析为显式开启，凭证只从环境变量读取：

```bash
OPENAI_API_KEY=... OPENAI_BASE_URL=https://provider.example/v1 OPENAI_MODEL=model-name \
  python pipeline.py ingest incoming/ --analyze --analyzer openai-compatible
```

## 发布、向量与远端同步

一键编排（已完成步骤自动跳过，可安全重跑）：

```bash
python scripts/release.py --version 1.0.4             # 本地发布 + 向量挂载 + 远端只读预检
python scripts/release.py --version 1.0.4 --apply     # 含远端同步与热切换
python scripts/release.py --version 1.0.5 --generate-missing --apply  # 有新条目时补生成缺失向量
```

仅远端操作（默认只读预检，`--apply` 才写远端）：

```bash
python scripts/sync_release.py --version 1.0.2
python scripts/sync_release.py --version 1.0.2 --apply
python scripts/sync_release.py --rollback-to 1.0.0 --apply
```

同步语义：远端按不可变版本目录暂存，校验通过后原子切换 `current` 符号链接并调用 `/kb/reload` 热加载；热加载失败自动恢复旧指针。旧版本目录永不删除。默认目标主机 `root@bk.rcar.vip`，远端根目录 `/root/yj-kb/cleaner_releases`，可用 `--host` 覆盖。

## 安全不变量

- `output/`、`data/`、`.state/`、`.env` 与向量文件不提交；`releases/` 只包含消毒后的 QA 数据。
- 发布基线为 `output/kb_entries_official_v4.jsonl`（SHA-256 见 `output/kb_official_v4_contract.json`）：仅含经 yj-kb kb-admin 通道修订的 26 条存活条目；619 条 v3 退役条目保留修订链但永不重回快照。
- 增量新知识只来自 2026-03-01 之后的聊天，经 funnel v2 脚本（`scripts/69`~`76`）进入，新 ID 从 `KB-0646` 起。
- 同步远端前必须先跑对账闸：`python scripts/76_release_guard_v4.py --release releases/<version>`。
- 文本/状态变更必须走复核；源文件变更从不静默改写已发布的知识条目。

## 测试

```bash
python -m unittest discover -s tests -v
python -m compileall -q pipeline.py incremental_kb tests
```

## 当前状态

整条流水线已迁移进 [yj-kb](docs/YJ_KB_CONSUMER_CONTRACT.md) 服务端（`app/kb_pipeline` 包 + Web 界面，见上方截图），聊天文件以服务器 `markdowns/` 目录为唯一来源；本仓库转为**只读归档**——保留历史审计链、冻结的旧全量脚本（01~67）与全部发布产物。解析/切片逻辑迁移经过奇偶校验：363 个文件 / 3606 个切片与冻结的 issue_candidates 逐条一致。

## 文档

- [docs/YJ_KB_CONSUMER_CONTRACT.md](docs/YJ_KB_CONSUMER_CONTRACT.md) — 发布格式与 yj-kb 消费端契约
- [registry/README.md](registry/README.md) — 条目身份与修订登记说明
- [PROJECT_HANDOFF.md](PROJECT_HANDOFF.md) — 完整开发交接记录（决策、事故、验证记录）
