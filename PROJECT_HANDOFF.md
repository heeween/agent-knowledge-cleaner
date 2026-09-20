# CRM RAG / Agent 知识库项目交接

## 1. 项目目标

本项目原始数据是：

- CRM 客服/客户聊天记录
- 少量正式 CRM 产品/运营文档

最终目标不是把 Markdown 整体直接做 RAG。

正确流程是：

原始聊天
→ 问题/事件切分
→ 问题 + 答案抽取
→ 问题规范化
→ embedding
→ 相似问题聚类
→ Cluster 意图验证
→ Source Q/A Grounding
→ 答案结构识别
→ 冲突/时效处理
→ Publishability Gate
→ 正式 RAG KB

核心原则：

- 一个 Markdown 文件可以包含很多不同 CRM 问题
- 不按客户/租户/SA/客服人员分类
- 这些只作为 source metadata
- embedding 针对 question_normalized
- 不能把整个聊天文件直接 embedding
- 最终 KB 必须可追溯 source
- 同一问题不同原因 ≠ 不同 intent
- 同一问题不同原因 ≠ conflict
- temporary ≠ conflict
- solution 是 LLM 抽取字段，不能反过来证明 answer 正确


---

# 2. 数据规模

原始数据：

- Markdown：363
- 聊天记录：325
- 正式/非聊天 CRM 文档：38
- 解析消息：43,062
- speaker：579

原始 ZIP：

data/raw/markdowns.zip

Markdown 解压目录：

data/raw/markdowns

脚本目录：

scripts/

输出目录：

output/


---

# 3. 工作方式要求

必须一步一步推进。

正确模式：

解释当前 Step
→ 给完整脚本
→ 给运行命令
→ 用户运行
→ 查看输出
→ 判断是否冻结
→ 再进入下一 Step

禁止：

- 一次性重写整个 pipeline
- 随意重跑已经冻结的数据
- 因为发现局部问题就推翻全部流程
- 直接生成最终 KB 而绕过 Grounding / Publishability Gate


---

# 4. 已完成并冻结的基础 Pipeline

## Step 1
document inventory

输出：

output/document_inventory.xlsx

363 个 Markdown。


## Step 2
Exact duplicate detection

结论：

没有 exact duplicate document。


## Step 3
聊天解析

输出：

output/messages.jsonl

共：

43,062 messages


## Step 4
Speaker Profile

输出：

output/speaker_profile.xlsx


## Step 5
Speaker Roles

输出：

output/speaker_roles.xlsx

角色统计：

- customer 282
- unknown 264
- support 32
- internal 1


## Step 6
Issue candidate segmentation

脚本：

scripts/06_segment_issues.py

输出：

output/issue_candidates.jsonl
output/issue_candidates.xlsx

共：

3606 candidate blocks

注意：

candidate block 不是最终 issue boundary。


---

# 5. Step 7：LLM Issue Extraction

已完成并冻结。

模型：

Qwen3.5-Flash

enable_thinking=False

3606 candidates 最终：

- effective candidates: 2877
- discarded: 729
- extracted issues: 4095
- multi-issue candidates: 954
- avg confidence: 0.8647

knowledge_value：

- high 2326
- medium 1172
- low 597

resolution：

- resolved 2563
- unresolved 937
- partial 535
- feature_request 60

temporal：

- stable 2544
- temporary 1205
- future_plan 199
- unknown 144
- historical 3

重要结论：

high + unresolved 仍然可能是稳定知识，
例如“当前不支持某功能”。

所以后续不能要求：

resolution == resolved

才能进入 KB。


---

# 6. Step 8：Question Similarity / Clustering

## Step 8.1
Exact normalized question grouping

16 duplicate-question groups
38 members

同 question_normalized 不代表同知识，
仍需比较答案。


## Step 8.2
Embedding

模型：

Alibaba/Qwen text-embedding-v4

维度：

1024

共：

4095 vectors


## Step 8.3
similar question pairs

脚本：

scripts/10_find_similar_questions.py

主要结果：

>=0.88：1036 pairs


## Step 8.4
pair classification

脚本：

scripts/11_classify_similar_pairs.py

关系：

- same_intent 565
- related 271
- parent_child 190
- different 10


## Step 8.5
question clusters

脚本：

scripts/12_build_question_clusters.py

使用 conservative complete-link / clique merging。

结果：

- total issues 4095
- multi-member clusters 219
- clustered issues 497
- singleton 3598
- largest cluster 12


## Step 8.6 v3
Two-pass cluster validation

脚本：

scripts/14_validate_clusters_two_pass.py

输出：

cluster_validations_v3.xlsx

已冻结。

结果：

- clusters 219
- valid 187
- needs_split 32
- uncertain 0

answer_relation：

- complementary 67
- incomplete_vs_complete 42
- multiple_causes 19
- duplicate 18
- mixed 15
- conflict 15
- temporal_versions 11


重要原则：

Question intent 判断必须只看 Question。

不能因为：

- 不同原因
- 不同 answer path
- 不同 solution

就认为 Question intent 不同。


---

# 7. Step 9：Safe KB Pipeline

## 9.1 safe generation

只处理：

- duplicate
- complementary
- incomplete_vs_complete

127 entries。


## 9.2 generated KB grounding audit

发现 source issue 自身可能 Q/A 错位。


## 9.3 Source Issue Q/A Alignment

脚本：

scripts/17_validate_issue_qa_alignment.py

输出：

issue_qa_alignments.xlsx

结果：

- validated 269
- aligned 204
- partially_aligned 13
- misaligned 14
- no_answer 38
- unusable 53


## 9.4 contamination check

脚本：

scripts/18_check_kb_source_quality.py

结果：

- APPROVED 69
- REGENERATE 16
- MANUAL_REVIEW 42


## 9.5 regenerate contaminated entries

脚本：

scripts/19_regenerate_kb_with_clean_sources.py

16 中：

- regenerated 10
- NO_VALID_SOURCE 6


## 9.6 publishability

脚本：

scripts/20_validate_regenerated_publishability.py

10 中：

- publish 2
- reject 8


最终 Safe KB 冻结：

69 + 2 = 71

这 71 条正式 KB 禁止重跑、禁止重写。


---

# 8. Step 10：Multiple Causes Pipeline

已完整完成并冻结。


## 10.1 generation

scripts/21_generate_multicause_kb.py


## 10.2 v2 Cause Grounding Gate

scripts/23_validate_multicause_items_v2.py

结果：

- causes 59
- valid 54
- rejected 3
- action_fix 2


## 10.3 rebuild

scripts/24_rebuild_multicause_kb.py


## 10.4 v2 Merge Integrity Gate

scripts/26_validate_rebuilt_multicause_v2.py

发现 5 个错误合并。


## 10.5 deterministic repair

scripts/27_apply_multicause_rebuild_fixes.py

输出：

kb_entries_multicause_structurally_clean.xlsx

最终：

17 clusters
54 causes


## 10.6 final publishability

scripts/28_validate_multicause_publishability.py

结果：

- publish 3
- manual_review 13
- reject 1


正式 publish：

QCLUSTER-0030
QCLUSTER-0067
QCLUSTER-0150


## 10.7 export

scripts/29_export_publishable_multicause_kb.py

输出：

kb_entries_multicause_publishable.xlsx


Multiple-causes 正式 KB：

3 条


---

# 9. 当前正式冻结 KB 数量

Safe KB：

71

Multiple Causes：

3

总计：

74

这 74 条正式 KB：

禁止重跑
禁止覆盖
禁止重新生成


---

# 10. Step 11：Mixed Clusters

原始 mixed cluster：

15 个


## Step 11.1 v1

scripts/30_audit_mixed_clusters.py

结果不冻结。

原因：

把不同原因/处理路径误当成不同 intent。


## Step 11.1 v2

mixed_cluster_audits_v2.xlsx

已冻结。

关键原则：

不同原因 ≠ 不同 intent。

结果大致：

- single_intent_structurable 13
- mixed_with_noise 2
- unsafe 0

经过 member filtering：

keep 43
noise 3

15/15 cluster 可以进入下一阶段。


---

# 11. Step 11.2：Mixed Source Q/A Grounding

v1 不冻结。

问题：

solution 被错误地用于补 answer。


## Step 11.2 v2

脚本：

scripts/33_validate_mixed_source_alignment_v2.py

输出：

output/mixed_source_qa_alignments_v2.xlsx

已冻结。

Grounding 原则：

question
↓
answer
↓
solution

Answer 是主要 Grounding Evidence。

Solution：

只能被 Answer 验证，
不能反过来证明 Answer。


结果：

43 keep sources 中：

- aligned 33
- partially_aligned 4
- no_answer 6
- usable_for_kb 37
- unusable 6

15 个 cluster 全部仍至少有 1 个有效 source。

后续 Mixed Pipeline：

只能使用这 37 条：

usable_for_kb == true


---

# 12. Step 11.3：Mixed Knowledge Structure Classification

目标：

对 15 个 mixed cluster 判断未来知识结构。

候选类型：

- direct_synthesis
- structured_howto
- multiple_causes
- scenario_branches
- troubleshooting
- conflict_temporal
- insufficient_evidence


重要原则：

Different Cause != Conflict
Different Workaround != Conflict
Temporary != Conflict


True Conflict 必须满足：

同一个问题
+
同一个条件
+
同一个事实维度
+
结论互斥


例如：

默认密码：

123456
vs 手机号

可能是真冲突。


但：

登录失败：

密码错误
vs 登录地址错误

是不同原因，不是冲突。


---

# 13. Step 11.3 v1 / 换模型测试结果

之前版本存在问题：

错误把：

QCLUSTER-0014
QCLUSTER-0066
QCLUSTER-0139

判成 true conflict。


这些实际上更可能是：

- structured_howto
- multiple_causes
- troubleshooting

而不是 conflict。


QCLUSTER-0008：

默认密码 / URL 存在真正冲突，
应继续 blocked。


QCLUSTER-0074：

只有 temporary + unresolved 临时故障证据，
应 blocked，

但：

has_true_conflict = false


QCLUSTER-0006：

短信开通：

签名备案
+
购买套餐

不能凭空创造：

“备案场景”
vs
“套餐场景”。

Scenario 必须有 Source-grounded 条件。


---

# 14. 当前脚本状态

## scripts/36_classify_mixed_knowledge_structure_v3.py

Step 11.3 v3 Gate。

已修复 validate_scenarios：

原来只统计 role == "scenario" 的 evidence，

导致模型把 scenario evidence 标成
procedure / condition 时，
grounded scenario 被误判为
SCENARIO_GROUNDING_FAIL。

现在只要 evidence 带
scenario_label 或 scenario_evidence，
就必须做 Source Grounding 校验。

Grounding 判定本身没有放宽。

## scripts/37_revalidate_scenario_grounding.py

不调用 LLM。

基于已冻结的 v3 JSONL
重新做 deterministic hard rule 推导，

输出：

output/mixed_structure_classifications_v3_1.jsonl
output/mixed_structure_classifications_v3_1.xlsx

v3 原始输出不被覆盖。

## scripts/38_generate_mixed_candidate_kb.py

Step 11.4 Candidate KB 生成。

支持：

--dry-run 只打印计划
--redo <cluster_ids> 只重做指定 cluster
--recheck 不调用 LLM，只重算 deterministic 判定

## scripts/39_validate_mixed_candidate_grounding.py

Step 11.5 Grounding Validation。

支持：

--dry-run 只跑 deterministic 部分
--only <cluster_ids> 只验证指定 cluster
--candidate <path> 指定 Candidate JSONL
--parse-check 不调用 LLM，检查 schema 与 prompt

## scripts/40_repair_mixed_candidate_grounding.py

Step 11.5.1 deterministic repair。

不调用 LLM。

读取 Step 11.4 冻结 Candidate 与 Step 11.5 validation，
只修正已确认的 summary 级缺陷，
输出新的 Candidate v2：

output/kb_entries_mixed_candidate_v2.jsonl
output/kb_entries_mixed_candidate_v2.xlsx

不会覆盖 Step 11.4 冻结输出。

---

# 15. 当前下一步

Step 11.3（含 v3.1）与 Step 11.4 已冻结。

不要重跑 36 / 37 / 38 / 40。

Step 11.5 v1 与 Step 11.5.1 repair 已完成。

v1 validation 输出：

output/kb_entry_grounding_validations_mixed.jsonl
output/kb_entry_grounding_validations_mixed.xlsx

repaired candidate v2 输出：

output/kb_entries_mixed_candidate_v2.jsonl
output/kb_entries_mixed_candidate_v2.xlsx

现在运行 v2 revalidation：

.venv/bin/python scripts/39_validate_mixed_candidate_grounding.py \
  --candidate output/kb_entries_mixed_candidate_v2.jsonl

输出：

output/kb_entry_grounding_validations_mixed_v2.jsonl
output/kb_entry_grounding_validations_mixed_v2.xlsx

无 LLM 预览：

.venv/bin/python scripts/39_validate_mixed_candidate_grounding.py \
  --candidate output/kb_entries_mixed_candidate_v2.jsonl \
  --dry-run

需要重点检查 Sheet：

- summary
- status_stats
- unit_stats
- flag_stats
- entry_validations
- unit_validations
- invented_elements
- validation_flags
- needs_fix
- manual_review

## v2 预期恢复项

### QCLUSTER-0004

summary 已移除"管理员"，
并把员工账号 UI 细节保留在 scenario unit。

预期 summary 不再出现：

INVENTED_ACTOR
INVENTED_MENU_PATH
INVENTED_BUTTON_NAME

### QCLUSTER-0024

summary 已明确区分：

当前统计逻辑
vs
未来门店自定义规划

预期 INVENTED_TIME 消失。

### QCLUSTER-0014 / QCLUSTER-0033

summary 已按来源场景改写。

预期 SUMMARY_PARTIALLY_GROUNDED 消失或降为可解释。

### QCLUSTER-0094

v1 validator 把"财务人员"误判为 invented actor。

grounded source 原文包含：

财务人员修改结算时间会导致数据不准

以及：

严禁财务人员修改结算时间

39 已增加 deterministic override：

当 location = unit、
element_type = actor、
actor 原文出现在该 unit 引用的
supported_answer / supported_solution 时，
撤销该 invented actor flag。

预期该 false positive 消失。

## 仍然必须 blocked 的项

### QCLUSTER-0021 / QCLUSTER-0066

frozen Step 8.4 已判定两条 same_intent pair：

similarity 0.8942
similarity 0.8921

entry centroid cosine 0.9175。

deterministic 规则会直接给出：

CROSS_ENTRY_SAME_INTENT（hard）
CROSS_ENTRY_NEAR_DUPLICATE（medium）

这两条不允许同时进入正式 KB。

必须在 publishability 之前
做 merge / 主从 决策。

这是 Step 11.6，
不属于 39 v2 revalidation。

### QCLUSTER-0021

TEMPORAL_CAUTION_MISSING（medium）。

1 条 source 是 partial，
但 notes / limitations 没有时效提示。

可在 Step 11.6 / temporal 处理，
不要求 v2 grounding revalidation 解决。

---

# 16. 后续计划

已冻结：

Step 11.3 Mixed Knowledge Structure Gate（v3 + v3.1）

Step 11.4 Mixed Candidate KB（10 entries / 25 units）

进行中：

Step 11.5 v2 Grounding Revalidation

之后仍然必须有：

- cross-entry dedup / merge 决策
  首先处理 QCLUSTER-0021 vs QCLUSTER-0066
- temporal handling
  需要真实 source timestamp，见 18
- final publishability gate

只有 publishability 通过的条目
才能进入正式 KB。

正式 KB 当前仍是 74 条。

Step 11.4 的 10 条只是 Candidate，

禁止直接当作正式 KB 使用。

被 Gate 拦下的 5 个 mixed cluster：

QCLUSTER-0006 insufficient_evidence
QCLUSTER-0008 material temporal
QCLUSTER-0074 material temporal
QCLUSTER-0139 material temporal
QCLUSTER-0194 material temporal

其中 4 个 material temporal cluster

在拿到真实 timestamp 之前保持 blocked。

---

# 17. 尚未处理的重要数据

还有 38 个正式 CRM 文档尚未系统整合。

这些文档后续应作为：

高权威 evidence

主要用于：

- conflict resolution
- temporal version resolution
- current product rule validation
- final publishability


尤其：

conflict 15
temporal_versions 11
mixed 中的 temporal/conflict risk

最终不能只靠聊天记录判断当前版本。


---

# 18. 时间信息原则

后续判断：

current vs historical

不能只看：

temporal_status

需要真实 source timestamps。

当前 issue workbook 不一定直接带 timestamp。

可以通过：

source_candidate_id
→ issue candidates
→ messages

映射回原始聊天 timestamp。


不要根据：

temporary
stable
future_plan

直接推断先后顺序。


---

# 19. Singleton Issues

当前还有：

3598 singleton issues

尚未做最终 KB filtering。

后续需要结合：

- knowledge_value
- confidence
- temporal_status
- answer quality
- Q/A grounding
- resolution
- source authority

判断是否进入候选 KB。

再次强调：

不能要求：

resolution == resolved

才允许进入 KB。


---

# 20. Codex 工作要求

你现在接手的是一个已经运行很久的 pipeline。

不要重新设计整个项目。

必须：

1. 先阅读本文件
2. 检查 scripts/ 和 output/ 是否存在上述文件
3. 不修改已冻结 74 条正式 KB
4. 不重跑冻结步骤
5. 从 Step 11.3 v3 继续
6. 每一步先验证输出，再决定下一步
7. 对 Excel 输出做数据级检查，不只看脚本成功退出
8. 遇到模型误判，优先增加 Grounding / deterministic validation，
   不要单纯反复换模型
9. 不允许模型自行发明 Scenario 条件
10. 不允许 Different Cause 被判成 Conflict


---

# 21. Step 11.3 v3 / v3.1 结果（已冻结）

## v3 运行结果

15 个 mixed cluster：

- generation_ready 9
- blocked 6
- has_true_conflict 0
- conflict_facts 空

structure：

- multiple_causes 6
- structured_howto 3
- insufficient_evidence 2
- troubleshooting 2
- scenario_branches 1
- direct_synthesis 1

temporal：

- material 4
- minor 6
- none 5

v1 的误判已消失：

QCLUSTER-0014 → structured_howto
QCLUSTER-0066 → multiple_causes
QCLUSTER-0139 → multiple_causes

三者都不再被判 true conflict。

QCLUSTER-0074 blocked，且 has_true_conflict = false。

QCLUSTER-0006 判 insufficient_evidence，
没有虚构"备案场景 / 套餐场景"。

QCLUSTER-0033 判 structured_howto。

QCLUSTER-0008 blocked，
但结构是 troubleshooting + material temporal，
不是 true conflict，
conflict_facts 为空。

## v3 的唯一缺陷

QCLUSTER-0004 被误 blocked。

原因是 validate_scenarios
只统计 role == "scenario"，

而模型把 3 条 evidence 都标成 procedure，
同时正确填写了
scenario_label 与 scenario_evidence。

用 36 自己的 contains_grounded_phrase 复核：

3 条 scenario_evidence 全部命中 grounded source，

2 个不同 scenario_label：

客户账号密码重置
员工账号密码重置

## v3.1 结果

scripts/37 不调用 LLM，
只重算 deterministic hard rules：

- unchanged 14
- unblocked_scenario_grounding 1
- changed_review_required 0

最终：

- generation_ready 10
- blocked 5
- hard rule flag 只剩 4 个
  MATERIAL_TEMPORAL_UNRESOLVED

冻结文件：

output/mixed_structure_classifications_v3_1.xlsx

v3 原始输出保持不变，
两个文件都保留。

Grounding 一致性已核对：

15 个 cluster 与 mixed_cluster_audits_v2 完全一致，

37 条 evidence 与
mixed_source_qa_alignments_v2 中
usable_for_kb == true 的 37 条完全一致。


---

# 22. Step 11.4 Mixed Candidate KB（已冻结）

脚本：

scripts/38_generate_mixed_candidate_kb.py

模型：

qwen3.5-flash
temperature 0
enable_thinking False

输出：

output/kb_entries_mixed_candidate.jsonl
output/kb_entries_mixed_candidate.xlsx

结果：

- entries 10
- units 25
- candidate_ok 10
- needs_review 0
- hard flag 0
- source coverage 26 / 26

mode 分布：

- cause_items 5
- howto_sections 3
- scenario_sections 1
- single_answer 1

unit 数量与 Gate 完全对齐：

QCLUSTER-0002 3 cause
QCLUSTER-0021 3 cause
QCLUSTER-0023 3 cause
QCLUSTER-0066 2 cause
QCLUSTER-0094 4 cause
QCLUSTER-0004 2 scenario
QCLUSTER-0024 1 answer
QCLUSTER-0014 2 section
QCLUSTER-0033 2 section
QCLUSTER-0029 3 section

剩余 soft flag：

QCLUSTER-0021 TEMPORAL_CAUTION_MISSING
QCLUSTER-0094 SINGLE_SOURCE_ENTRY

## 生成过程中修掉的两个问题

1. QCLUSTER-0094

Gate 给出 independent_cause_count = 4，

模型只产出 1 个 cause，
并把另外 3 个原因写进 summary_answer。

根因是 prompt 让模型误以为
一条 source 只能对应一个 unit。

已修：

同一条 source 可以支撑多个 unit，
unit 数量不受 source 数量限制，
不允许把 unit 级内容写进 summary / notes。

重做后 4 个 cause 与 source 的
4 条编号内容一一对应。

2. QCLUSTER-0004

scenario unit 的 condition 为空。

已修：condition 必填，
并且要尽量使用 source 原文表述。

重做后 condition 为：

客户账号
员工账号

## 词面 grounding 规则修正

contains_grounded_phrase 现在区分长短语：

>= 6 个规范化字符：
原文命中，或字符覆盖率 >= 0.85

< 6 个规范化字符：
字符覆盖率必须 == 1.0，
并且至少一个连续 2-gram 在原文出现

这样"客户账号"可以被判 grounded，

而"企业微信账号""管理员账号""客账"
仍然会被拒。

## recheck 安全性

--recheck 不改任何 KB 内容字段。

已核对：

10 / 10 entry 的
summary_answer / units / notes /
limitations / source_issue_keys
与 recheck 前完全一致，

只有 QCLUSTER-0004 的判定字段发生变化。


---

# 23. 已知待处理问题

## 1. QCLUSTER-0021 vs QCLUSTER-0066

frozen Step 8.4 中存在两条跨 cluster 的
same_intent pair：

0.8942
0.8921

entry centroid cosine 0.9175。

Step 8.5 使用 conservative complete-link /
clique merging，
所以 same_intent pair 不必然合并，
这是设计行为，不是 bug。

但对 KB 来说，
两条同意图条目不能同时发布。

需要在 publishability 之前做：

merge，或
主从（parent / child），或
只保留一条

## 2. QCLUSTER-0004 summary 的执行者

summary_answer 里的"管理员"
在被引用 source 中不存在。

unit 内容本身是干净的。

Step 11.5 必须抓到这一条。

## 3. 4 个 material temporal cluster

QCLUSTER-0008
QCLUSTER-0074
QCLUSTER-0139
QCLUSTER-0194

在拿到真实 source timestamp 之前
无法判断 current vs historical，

保持 blocked。

映射路径见 18：

source_candidate_id
→ issue candidates
→ messages

## 4. 38 个正式 CRM 文档

仍未整合，见 17。

conflict / temporal 的最终判断
不能只靠聊天记录。

## 5. 3598 singleton issues

仍未做最终 KB filtering，见 19。

---

# 24. Step 11.5 v1 / 11.5.1 结果

## Step 11.5 v1

脚本：

scripts/39_validate_mixed_candidate_grounding.py

输出：

output/kb_entry_grounding_validations_mixed.jsonl
output/kb_entry_grounding_validations_mixed.xlsx

结果：

- grounding_pass 3
- manual_review 2
- needs_fix 5
- unit 25 条中 grounded 24
- partially_grounded 1
- material invented 4

v1 成功抓到：

QCLUSTER-0004 summary 中的
无来源执行者"管理员"

QCLUSTER-0024 summary 把
未来计划时间写成当前规则

QCLUSTER-0021 / QCLUSTER-0066
存在 frozen same_intent 证据

v1 false positive：

QCLUSTER-0094 的"财务人员"

该词原样出现在
supported_answer 与 supported_solution，
不应判为 invented actor。

## Step 11.5.1

脚本：

scripts/40_repair_mixed_candidate_grounding.py

输出：

output/kb_entries_mixed_candidate_v2.jsonl
output/kb_entries_mixed_candidate_v2.xlsx

deterministic repair：

- QCLUSTER-0004 summary
- QCLUSTER-0014 summary
- QCLUSTER-0024 summary
- QCLUSTER-0033 summary

false positive 复核：

- QCLUSTER-0094

完整性：

- entries 10
- units 25
- unit 内容全部未修改
- source_issue_keys 全部未修改
- 其余 5 条 entry 内容字段未修改
- Step 11.4 冻结输出未被覆盖

当前下一步是 39 v2 revalidation，
命令见 15。

---

# 25. 进度更新（2026-09-10 会话）

本节起为最新进度，覆盖上文 15 / 16 中
"当前下一步" 的过期描述。

## Step 11.5 v2（已完成）

`kb_entry_grounding_validations_mixed_v2`：
8 pass / 2 needs_fix
（0021 / 0066 仅为跨条目 same_intent，
内容级 grounding 全部干净）。

## Step 11.6 跨条目合并（已完成并冻结）

脚本：scripts/41_merge_cross_entry_duplicates.py
（deterministic，无 LLM）。

决策：merge，survivor = QCLUSTER-0021，
absorbed = QCLUSTER-0066。

- 选择规则：usable source 多者优先（3 > 2），
  平局取 confidence 高者
- units 逐字保留（3 + 2 = 5），
  summary 由两个 parent 已验证 summary 子句
  deterministic 重编号拼装，
  limitations 为 3 条 risky source 自动生成时效提示
- 输出：kb_entries_mixed_candidate_v3（9 条）
  + cross_entry_merge_decisions.jsonl
- 39 增加 merged-aware（additive）：
  source 按 merged_from_clusters 求并集、
  gate 计数求和、输出后缀通用化
  （原实现 _v3 会覆盖 v1 冻结输出，已修）

## Step 11.6.1 v3 重验证（已完成并冻结）

`kb_entry_grounding_validations_mixed_v3`：
9/9 grounding_pass，合并 0021
5 units / 5 sources 全部 grounded。

## Step 12.1 真实 timestamp 提取（已完成并冻结）

脚本：scripts/42_extract_issue_timestamps.py
（deterministic，无 LLM）。

- 路径：issue_key → source_candidate_id
  → source_message_indexes（messages.jsonl
  1-based 物理行号，与 06 的枚举同构）
  → messages
- 覆盖 4095/4095，0 collision，0 fallback
- 已验证用例：000585#ROW-00638 的
  6966/6967/6968/6972 精确命中问答原话
- 发现 1 处边界泄漏（1/4095）：
  001565#ROW-01768 引用了相邻文档 D0148
  的消息，已确定性排除并记录 mismatches
- 语料时间范围：2024-01-24 → 2026-08-06
- 输出：issue_timestamps.jsonl / .xlsx
  （含 temporal_blockers、candidate_sources）

## Step 12.2 Temporal Validity（已完成并冻结）

脚本：scripts/43_validate_temporal_validity.py
（deterministic 证据 + glm-5.3-flash 判定
 + deterministic 时效门）。

recent 定义：距语料最后证据日 2026-08-06
12 个月内（cutoff 2025-08-06）。

| cluster | final_decision | unblock |
|---|---|---|
| QCLUSTER-0008 | CURRENT_CONFIRMED | 是 |
| QCLUSTER-0074 | INCIDENT_TEMPORARY | 否 |
| QCLUSTER-0139 | INCIDENT_TEMPORARY | 否 |
| QCLUSTER-0194 | STALE_NO_RECENT_CONFIRMATION | 否 |

- 全部通过 quote grounding / 日期合法性 /
  key 合法性 / 决策-事实兼容四道门
- 0008 携带 3 条时效注记（逐字注入 limitations）
- LLM schema 失败有修复重试机制
  （output/temporal_validity_llm_errors.log）；
  confidence 字段可能出现 0.0
  （模型返回 "high" 被归一化为 0，仅影响展示）
- 输出：temporal_validity_decisions.jsonl / .xlsx

## Step 12.3 0008 候选生成（已完成并冻结）

脚本：scripts/44_generate_0008_temporal_candidate.py。

- 0008（troubleshooting）按 cause_items 生成
  4 unit 对应 4 usable source
- 时效注记不由 LLM 生成：
  12.2 注记逐字注入 + historical/temporary
  成员自动提示（limitations 共 6 条）
- entry 带 temporal_validity 溯源字段
- 39 增加 additive 豁免：带 CURRENT_CONFIRMED
  溯源的 entry 豁免 cause_items 严格计数门
  （v3.1 gate 对 troubleshooting 不计数，
  independent_cause_count=0），
  以 info 级 TEMPORAL_UNBLOCK_COUNT_OVERRIDE 留痕
- 输出：kb_entries_mixed_candidate_v4
  （v3 的 9 条逐字保留 + 0008）

v4 验证发现两个新 flag（v3 的 9 条逐字未变，
属验证器波动 + 漏检）：

- QCLUSTER-0029 needs_fix（hard）：
  summary 写"两种路径"但 UNIT 2 有 source
  支撑的第三条路径 —— 真缺陷
- QCLUSTER-0002 manual_review（medium minor）：
  unit 2 一个推断步骤不在 source 原文

## Step 12.3.1 v4 deterministic repair（已完成并冻结）

脚本：scripts/45_repair_v4_candidate.py
（无 LLM，40 同模式）。

- 0029 summary 重写为三条路径
  （clause 2 为 SOURCE 0002446#ROW-02718
  原文连续子串）
- 0002 删除 unit 2 推断步骤，
  保留两条 source 逐字步骤
- 输出：kb_entries_mixed_candidate_v5
  （10 条，其余 8 条逐字未变）

## Step 12.3 v5 重验证（已完成）

`kb_entry_grounding_validations_mixed_v5`：
10/10 grounding_pass，0 hard / 0 medium。

剩余 info 级 flag（不阻断）：

- 0021 ↔ 0094 CROSS_ENTRY_RELATED（0.8314）
- 0008 TEMPORAL_UNBLOCK_COUNT_OVERRIDE

---

# 26. 当前状态与下一步

## Mixed candidate 当前状态

candidate v5 共 10 条 entry，全部通过
grounding validation：

8 条原 mixed + 合并 0021（吸收 0066）+ 0008
（temporal unblock，带 6 条时效 limitations）。

注意：10 条仍是 Candidate，
进入正式 KB 必须通过 final publishability gate。

仍 blocked 的 mixed cluster：
0006（insufficient_evidence）、
0074 / 0139 / 0194（temporal，等正式文档）。

## 后续步骤

1. 整合 38 份正式 CRM 文档作为高权威证据
   （见 17）—— temporal/conflict 的最终裁决
2. Final publishability gate（10 条 candidate）
3. Singleton filtering（3598 条，见 19）

正式 KB 仍为 74 条，未动。

---

# 27. Step 13.1 正式文档盘点（2026-09-10 会话，已冻结）

脚本：scripts/46_inventory_formal_docs.py
（deterministic，无 LLM）。

## 关键更正

交接文档 2 / 17 所说"38 份正式文档"
实际为 21 份正式文档 + 17 份空壳：

- 38 = 全部非聊天解析覆盖的文件
- 17 份是零消息聊天导出（仅一行标题，
  无任何知识内容），不能作为证据使用
- 正式文档为 21 份"衍景CRM*.md"
  （D0277–D0299 区间内）
- 3 份以"衍景CRM"开头但被 Step 3 归类为
  聊天的文件核实为需求沟通群群聊，
  分类正确

## 输出

output/formal_docs_inventory_v1.jsonl / .xlsx

sheets：summary / all_non_chat /
formal_documents / empty_stubs /
heading_outline / date_mentions /
version_mentions / frozen_crosscheck

## 校验

- 21 份 sha256 / filename 与冻结
  document_inventory.xlsx 交叉校验
  0 不一致
- 0 份含聊天格式行（无错分类）

## 正式文档形态结论

- 全部为口语转写文本（培训讲解的
  ASR 转录），0 heading / 0 table / 0 image
- 内容日期 0 处、版本号 0 处，
  mtime 为批量解压时间不可用
  → 无时间锚点，不能独立判定
  current vs historical，
  temporal 仲裁必须与聊天语料
  时间线交叉验证
- 总字符 36,265，单份 300–5,732

## blocked cluster 覆盖评估（关键词扫描）

| cluster | 主题 | 覆盖 | 前景 |
|---|---|---|---|
| 0008 | 密码重置 | D0293 有"初始密码"机制描述；默认密码/123456 0 命中 | 部分可仲裁 |
| 0006 | 开通短信 | D0277/D0299 有短信内容；备案 0 命中 | 备案场景不可仲裁 |
| 0074 | 回访工单录音 | D0290 一键外呼（录音/通话） | 较好 |
| 0139 | 企微车主画像 | D0277（企微 23 处） | 部分可仲裁 |
| 0194 | 登录验证过期 | 验证/过期/验证码基本无覆盖 | 维持 blocked |

用户已确认冻结 Step 13.1。

---

# 28. Step 13.2 正式文档证据解析（已冻结）

脚本：scripts/47_parse_formal_docs.py
（deterministic，无 LLM）。

两层证据单元：

- 段落层 output/formal_doc_paragraphs_v1.jsonl
  978 段（FD-P-00001..00978），
  最小引用单元，带 document_id /
  para_index / line_number / 逐字文本
- 证据块层 output/formal_doc_evidence_chunks_v1.jsonl / .xlsx
  90 chunk（FD-CH-00001..00090），
  连续段落聚合（目标 400 字符，不切段落），
  中位 420 字符，min 44 / max 492

每个 chunk 携带：

- document_id + 行区间 + para 区间 + 逐字文本
- authority = official_training
- form = asr_transcript
- temporal_anchor = none

与聊天证据（messages 体系）完全分开，
权威级高于聊天。

校验（全部通过）：

- 行区间反查原文件逐字重建 90/90
- chunk text == 段落原文精确 join 90/90
- 文件与 13.1 盘点相比 0 变化
- 段落覆盖 0 遗漏 0 重叠
- 过程修复 1 个真 bug：
  chunk_id 曾按文档内计数导致跨文档重复，
  已改为全局顺序分配并复核唯一

---

# 29. Step 13.3 正式文档证据关联（已完成，待冻结确认）

脚本：scripts/48_link_formal_doc_evidence.py
（deterministic 组装 + glm-5.3-flash 判定
 + deterministic 逐字 grounding 门，
 schema 修复重试模式同 43）。

90 个 FD-CH 与 5 个目标 cluster 关联，
争议上下文全部来自冻结输出
（core_question / v3.1 结构判定 / 12.2
 temporal 决策 / 成员聊天 Q&A 截断）。

原则：

- quote 必须是 chunk 原文逐字子串，
  不通过则剔除，全部失败丢弃该 link
- 聊天证据只用于理解争议点，
  禁止写入对文档的引用
- 本步骤只产出证据关联清单，
  不做仲裁决定

输出：

output/formal_doc_evidence_links_v1.jsonl / .xlsx
（summary / links / cluster_notes /
gate_events / dropped_quotes）
错误日志：formal_doc_evidence_links_llm_errors.log

## 结果

| cluster | raw | kept | gate 拦截 | 覆盖结论 |
|---|---|---|---|---|
| 0006 短信开通 | 4 | 4 | 0 | 无直接覆盖（签名备案/套餐开通均未提及），仅使用侧机制与类比背景 |
| 0074 回访录音 | 3 | 1 | 2 link | 仅功能基线（外呼通话应带录音），无 Bug 相关证据 |
| 0139 企微画像 | 3 | 3 | 0 | 画像定义 + 企微授权前置依赖，可部分仲裁 |
| 0194 验证过期 | 1 | 1 | 0 | 仅"谷歌浏览器更稳定"一条，基本无覆盖 |
| 0008 无法登录 | 1 | 1 | 0 | 机制层证据：初始密码由创建者输入 + 员工可自行改密 |

5 个 cluster 均 attempt=1 通过 schema。

数据级校验（独立复核）：

- kept quotes 逐字性 17/17
- dropped quotes 误杀 0
- 0074 被拦 2 条均为模型"洗净化转写"
  （合并句子/去语气词）导致非逐字，拦截正确

## 对后续仲裁的含义

- 0008：文档证据支持"初始密码机制"，
  但"默认密码取值（123456 vs 手机号）"
  仍无文档直接证据
- 0006：备案 vs 套餐关系无文档证据，
  insufficient_evidence 维持
- 0074 / 0194：文档基本无覆盖，
  temporal blocked 大概率维持
- 0139：授权依赖机制可写入候选 KB 的
  cause/排查条件，INCIDENT 判定可复核

---

# 30. Step 13.4 仲裁结案（已完成）

脚本：scripts/49_arbitrate_formal_doc_clusters.py
（deterministic，无 LLM）。

决策表为人工复核结论；
脚本只做三类守卫：

1. 一致性守卫：决策与冻结硬事实兼容
   （temporal unblock / 结构判定 / direct 关联数）
2. 引用守卫：reason 引用的 coverage_note
   与 13.3 输出逐字一致
3. 取证守卫：gate input quote 逐字来自
   冻结 chunk / 13.3 links

## 仲裁结果

| cluster | 仲裁 | 依据 |
|---|---|---|
| 0006 | KEEP_BLOCKED_INSUFFICIENT_EVIDENCE | 无 temporal 记录 + insufficient_evidence + 文档 0 direct |
| 0074 | KEEP_BLOCKED_INCIDENT_TEMPORARY | 12.2 unblock=false + 文档 0 direct |
| 0139 | KEEP_BLOCKED_INCIDENT_TEMPORARY | 12.2 unblock=false；补生成被拒（文档只说授权前置，未说"未授权→空白"因果，补生成即发明条件） |
| 0194 | KEEP_BLOCKED_STALE | 12.2 STALE + unblock=false |
| 0008 | DOC_MECHANISM_CONFIRMED_GATE_INPUT | CURRENT_CONFIRMED + 机制层文档证据 |

## 0008 的 publishability gate 输入

4 条逐字机制确认（FD-CH-00062，D0293）：

- L3 修改密码入口
- L5 点击修改密码可直接修改
- L15 输入给员工一个初始密码
- L17 员工后期自行修改密码

用法限定：仅作为 final publishability
gate 的验证输入；
不修改 v5 条目；
"默认密码取值（123456 vs 手机号）"
仍无文档证据。

## 输出与校验

输出：

output/cluster_arbitration_v1.jsonl / .xlsx
（summary / arbitration / gate_inputs /
regeneration_review）

校验（独立复核通过）：

- coverage_note 逐字一致 PASS
- gate inputs 4/4 逐字 + 行号反查 PASS
- linked_chunks 与 13.3 一致 PASS
- v5 仍 10 条，未被触碰

## Mixed pipeline 当前结案状态

- blocked 定论：0006 / 0074 / 0139 / 0194
  （4 个不再等待文档仲裁，
  文档证据已入档：0006 备案场景不可仲裁、
  0074/0194 无覆盖、0139 补生成被拒）
- candidate：v5 10 条不变，
  待 final publishability gate（0008 带文档确认输入）

下一步：

1. Final publishability gate（v5 的 10 条）
2. Singleton filtering（3598 条，见 19）

---

# 31. Step 14 Mixed Final Publishability Gate（已完成，待冻结确认）

脚本：
scripts/50_validate_mixed_final_publishability.py
（deterministic 组装 + glm-5.3-flash 三 verdict
 + deterministic 守卫，模式同 20/28）。

判定聚焦发布维度
（question-unit 匹配 / scope / 可复用性 /
单次事故是否写成通用规则 / limitations 充分性 /
未决冲突），不重新纠结 quote 级 grounding。

deterministic 守卫（publish 自动降级
manual_review）：非 grounding_pass / hard flags /
material temporal 无 limitations /
0008 缺文档确认 / 合并缺溯源。
本次 0 触发。

## 结果

| verdict | 数量 | 条目 |
|---|---|---|
| publish | 2 | 0004（密码重置）、0033（批量导入潜客） |
| manual_review | 7 | 0002 / 0014 / 0021 / 0023 / 0024 / 0029 / 0094 |
| reject | 1 | 0008 |

10 条全部 attempt=1，raw == final，
v5 未被触碰。

## 关键判定理由

- 0008 reject：4 个 unit 全部是单次会话个案
  （一次性重置 123456 / 手机号作密码个案 /
  已被取代的临时链接 / 等待修复），
  limitations 自身已标注 temporary/historical；
  temporal CURRENT_CONFIRMED ≠ 可复用性，
  两维度独立，gate 判定成立。
  官方文档确认（4 条机制 quote）已归档在
  cluster_arbitration_v1，供未来
  "基于文档重写通用登录排查"时复用
- 0004 publish：两分支稳定产品流程；
  note 提示默认密码 123456 属安全敏感，
  发布侧可考虑展示层脱敏策略（不拦截）
- 0033 publish：两条导入路径清晰
- 0024 manual_review：统计规则稳定但
  "一年有效期将升级门店自定义"属强时效窗口
- 0094 manual_review：SINGLE_SOURCE_ENTRY
  （4 unit / 1 source）需人工核对原文
- 0021 manual_review：原因1-3 稳定，
  原因5（unresolved/temporary）混入期望逻辑，
  建议改写为已知缺陷口径或移除
- 0029 manual_review：unit2 含客户特定示例
  （"如康总"）依赖会话现场材料

## 输出

output/mixed_final_publishability_v1.jsonl / .xlsx
（verdicts / llm_notes / full_records）
错误日志：
mixed_final_publishability_llm_errors.log

## 当前正式 KB 状态

仍为 74 条（71 safe + 3 multicause），
本次 2 条 publish 尚未导出合并。

---

# 32. Step 14.2 正式 KB v2 导出（已完成）

脚本：scripts/51_export_official_kb_v2.py
（deterministic，无 LLM）。

正式 KB v2 = 76 条：

- safe 69（kb_source_quality.xlsx#approved）
- regenerated 2（regenerated_publishability.xlsx#approved）
- multicause 3（kb_entries_multicause_publishable）
- mixed 新增 2：
  QCLUSTER-0004 / QCLUSTER-0033
  （mixed_final_publishability_v1 verdict=publish）

导出结构：统一信封
（kb_id KB-0001..KB-0076 / part / cluster_id /
question / answer / source_issue_keys /
provenance）+ payload 为原生记录逐字内嵌。

守卫（全部通过）：

- 数量 69+2+3+2=76
- cluster_id 全局无重复
- 0008（reject）被排除
- mixed payload 与 v5 逐字一致（2/2）
- safe payload 与冻结源逐字一致（0 不一致）
- mixed 与 legacy 74 无 cluster 重叠

输出（全部新文件，未覆盖任何既有导出）：

output/kb_entries_official_v2.jsonl / .xlsx
（summary / all_entries / safe_entries /
regenerated_entries / multicause_entries /
mixed_entries）

注意：

- 旧文件（kb_entries_safe.jsonl 等）
  仍为原始生成集（safe 文件本身是 127 条
  生成集，不是正式 71 条），
  正式 74 条一直以判定文件为准；
  v2 是首个合并导出
- 7 条 manual_review 保留 candidate 身份，
  等待用户逐条决策（可改写后重审）；
  0008 reject 定论，其 4 条文档机制确认
  归档于 cluster_arbitration_v1，
  供未来"基于文档重写通用登录排查"新条目使用

下一步：

1. manual_review 7 条逐条决策（可选，
   建议放在 singleton filtering 之后统一处理）
2. Singleton filtering（3598 条，见 19）

---

# 33. Step 15 Singleton Filtering（已完成）

## Step 15.1 画像（deterministic）

singleton 名单重建：与 12 同源
make_issue_key（extracted_issues.xlsx
行序）减去 question_clusters#cluster_members
497 条 = 3598，与冻结 summary 一致。

注意：extracted_issues.jsonl 的 issue_id
与 cluster 体系 key 是不同 id 空间，
join 必须走 xlsx + make_issue_key
（52 已内置该逻辑与守卫）。

## Step 15.2 A+B 硬过滤

脚本：52_filter_singleton_funnel_v1.py
（deterministic）。

- A 层：conf<0.7 (462) / knowledge_value=low
  (528) / answer<20字 (349) /
  future_plan (187) / historical (2)
- B 层：last evidence < 2025-08-06 →
  STALE_NO_RECENT_CONFIRMATION (2079)
- 3598 → 1145（unresolved/partial 256 条
  保留，红线守卫内置；
  feature_request 不硬排送 LLM 层）
- 2079 条 stale 全部留档为救回候补
  （未来凭正式文档佐证可救回）
- 输出：singleton_funnel_v1 /
  singleton_survivors_ab_v1

## Step 15.3 C 层 embedding 去重

脚本：53_dedup_singleton_embeddings.py。
AI 配置与 50 一致
（bigmodel / embedding-3，
用户 OPENAI_API_KEY）；
向量缓存带模型标记，
支持 --threshold 重算不重嵌。

- C1 与官方 KB v2 question ≥0.90 → 3 条
  （全部撞 KB-0075 密码重置）
- C2 幸存者内部 ≥0.90 分组留优
  （conf → knowledge_value → 新近）→ 14 条
- 1145 → 1128
- 分离度验证：kept 官方相似度最大
  0.8915，阈值无边缘拉扯
- 输出：singleton_dedup_v1 +
  singleton_dedup_embeddings_cache

## Step 15.4 D 层 LLM 判定

脚本：54_singleton_llm_filter.py
（glm-5.3-flash，10 条/批 113 批，
精确 item_index 全覆盖校验 + 修复重试，
断点续跑 + issue_key 去重）。

判定维度：qa_aligned / reusable /
incident_risk / stable_limitation / kb_worthy。

守卫：candidate 且 qa_aligned=no → reject；
feature_request candidate 须
stable_limitation=true。

结果：

- candidate 858 / reject 270
- qa_aligned: yes 886 / partial 204 / no 38
- unresolved 50 + partial 79 = 129 条
  candidate（红线落实）
- feature_request candidate 2 条
  （均 stable_limitation=true）
- 0 重试 / 0 守卫触发
- reject/candidate 抽验全部成立

## 事故记录（已修复）

54 首版从 53 的 dedup 决策文件直接取条目，
而该文件不携带 answer 字段，
导致第一轮 113 批判定对象为空答案
（LLM 诚实回答"答案为空"）。
已修复：54 从 survivors 文件 join answer
+ 答案非空确定性守卫。
无效输出存档：
singleton_llm_filter_v1.invalid_empty_answers.jsonl。

## Step 15.5 汇总导出

脚本：55_singleton_candidates_export.py
（deterministic，链路一致性守卫全过）。

漏斗全链路：

3598 → A+B 排除 2453 → 1145
→ C 排除 17 → 1128 → D reject 270
→ **candidate 858**

输出：

- singleton_candidates_v1.jsonl / .xlsx
  （858 条，带全字段 + 漏斗溯源）
- singleton_funnel_audit_v1.xlsx
  （funnel / excluded_ab / rejected_d）

重要：858 条是 candidate 池，
不是正式 KB；下一步需独立
publishability gate（同 Step 14 模式）。

## 当前总状态

- 正式 KB：76 条（kb_entries_official_v2）
- mixed：7 条 manual_review 待决策
- singleton：858 条 candidate 待 gate
- stale 救回候补：2079 条（长期）

---

# 34. Step 16 Singleton Publishability Gate + KB v3（已完成）

## Step 16.1 gate

脚本：56_singleton_publishability.py
（glm-5.3-flash，10 条/批 86 批，
配置同 50/54，item_index 覆盖校验 +
修复重试 + 断点续跑）。

singleton 特有判定维度
（在 15.4 之上更严）：

- self_contained：答案脱离聊天上下文
  是否可读（悬空指代检测）
- client_specific_risk：客户专名/
  店名/专属配置
- temporal_risk
- deterministic 前置守卫：
  PRIVACY_PHONE（regex 命中 1 条，
  答案含真实手机号作账号示例，
  强制 manual_review）/ ANSWER_TOO_SHORT

结果（858 条全部判定）：

| verdict | 数量 |
|---|---|
| publish | 569 |
| manual_review | 258 |
| reject | 31 |

- 0 守卫降级（LLM 判定与守卫口径一致）
- 120 条经第 2 次尝试恢复（重试机制正常）
- publish 含 unresolved 24 条 +
  partial 35 条（"当前不支持X"型
  稳定限制知识，红线正确落地）
- publish/reject 抽验全部成立

输出：singleton_publishability_v1
.jsonl / .xlsx

## Step 16.2 官方 KB v3 导出

脚本：57_export_official_kb_v3.py
（deterministic，守卫全过）。

**正式 KB v3 = 645 条**：

- v2 继承 76（safe 69 / regenerated 2 /
  multicause 3 / mixed 2，payload 逐字不变）
- singleton 新增 569

校验：

- kb_id KB-0001..KB-0645 连续唯一
- 全量手机号复扫 0 命中
- v2 前 76 条逐字一致
- singleton issue_key 与 cluster_id
  空间无交集

输出（新文件，未覆盖任何既有导出）：

output/kb_entries_official_v3.jsonl / .xlsx
（summary / all_entries /
singleton_publish）

## 遗留事项

- singleton manual_review 258 条
  保留 candidate 身份待决策
- singleton reject 31 条留审计
- mixed manual_review 7 条待决策
- stale 救回候补 2079 条（长期，
  凭正式文档佐证可救回）
- 0008 文档机制确认 4 条
  （cluster_arbitration_v1），
  可作为未来"基于文档重写通用
  登录排查"新条目的素材

---

# 35. Step 17 RAG 数据层（已完成）

## 17.1 chunks

脚本：58_export_rag_chunks.py。

- output/rag_chunks_v1.jsonl / .xlsx
- 645 chunks（1 KB 条目 = 1 chunk，
  text 42–238 字符）
- embedding 文本 = "问题：{q}\n答案：{a}"
- metadata：part / cluster_id /
  provenance / crm_module / crm_feature /
  problem_type / resolution /
  knowledge_value / temporal_date

## 17.2 向量

脚本：59_embed_rag_chunks.py
（bigmodel embedding-3，2048 维，
断点续跑，用户终端已跑完）。

- output/rag_chunk_embeddings_v1.jsonl
- 索引完整性验证：645/645、
  自相似全 1.0、库内最近邻
  p50=0.777 / max=0.892
  （无一对 ≥0.90，与 C 层去重一致，
  检索阈值区间干净）

## 17.3 检索冒烟测试

脚本：60_rag_smoke_test.py
（numpy 余弦 top-5，查询向量走 API，
用户终端执行，8 样例查询效果确认可用）。

## 17.4 平台导入包

脚本：61_export_platform_import.py。

output/platform_import/：

- import_qa_full.csv（645）
- import_qa_core.csv（核心 76）
- import_qa_singleton.csv（569）
- CSV 表头 question,answer,part,kb_id,
  cluster_id,crm_module,problem_type,
  temporal_date（UTF-8 BOM）

平台接入：

- Dify：导入 CSV + 问答模式
  （识别 question/answer 表头）
- FastGPT：表格导入，
  question 列映射索引、answer 映射答案
- 平台用自身 embedding 重建索引；
  我们 embedding-3 向量保留用于
  自建 API 路线与 golden set 评测
- 建议检索参数：top_k 3-5，
  相似度下限 0.40 起步实测调整

## 客服 AI 后续路线（已与用户对齐）

1. golden set 评测（真实客户问题
   held-out，量化 hit@k / MRR）
2. 服务管线：查询理解 → 检索 →
   带 grounding 生成（附 chunk_id 引用）→
   低相似度兜底（转人工，不编造）
3. 上线形态：Dify/FastGPT（进行中）
   或自建 FastAPI
4. 运营回流：manual_review 清理
   （258+7）、stale 救回、新聊天
   周期性重跑管道

---

# 36. Step V1 培训视频盘点（已完成，2026-09-11 会话）

用户需求：客户问题涉及视频培训过的内容时，
直接把视频链接发给客户。

## 素材现状（video_learning 项目）

视频仓库：/Users/qiyue/Desktop/video_learning
（独立 git 工程，已有完整生产链路：
百炼 ASR → OSS 上传 → videos.json → YJ API 导库）。

- 22 个视频（video-001..022），
  每个视频对应一个功能点，1–17 分钟
- 22 份 SRT 字幕（百炼 ASR + 术语校正），
  共 1055 段 / 规范化 34,308 字符
- metadata.csv 带标题 / 摘要 / 分类
- OSS 公网直链：
  https://aierka-car.oss-cn-hangzhou.aliyuncs.com/
  + help/videos|transcripts|thumbnails/...
  44 个 URL HEAD 校验全部 200

## 关键结论：正式文档 = 视频转写

21 份正式文档与 22 个视频做逐字比对
（shingle 包含 + difflib 兜底）：

- 21/22 视频与 21/21 正式文档
  一一对应，相似度 >= 0.996
- 即 D0277–D0299 就是视频 001–021
  的 ASR 转写（此前 13.1 的判断成立）
- 唯一全新内容：video-022
  《自定义商机功能》（2,397 字符，
  无对应文档）
- 比对曾出现 3 个假 none：ASR 把
  "衍景CRM"写成"Cm"导致 2 字符错位、
  25 字 shingle 全断，difflib 第二 pass
  已兜住（015 修复后 0.998）

## 对知识库状态的修正含义

正式文档内容 = 视频内容，已进证据库
（90 chunk），但如 30/31 所述，
至今 0 条正式 KB 条目来自它们。
"发视频链接"是把这批培训知识
利用起来的最短路径。

## 输出

脚本：scripts/62_video_inventory_v1.py
（deterministic，无 LLM；
--check-urls 做 URL 活性校验）

输出：

output/video_inventory_v1.jsonl / .xlsx
（summary / videos / doc_matches /
unmatched_videos / unmatched_formal_docs /
url_checks）

校验：0 errors / 0 warnings；
metadata.csv ↔ videos.json ↔ SRT
id 集合一致；OSS key 前缀 help/ 全合规。

## 后续规划（未实施）

- Step V2 触发问题生成：
  每视频从字幕+摘要生成客户问法，
  LLM + deterministic grounding 门
- Step V3 向量化：触发问题用 embedding-3
  嵌入（与 rag_chunk_embeddings_v1 同空间），
  阈值标定 + 视频↔KB 条目互挂计算
- Step V4 接入：平台 CSV（Dify/FastGPT
  独立视频库）与/或自建服务管线
  的视频检索源
- 护栏原则：阈值宁缺毋滥、单次最多
  1–2 个视频、video-011（产品价值宣传）
  默认不进客服路由

---

# 37. Step V2–V4 视频检索层（已完成，2026-09-11 会话）

API 配置：项目根 .env（OPENAI_API_KEY，
bigmodel glm-5.3-flash + embedding-3）。
执行方式：`set -a && . ./.env && set +a`。
（事故记录：用户首给的 key 只有 32 位 hex
前半段，401；bigmodel key 完整格式为
`{id}.{secret}`，补全后通过。）

## Step V2 触发问题生成（已完成并冻结）

脚本：scripts/63_generate_video_trigger_questions.py
（glm-5.3-flash，json_object + temperature 0 +
3 次修复重试 + 断点续跑 + --recheck 无 LLM 重判 +
--video 单视频重做）。

- 22 视频全部 attempt=1 一次通过 schema
- 生成 172 条问题，gate v2 通过 162 条
  （22 视频每个 >= 5 条，无零通过视频）
- gate v1 事故与修正：v1 用"问题 vs 被引段落"
  字符 bigram 覆盖率 >= 0.60，误杀 124/172
  （口语改写如"别的店"vs 原文"其他店"）。
  审计确认被拒问题几乎全部真实有据
  （发明特征词应落 0-0.15，尾部实际 0.13-0.6）。
  gate v2 改为：
  a) 全池覆盖率（title+summary+全部字幕）>= 0.40
  b) REFS_UNRELATED：被引段必须与问题共享
     >=1 内容 bigram（防乱引）
  c) 长度 8-40 / 视频内去重 / refs 存在性
  --recheck 只重算判定字段，问题原文逐字不变
- 剩余 10 条 reject（0.27-0.38 边缘改写）
  保守拦下留档，不影响任何视频覆盖
- 输出：video_trigger_questions_v1.jsonl / .xlsx

## Step V3 向量化 / 标定 / 互挂（已完成）

脚本：scripts/64_embed_video_linkage.py
（embedding-3，807 条向量 =
162 视频问题 + 645 KB question，
sha256 内容寻址缓存，断点续跑）。

- 跨视频撞车（>=0.90）仅 2 对，均为真实的
  多视频覆盖同一主题（潜客邀约卡片 005/009、
  客户列表可见性 005/008），非错误
- 视频↔KB 互挂 50 对（阈值 0.80），
  抽验语义对应良好（切门店↔切门店、
  卖车停止提醒↔卖车停止提醒）
- video-018（公海客户）无 KB 互挂：
  该主题聊天 QA 少，属正常
- 标定分布：自视频内聚 p50 0.844；
  异视频噪声带 p50 0.762 / max 0.927；
  KB 相关带 p50 0.787 / max 0.907。
  同视频 vs 异视频间隔仅 ~0.08
  （同域 CRM 主题所致），检索阈值
  上线前必须用真实 query 冒烟校准
- 输出：video_kb_linkage_v1.jsonl / .xlsx +
  向量缓存 video_kb_embeddings_v1.jsonl

## Step V4 平台导入包（已完成）

脚本：scripts/65_export_video_kb_import.py
（deterministic；修复 linkage CSV 的
extrasaction bug 后通过）。

- output/platform_import/import_video_qa.csv：
  162 行 / 22 视频全覆盖，answer =
  【视频讲解】标题+摘要+OSS 链接+时长
  （UTF-8 BOM），校验：无空字段、
  无重复问题、链接全部 https
- output/platform_import/video_route_manifest.json：
  22 视频路由清单，video-011
  include_in_cs_route=false
- output/platform_import/import_video_kb_linkage.csv：
  50 行互挂表

## 后续（未实施）

1. 真实 query 冒烟校准检索阈值（60 模式）
2. Dify/FastGPT 建独立视频知识库导入
   import_video_qa.csv（问答模式）；
   或并入现有 QA 库
3. 自建服务管线按 manifest 接入视频源：
   文字答案后附"相关培训视频"（最多 1-2 个，
   阈值宁缺毋滥），KB 条目按 linkage CSV
   互挂视频

---

# 38. 脚本资产分类索引（2026-09-11 会话）

原则：scripts/ 下 68 个脚本一律不删除。
脚本是冻结输出的审计链，"不会再跑"
不等于"没用"。状态以此节为准，
后续会话不要再从头盘点。

## A. 一次性实验脚本（4 个，无冻结链依赖）

- scripts/03_semantic_duplicate.py
  （早期本地 sentence-transformers 查重实验，
  正式管线为 09–12 bigmodel embedding 聚类）
- scripts/07_compare_thinking.py
- scripts/08_compare_thinking_results.py
  （enable_thinking A/B 对比，结论已落：
  qwen3.5-flash 关 thinking；
  产物 output/thinking_*.xlsx/jsonl 留档）
- scripts/test_qwen_speed.py（测速）

## B. 被新版本取代的 v1 脚本（6 个，留作否决依据）

| 旧版 | 取代者 | 依据 |
|---|---|---|
| 13 | 14 | Step 8.6 |
| 22 | 23 | Step 10.2 |
| 25 | 26 | Step 10.4 |
| 30 | 31 | Step 11.1 |
| 32 | 33 | Step 11.2 |
| 34 | 36 | Step 11.3（35 号空缺） |

## C. 其余 58 个保留，其中明确会再跑的

- 57：manual_review 决策后导出 KB v4
- 58 / 59 / 61：KB 更新后重导
  RAG chunks / 向量 / 平台包
- 53 / 59 / 64：带内容寻址缓存，
  新内容增量嵌入直接续跑
- 60：检索冒烟测试（视频阈值校准
  将仿照此模式新增脚本）
- 62 / 63 / 65：视频层增量维护
  （63 --video 单视频重做 / --recheck 重判）

## 新聊天记录到达时的原则（不要直接顺序重跑）

- 新 markdown 放独立目录（如
  data/raw/markdowns_v2/），老语料永不重跑：
  issue_key / QCLUSTER 按 xlsx 行序推导，
  合并重跑会使全部 ID 移位、冻结引用作废
- 复用骨架（01–12、39、50 模式）但按
  新批次跑：输入路径参数化 + 独立 ID 空间
  （如 B2- 前缀）+ 数量守卫需相应调整
- 新 issue 先与官方 KB 做相似度匹配
  （53 的 C 层逻辑）：>=0.90 不生成新条目，
  记为"近期再次确认"，用于救回 2079 条
  stale 候补与刷新 temporal 状态；
  其余走候选生成 + gate
- 通过后合并导出 KB v4（57 改版），
  重跑 58/59/61 刷新下游

---

# 39. Step V4.1 视频路由阈值校准（已完成，2026-09-17 会话）

对应 37 节"后续（未实施）"第 1 项：
真实 query 冒烟校准检索阈值。

脚本：scripts/66_video_route_calibration.py
（A/C 组全走 64 的内容寻址缓存不调 API，
仅 B 组口语改写增量嵌入 21 条）。

## 查询集（85 条）

- A 真实正例 44：互挂表 KB 问题
  （聊天里真实问过、已与视频互挂，
  50 对去重后 44 条）
- B 口语正例 21：每视频一条客服视角
  口语改写（措辞刻意不同于触发问题；
  video-011 不在客服路由故略）
- C 真实负例 20：645 条 KB 问题中对
  全部触发问题 max_sim<=0.70 的真实
  未覆盖主题，等距抽样
- 索引 154 条触发问题（排除 video-011）

## 分布结论

- 正例命中正确视频 (n=60)：
  min 0.745 / p10 0.801 / p50 0.824
- 负例 (n=20)：p50 0.668 / max 0.699
  —— 0.70 以下完全分开，误触为 0
- 正例 top1 落错视频 5 条（全部 B 组），
  逐一人工核对：2 条为双视频同主题覆盖
  （查客户名下车辆 → video-004/008 均相关；
  与 005/009、005/008 同型），
  3 条为口语改写过泛导致的近似主题误挂
  （0.733–0.813）

## 阈值扫描（关键行）

| 阈值 | 命中率 | 落错视频 | 负例误触 |
|---|---|---|---|
| 0.70 | 0.923 | 5 | 0 |
| 0.76 | 0.908 | 3 | 0 |
| 0.78 | 0.877 | 2 | 0 |
| 0.80 | 0.846 | 2 | 0 |

## 推荐运行点（宁缺毋滥）

- 检索阈值 0.78，只取 top1、单次最多
  1 个视频；top1-top2 差 <0.02 且 top2
  同 >= 阈值时为同主题多视频，可并列
  附第二个
- 视频链接只作为 QA 答案的附属
  "相关培训视频"，不单独成答案
  （对近似主题误挂的低风险兜底）
- video-011 持续排除；
  灰区（0.70–0.80）有 361 条 KB 问题，
  属同域 CRM 正常现象，上线后用真实
  进线 query 持续观察

## 接入路径（未实施，二选一或并行）

- 确定性直挂：50 对互挂表
  （import_video_kb_linkage.csv）——
  命中对应 KB 条目时直接附视频，
  无检索误差
- 检索路由：query → 触发问题 →
  视频（>=0.78 才回），覆盖 KB 外问法

输出：output/video_route_calibration_v1.jsonl
（逐条 86 + summary 含扫描表）

## Step V4.2 发布层组装（已完成，2026-09-17 会话）

脚本：scripts/67_export_video_route.py
（deterministic，无 LLM 无 API，向量全走
64 的内容寻址缓存）。

- output/video_route.jsonl：154 条路由索引
  （manifest 元数据 × 触发问题 × 向量，
  排除 video-011；字段含 content_sha256
  = sha256(question)，同 embeddings 惯例）
- output/video_linkage.jsonl：50 对互挂
- 自校验：https、维度 2048、条数守卫、
  互挂 kb_id 存在于当前快照且 >=0.80

publish() 扩展（incremental_kb/core.py）：
两文件同时存在时复制进 release 并写
manifest["video"] 段（自带 sha256，
不走 SHA256SUMS；yj-kb _validate_manifest
对额外段/文件兼容已核对）。
validate_release 同步校验 video 段哈希
并纳入敏感扫描。任一文件单独缺失即拒发。

1.0.3 = 1.0.2 内容 + 视频层（QA 零变化，
changelog added/revised/removed 全空）。

## 一键发布编排（2026-09-17 会话）

scripts/release.py 串起 publish →
build_release_embeddings → sync_release
三步（不替代守卫，已完成步骤自动检测
跳过，可安全重跑）：

- 默认：本地发布 + 向量挂载 + 远端只读预检
- --apply：含远端暂存/SHA 校验/原子切换/热加载
- --sync-only：本地已完成，只走远端

1.0.3 已于 2026-09-17 用该脚本同步远端：
current 1.0.2 → 1.0.3，/kb/reload 200，
/kb/status 三连抽均为 1.0.3（645 chunk
+ 645 向量）。视频层文件已随行上服务器，
待 yj-kb 侧加载代码后生效。

## 客户可见链接改为应用内地址（2026-09-17 会话）

yj-kb 回复中的视频链接不用 OSS 直链，
改为 xcrm 使用说明页：
https://a.rcar.vip/instruction/video/{video_id}
（67 号脚本 VIDEO_APP_URL_TEMPLATE）。
配套改动：xcrm 路由守卫放行 /instruction
前缀子路径（免登录，xcrm_2 1832f66）；
yj-kb 回复拼接与校验逻辑不变（URL 无关）。
1.0.4 = 1.0.3 内容 + 新链接，已一键发布
并同步远端（/kb/status 确认 1.0.4）。

## 冻结条目修订解禁 + 向量生成路径（2026-09-18 会话）

背景：KB-0004（企微接口许可续费）r1 的
"电脑端打开购买页面，手机企微APP扫码"
把购后授权扫码错误并入购买步骤
（溯源 ISSUE-CAND-001204 博捷汽修群
2025-08-27：付款在前 18:37，扫码是
20:14 之后两个授权二维码）。修复被
publish 守卫与向量挂载两处卡死，本次
只打通管道（①②），不改正文。

① publish 守卫（incremental_kb/core.py）：

- 原"frozen KB identity/revision guard"
  要求 KB-0001..0645 全部 r1，任何
  冻结条目升 r2 即拒发
- 现改为"frozen KB identity guard"：
  只锁 645 个 id 连续唯一不重排，
  revision 可 >= 2；新增全量历史链
  校验（每 kb 的 revision 必须从 1
  连续到 N、仅末位 active 其余
  superseded、rN 的 supersedes 精确
  指向 {kb_id}@r{N-1}，任一断裂拒发）
- 冻结基线文件与旧 release 不可变性
  不变；修订只能走 review 决议产生
  （decide_review 本就支持任意 target）
- 测试 +4：r2 可发布且 changelog 记
  revised、supersedes 断链拒发、
  revision 跳号拒发、缺冻结条目拒发

② 向量挂载（scripts/build_release_embeddings.py）：

- 默认行为不变：纯离线，冻结缓存
  （rag_chunk_embeddings_v1）逐字文本
  匹配，miss 即拒，"never calls an API"
- 新增 --generate-missing：缓存没有的
  修订/新增文本走 embedding API（model
  取 manifest 声明；OPENAI_API_KEY 必填，
  OPENAI_BASE_URL 可覆盖，默认 bigmodel
  paas v4；重试退避同 59 号脚本）
- 生成向量落入内容寻址 supplement 缓存
  output/rag_chunk_embeddings_supplement_v1
  .jsonl（键 content_sha256=sha256(text)），
  之后重跑同版本纯离线可复现
- scripts/release.py 透传同名开关到
  步骤 2/3；测试 +1（生成→落缓存→
  离线复跑全链）

待办（后续会话）：③ 人工核定 KB-0004
r2 完整文案（购买与购后授权两步走，
保留计费/名单/应用授权正确部分）；
④ revision.py revise 人工入口（当前
r2 只能经 ingest+review 产生）；
⑤ release.py --generate-missing --apply
出 1.0.5 并同步远端。

## 服务端快速修订通道落地 yj-kb / yj-crm-kb（2026-09-18 会话）

按用户决策，人工审核+修改+一键发布功能
没有做进本仓库，而是做进了消费端：

- yj-kb：`app/retrieval/kb_editor.py`
  （待发布存储 + 发布器，staging 全量
  自校验后原子切 current + 热加载）+
  `app/api/kb_admin.py`（/kb-admin 路由，
  X-Kb-Admin-Token 门禁，令牌未配置
  即 503 禁用）。修订守卫与本仓库口径
  一致（≥20 字、禁手机号）；修订条目
  向量用 AsyncEmbeddingClient 重新生成。
- yj-crm-kb：KbAdminView.tsx"知识修订"
  视图（搜索/编辑/待发布徽章/一键发布
  确认弹层/令牌设置）。
- 端到端已验证：真实 1.0.4 副本上修订
  KB-0004（两步走文案）→ 发布 1.0.5 →
  645 chunk + 645 向量热加载，旧 release
  字节不变。
- ⚠️ 对账注意：该通道产出的 r2+ 修订
  存在服务器 release 里，不回写本仓库
  registry。本仓库下次 publish 前必须
  先核对服务器 current 版本，否则可能
  版本号冲突或修订被 r1 基线覆盖
  （例如服务器已 1.0.5 时，本仓库应
  以服务器 changelog 为准合并后再发）。
- ④⑤ 因此转为主通道；本仓库 ①② 的
  守卫放开与向量生成路径保留，供
  全量重建场景使用。

## 2026-09-18 生产上线记录（kb-admin 通道）

- yj-kb a6e68b4 + a656d88 已发布
  （deploy_yjkb.sh，服务器 fast-forward
  + supervisor 重启，健康检查通过）；
  KB_ADMIN_TOKEN 按该仓库惯例写入
  被跟踪的 .env（私有 gitee 仓库）。
- yj-crm-kb 3bc7510 已 build+deploy
  （deploy.sh，静态包覆盖即生效）。
- nginx /etc/nginx/conf.d/yj-agent.conf
  新增 /kb/kb-admin/ → yj_kb/kb-admin/
  代理块（改动前备份为 .bak-kbadmin）。
- **服务器 current 已是 1.0.5**：
  KB-0004@r2（两步走修正文案，
  provenance kb-admin:manual-revision），
  645 chunk + 645 向量（服务器侧真实
  embedding 生成）+ 视频层完整。
- ⚠️ 本仓库 registry 仍停在 1.0.4/r1。
  下次全量 publish 前必须先把服务器
  1.0.5 的 changelog/revisions 合并进
  本地 registry（或以服务器为准重建
  基线），否则会版本冲突或覆盖 r2。


# 33. v4 重立基线 + funnel v2（2026-09-20 会话）

用户决策: 不走增量, 改为重立基线 --

- 保留: yj-kb kb-admin 通道修改过的 26 条
  (文本以服务器 1.0.8 为准)
- 退役: 未修改的 615 条 + 服务器已删的 4 条,
  共 619 条 (registry 内保留审计链, status=retired)
- 新知识: 只允许来源 2026-03-01 之后的聊天切片,
  新 id 从 KB-0646 起

已落地:

- scripts/68_rebaseline_v4.py:
  对账 1.0.4 vs 1.0.8 (revised 26 / removed 4 /
  unmodified 615), 干净起点校验 (本地 r1 == 1.0.4
  逐字一致), 产出 output/kb_entries_official_v4.jsonl
  (sha256 a4080d3168903ad330a8156ee6cbba54648fc5a511a363fa7cb58e4873161029)
  + output/kb_official_v4_contract.json,
  registry 升级前自动在线备份 (.state/registry.backup-*)
- core.py: bootstrap/publish 切到 v4 基线;
  publish 冻结守卫改为 contract 驱动 (survivor 必须 active,
  retired 不得 active); revision 链终态允许 retired;
  decide_review 新 id 跳过退役段 (KB-0646 起);
  视频联动优先 video_linkage_v2.jsonl (剔除 41 条死链,
  保留 9 条), 并校验联动 kb_id 全在快照内
- pipeline.py: BASELINE -> v4
- tests: 27 个全过 (含 survivor 缺失 / retired 复活 /
  retired 排除三个新守卫用例)
- funnel v2 (747 after 切片 = 566 复用 778 行 +
  61 切片富集 64 行 + 120 切片新抽取):
  scripts/69 (过滤, 747 守卫), 70 (组装, 三层覆盖守卫),
  71 (LLM 抽取/富集, glm-5.3-flash, 断点续跑),
  72 (Layer A 重算 + C1/C2 去重 + C3 对 26 条 survivor
  查重, embedding-3, 阈值 0.90 同 53),
  73 (LLM 准入, prompt 同 54),
  74 (publishability gate, prompt 同 56),
  75 (导出 kb_entries_incremental_v1.jsonl + registry 导入,
  守卫同 57: 无 pre_flags / 答案>=20字 / 手机号 0)
- scripts/76_release_guard_v4.py: 发布对账硬校验 --
  survivor 文本必须与服务器 1.0.8 逐字一致,
  retired 不得出现在快照, 新增 id 必须 >= KB-0646

发布 1.1.0 的顺序 (严格):

1. funnel v2 跑完 (71->72->73->74->75)
2. git commit (publish 要求 tracked tree 干净)
3. python pipeline.py publish --version 1.1.0
   (本地 current 1.0.4, 远端 current 1.0.8, 1.1.0 均可越过)
4. python scripts/76_release_guard_v4.py --release releases/1.1.0
5. python scripts/release.py --version 1.1.0 --generate-missing --apply
   (补新条目向量 + 远端 staging + current 切换 + 热加载)

注意:

- changelog.json 的 diff 基准是本地上一版 (1.0.4),
  因此会显示 619 removed / 26 revised / N added;
  服务器视角 (1.0.8 -> 1.1.0) 实际只有 removed + added。
  changelog 仅供审计展示, 不参与快照重建。
- 服务器 kb-admin 后续若再出修订, 必须先重跑
  scripts/68 的对账部分 (幂等) 再 publish。

## 33.1 funnel v2 产出（2026-09-20 实跑结果）

- 877 行合并（778 复用 + 64 富集 + 35 新抽取）
- Layer A 重算与 v1 excluded 标志 778/778 一致;
  A 层后 602 行
- 去重: C2 近似 5, C1/C3 存量重复 0 -> 597 行
- LLM 准入: 462 candidate / 135 reject
- publishability gate: **publish 300** / manual_review 153 / reject 9
- 导出 output/kb_entries_incremental_v1.jsonl:
  KB-0646..KB-0945 共 300 条, 已导入 registry (r1 active)
- registry 终态: 971 revisions =
  326 active (26 survivor r2 + 300 新 r1)
  + 619 retired + 26 superseded
- 临时目录演练 publish 1.1.0 通过:
  chunk_count=326, survivor 与服务器 1.0.8 文本 0 差异,
  视频联动 9 条无死链, contract/链式守卫全过

剩余步骤 (需要先 git commit, publish 要求 tracked tree 干净):

    python pipeline.py publish --version 1.1.0
    python scripts/76_release_guard_v4.py --release releases/1.1.0
    python scripts/release.py --version 1.1.0 --generate-missing --apply

注意: 真实仓库的 changelog 会以本地 1.0.4 为基准
(added 326 / revised 26 / removed 619), 属预期。
