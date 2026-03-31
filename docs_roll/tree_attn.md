# Tree Attention for Training-Time Prefix Sharing

> **开发时间**: 2026-03-28 ~ 2026-03-31  
> **开发者**: 谭泽霖 + AI Assistant (Claude)  
> **仓库**: ROLL (feature/training-time-tree-attention branch)  
> **状态**: 端到端验证通过 ✅

---

## 目录

1. [项目背景与动机](#1-项目背景与动机)
2. [相关工作调研](#2-相关工作调研)
3. [技术方案演进](#3-技术方案演进)
4. [最终方案: KV Expansion + flash_attn_varlen_func](#4-最终方案-kv-expansion--flash_attn_varlen_func)
5. [核心实现细节](#5-核心实现细节)
6. [Bug 修复历程](#6-bug-修复历程)
7. [端到端验证](#7-端到端验证)
8. [性能 Benchmark](#8-性能-benchmark)
9. [文件清单](#9-文件清单)
10. [未来工作](#10-未来工作)

---

## 1. 项目背景与动机

### 1.1 问题

在 ROLL 框架的 Agentic RL 训练中，同一个环境 seed 会产生多条 trajectory（例如 `group_size=8` 意味着 8 条轨迹共享同一个初始 prompt）。这些 trajectory 共享一个很长的 **前缀**（system prompt + environment observation），但后缀（agent 的 response）各不相同。

在标准训练中，每条 trajectory 都独立做 full forward，**前缀被重复计算 N 次**。对于 Sokoban 这类多轮交互环境，前缀可占总 token 数的 30-80%，这意味着大量的冗余计算。

### 1.2 灵感来源

- **MiniMax Forge** 的 Prefix Tree Merging + Magi Attention（声称 40x 加速）
- **Tree Training** (arxiv:2511.00413) — 直接相关的学术工作，报告 8.7x 加速
- **DPO Prefix Sharing** (arxiv:2410.20305) — 开源实现，使用 FlexAttention

### 1.3 目标

在 ROLL 的 Megatron training strategy 中实现 tree attention：
1. 前缀在所有模型层（embedding, FFN, layernorm, attention）只计算一次
2. 保持数学等价性（即 tree attention 的输出与标准独立训练完全一致）
3. 兼容 gradient checkpointing、sequence packing、gradient accumulation 等 Megatron 特性

---

## 2. 相关工作调研

### 2.1 学术论文

| 论文 | 方法 | 加速 | 关键点 |
|------|------|------|--------|
| Tree Training (2511.00413) | 单次 forward + gradient scale tensor | 8.7x | Linearize tree → dense mask |
| DPO Prefix Sharing (2410.20305) | FlexAttention | - | 开源, PyTorch 原生 |
| Prompt Trees (Scaled Cognition) | Tree-structured batch | 70x | 极端 prefix sharing 场景 |

### 2.2 工程参考

- **vLLM**: 推理时已有 prefix caching（APC, Automatic Prefix Caching），但仅用于 KV cache 层面
- **ROLL 现有 sequence packing**: 已有 `cu_seqlens` + `PackedSeqParams` 基础设施

### 2.3 方案对比

在调研中对比了 4 种实现方案：

| 方案 | 优点 | 缺点 |
|------|------|------|
| 1. Two-pass KV cache | 实现简单 | 需额外 KV cache 存储；两次 forward |
| 2. 4D attention mask | 概念清晰 | HuggingFace mask 兼容问题；O(n²) 内存 |
| 3. FlexAttention | PyTorch 原生 | Megatron 使用 TE/FA，不兼容 |
| **4. KV expansion + flash_attn_varlen_func** | **兼容 FA2；自动梯度** | **prefix KV 重复占内存** |

最终选择 **方案 4** —— 直接用 flash_attn 的 varlen 接口，完全兼容 Megatron 的 packed sequence 机制。

---

## 3. 技术方案演进

### Day 1 (2026-03-28): 探索与原型

**上午-下午**: 了解 ROLL 架构，分析 Megatron strategy 的 attention 路径

**晚上**: 实现了两个原型版本：

1. **v1 (`tree_attention.py`)**: Two-pass with KV cache
   - `find_common_prefix_length()`: 检测共享前缀长度
   - `build_prefix_tree()`: 从 grouped sequences 构建 prefix tree
   - `tree_forward_pass()`: 两阶段 forward（prefix KV cache → suffix forward）
   - 20/20 单元测试通过

2. **v2 (`tree_attention_v2.py`)**: Single-pass with custom 4D attention mask
   - Linearize tree → 自定义 4D causal mask → 单次 forward
   - 等价性验证：fp32 max_diff=0.000107 ✅
   - 但 HuggingFace 的 `create_causal_mask` 会覆盖自定义 mask

**关键发现**: 
- 单次 forward + custom attention mask 优于两次 forward + KV cache
- 但 Megatron 用的是 TransformerEngine / FlashAttention，不支持 HF 的 4D mask
- 需要用 `flash_attn_varlen_func` 的 `cu_seqlens` 机制来实现

### Day 2 (2026-03-29): Megatron 集成

**选择方案**: KV expansion + flash_attn_varlen_func

**核心实现** (`tree_attention_megatron.py`):
- `detect_tree_attention()`: 检测 batch 是否为单一 tree group
- `linearize_tree_batch()`: 将 batch 线性化为 `[prefix | suffix_0 | ... | suffix_N-1]`
- `install_tree_attention_hooks()`: Monkey-patch 模型的 attention 和 RoPE
- `_tree_attention_core()`: KV expansion + flash_attn_varlen_func
- `_apply_tree_rotary_pos_emb()`: 自定义 RoPE（suffix 位置从 P 开始）
- `unpack_tree_output()`: 将线性化输出还原为 per-sample 格式

**集成到 Megatron Strategy** (`megatron_strategy.py`):
- 在 `inner_forward_step` 中检测 tree group → linearize → install hooks
- 在 `loss_wrapper` 中 unpack output → per-sample loss 计算
- 在 `forward_backward_func` 返回后清理 hooks

**代码审查**: 发现并修复 6 个 bug：
- CRITICAL: 标准 sequence packing 路径被误删
- CRITICAL: hooks 没有 try/finally 保护
- HIGH: `cu_seqlens_q` 用了 padded 值而非 actual 值
- MEDIUM: PP > 1 未检查（tree attention 不支持 pipeline parallelism）
- LOW: logger.info 改 debug，删除未用变量

**单元测试**: 11/11 通过

**环境阻塞**: 
- 原机器的 megatron-core 0.13.1 太旧（缺 TESpecProvider）
- 升级到 0.15.0 后遇到 optimizer 接口变化，需手动 patch
- TransformerEngine 缺 cuDNN headers 编译不了
- **结论**: 需要有 TE 预装的环境才能跑 Megatron + packed sequence

### Day 3 (2026-03-30): 新环境 + Bug 修复

**新环境**: 2×H200，megatron-core 0.15.0, TE 2.13.0 预装

**Bug 1: RoPE padded vs actual 长度不匹配**
- 问题: `_apply_tree_rotary_pos_emb` 用 padded `cu_seqlens` 切分 tensor，但用 actual 长度索引 `freqs` → 数组越界
- 修复: 按 padded 长度切分 tensor，只对 actual tokens 应用 RoPE，padding 部分保持不变

**Bug 2: RoPE freqs 长度不足**
- 问题: `max_seqlen` 设为 max padded segment length，但 suffix RoPE position 是 `P..P+Si`，当 `P+max(Si) > max_seqlen` 时 freqs 不够
- 修复: `max_seqlen_for_rope = max(max_seqlen, P + max_suffix)`

**Bug 3: gradient checkpoint 时 hooks 已被移除**
- 问题: `loss_wrapper` 在 forward 后立即移除 hooks，但 gradient checkpointing backward 需要重跑 forward → hooks 不在 → 走标准路径 → crash
- 修复: 不在 `loss_wrapper` 移除 hooks，改在 `train_step` 的 `forward_backward_func` 返回后清理

**Code Review (vs Paper)**:
- ✅ 梯度正确性：KV expansion approach 天然处理 prefix 梯度累加
- ✅ Attention 正确性：flash_attn_varlen_func + causal=True 正确实现 attention mask
- ⚠️ Hook lifecycle：多 microbatch 场景下 hooks 可能跨 microbatch 残留（Finding 2）
- ⚠️ Step 2 crash (730 vs 728)：疑似 hook lifecycle 导致

**端到端进展**:
- step 0: ✅ forward + backward 成功
- step 1: ✅ 成功（有 output length mismatch warning）
- step 2: ❌ crash (730 vs 728 shape mismatch)

### Day 4 (2026-03-31): Hook 生命周期彻底修复

**诊断**: 多 microbatch + gradient checkpointing 的组合导致三层嵌套的 hook 状态问题

**问题 1: Thread-local 被后续 microbatch 覆盖**
- `_tree_ctx.meta` 是 thread-local，每个 microbatch forward 时会设置
- gradient checkpointing backward 重跑 forward 时，`_tree_ctx.meta` 已被后续 microbatch 覆盖
- 导致 residual (从正确 microbatch) 和 attention output (从错误 metadata) shape 不匹配

**问题 2: 非 tree microbatch 继承 stale metadata**
- tree microbatch 设置了 `_tree_ctx.meta`
- 紧接的非 tree microbatch 没有清除 → hooks 用 stale metadata 处理标准 batch → crash

**问题 3: hooks 重复安装覆盖 original forward**
- 每个 microbatch 都调用 `install_tree_attention_hooks`
- 第二次安装保存的 "original" 其实是第一次 patch 后的版本 → 清理永远恢复不到真正的 original

**修复（三轮迭代）**:

**Round 1**: Hooks 只安装一次，per-microbatch 更新 `_tree_ctx`
- 效果: 过了更多 step，但 gradient checkpointing 仍然用错误 metadata

**Round 2**: 把 metadata 绑到 `PackedSeqParams` 对象上
- 利用 Megatron 的 gradient checkpointing 闭包捕获机制
- `packed_seq_params._tree_meta` 和 `._tree_cu_pad` 随数据流正确传递
- 效果: 改善但非 tree microbatch 仍有问题

**Round 3**: 新增 `SelfAttention.forward` hook
- 在每层 forward 开始时从 `packed_seq_params._tree_meta` 恢复 `_tree_ctx`
- **始终更新**（设置或清空），确保非 tree microbatch 不会误用 stale metadata
- 这样 RoPE hook 和 DotProductAttention hook 都能看到正确的 per-microbatch metadata
- **效果: 20 步训练零错误** ✅

**关键洞察**: Megatron 的 gradient checkpointing 机制是：
```
TransformerBlock._checkpointed_forward 中 custom_forward 闭包捕获 packed_seq_params
→ tensor_parallel.checkpoint(custom_forward, ..., hidden_states, attention_mask, ...)
→ backward 时重跑 custom_forward
→ packed_seq_params 通过闭包正确传递（不是 checkpoint 的 saved tensor）
→ 因此绑在 packed_seq_params 上的 metadata 也正确传递
```

但 thread-local `_tree_ctx` 不在闭包中，所以 backward 重跑时用的是当前值（已被后续 microbatch 覆盖）。解决方案就是在 `SelfAttention.forward` 开头从 `packed_seq_params` 恢复 `_tree_ctx`。

---

## 4. 最终方案: KV Expansion + flash_attn_varlen_func

### 4.1 核心思想

将同一 tree group 的 N 条 trajectory：
```
sample 0: [prefix | suffix_0 | padding]
sample 1: [prefix | suffix_1 | padding]
...
sample N-1: [prefix | suffix_{N-1} | padding]
```

线性化为单条 packed sequence：
```
[prefix | suffix_0 | suffix_1 | ... | suffix_{N-1}]
```

这样 prefix 在所有层（embedding, FFN, layernorm, attention）只计算一次。

### 4.2 Attention 层处理

在每个 attention 层中：
1. 从线性化的 packed tensor 中提取 prefix 和各 suffix 的 Q, K, V
2. 构建 KV-expanded 的 Q, K, V：
   - Q: `[prefix_q | suffix_0_q | ... | suffix_{N-1}_q]`
   - K: `[prefix_k | prefix_k+suffix_0_k | ... | prefix_k+suffix_{N-1}_k]`
   - V: `[prefix_v | prefix_v+suffix_0_v | ... | prefix_v+suffix_{N-1}_v]`
3. 使用 `flash_attn_varlen_func` 计算 attention，通过不同的 `cu_seqlens_q` / `cu_seqlens_kv` 实现正确的 causal masking：
   - Prefix segment: Q 只 attend to prefix K,V
   - Suffix i segment: Q attend to (prefix + suffix_i) K,V

### 4.3 RoPE 处理

标准 packed sequence RoPE 给每个 segment 从 position 0 开始。但 tree attention 中 suffix 的位置应该从 P 开始：
- Prefix: positions `0, 1, ..., P-1`
- Suffix i: positions `P, P+1, ..., P+Si-1`

通过 monkey-patch `apply_rotary_pos_emb` 实现自定义位置编码。

### 4.4 梯度正确性

KV expansion 使用 `torch.cat([prefix_k, suffix_k_i])` 拼接，autograd 会自动将 N 个 branch 的梯度累加到 `prefix_k` 上。这与标准独立训练中 prefix 接收 N 份独立梯度的效果等价。不需要额外的 gradient scale tensor（论文中的 "gradient restoration" 是针对另一种 dense mask 实现的）。

---

## 5. 核心实现细节

### 5.1 检测 Tree Group

```python
def detect_tree_attention(input_ids, attention_mask, group_ids, min_prefix_ratio=0.05):
    # 1. 检查 batch_size > 1
    # 2. 检查所有 sample 属于同一 group
    # 3. 找到共享前缀长度 P
    # 4. 检查 P / avg_len >= min_prefix_ratio
    # 返回 TreeAttentionMetadata(prefix_len, suffix_lengths, num_branches)
```

### 5.2 线性化

```python
def linearize_tree_batch(input_ids, attention_mask, meta, pad_factor=1):
    # 1. 提取 prefix tokens (来自 sample 0)
    # 2. 提取每个 sample 的 suffix tokens
    # 3. 按 pad_factor 对齐各 segment
    # 4. 拼接: [prefix_padded | suffix_0_padded | ... | suffix_{N-1}_padded]
    # 5. 构建 cu_seqlens (actual) 和 cu_seqlens_padded
    # 6. 构建 PackedSeqParams（含 _tree_meta 和 _tree_cu_pad 附加属性）
    # 7. 构建 tree cu_seqlens (用于 flash_attn)
```

### 5.3 Hook 架构

```
install_tree_attention_hooks(model, meta, cu_seqlens_padded)
├── Patch: megatron.core.transformer.attention.apply_rotary_pos_emb
│   └── 检查 _tree_ctx.meta → 调用 _apply_tree_rotary_pos_emb
├── Patch: SelfAttention.forward (每个 SelfAttention 模块)
│   └── 从 packed_seq_params._tree_meta 恢复 _tree_ctx (或清空)
└── Patch: DotProductAttention.forward (每个 DotProductAttention 模块)
    └── 检查 _tree_ctx.meta → 调用 _tree_attention_core
```

**为什么要 patch 三层**:
- `apply_rotary_pos_emb`: 模块级全局函数，需要 tree-aware RoPE
- `SelfAttention.forward`: 接收 `packed_seq_params`（含 metadata），在 RoPE 之前恢复 `_tree_ctx`
- `DotProductAttention.forward`: 执行实际的 tree attention 计算

### 5.4 Gradient Checkpointing 兼容

```
forward_backward_no_pipelining (Megatron scheduler)
├── forward microbatch 0
│   ├── inner_forward_step → set_tree_context(meta_0, cu_pad_0)
│   └── model.forward → packed_seq_params._tree_meta = meta_0
│       └── TransformerBlock._checkpointed_forward
│           └── custom_forward (captures packed_seq_params via closure)
│               └── TransformerLayer.forward(packed_seq_params=packed_seq_params)
│                   └── SelfAttention.forward → _tree_ctx.meta = psp._tree_meta
├── forward microbatch 1 (non-tree)
│   ├── inner_forward_step → set_tree_context(None, None)
│   └── model.forward → packed_seq_params 无 _tree_meta 属性
│       └── SelfAttention.forward → _tree_ctx.meta = None (清空!)
├── ...forward all microbatches...
├── backward microbatch 7
│   └── gradient checkpointing 重跑 custom_forward
│       └── packed_seq_params 来自闭包 → _tree_meta 是 microbatch 7 的 ✅
├── backward microbatch 0
│   └── gradient checkpointing 重跑 custom_forward
│       └── packed_seq_params 来自闭包 → _tree_meta 是 microbatch 0 的 ✅
└── hooks cleanup
```

### 5.5 Loss 计算

Tree attention 的输出是线性化的 `[1, packed_len, hidden_dim]`。需要 unpack 回 per-sample 格式：

```python
def unpack_tree_output(tree_output, meta, cu_seqlens_padded):
    # tree_output: [1, packed_len, hidden_dim]
    # → [N, original_seq_len, hidden_dim]
    # prefix 部分共享到所有 sample
    # suffix_i 部分填入对应 sample 的 P:P+Si 位置
```

然后每个 sample 独立调用 `loss_func` 计算 loss，最终求和。

---

## 6. Bug 修复历程

### 6.1 总览

| 日期 | Bug | 严重度 | 症状 | 根因 |
|------|-----|--------|------|------|
| 03-29 | 标准 packing 路径被删 | CRITICAL | 非 tree batch 无法训练 | 代码生成遗漏 |
| 03-29 | hooks 无 try/finally | CRITICAL | 异常后 hooks 残留 | 代码生成遗漏 |
| 03-29 | cu_seqlens_q 用 padded 值 | HIGH | FA2 计算结果错误 | actual vs padded 混淆 |
| 03-29 | PP > 1 未检查 | MEDIUM | pipeline parallel 下 crash | 缺少 guard |
| 03-30 | RoPE padded vs actual | HIGH | 数组越界 | padded cu_seqlens 切分 + actual 索引 |
| 03-30 | freqs 长度不足 | HIGH | RoPE 越界 | max_seqlen 未考虑 suffix 绝对位置 |
| 03-30 | 梯度 checkpoint 时 hooks 已删 | CRITICAL | backward crash | hooks 清理时机错误 |
| 03-31 | _tree_ctx 被后续 microbatch 覆盖 | CRITICAL | shape mismatch (1728 vs 1699) | thread-local 不随 grad ckpt 传递 |
| 03-31 | 非 tree microbatch 继承 stale ctx | CRITICAL | shape mismatch (777 vs 761) | _tree_ctx 未清空 |
| 03-31 | hooks 重复安装 | HIGH | original forward 丢失 | 多次 install 覆盖 |

### 6.2 最具启发性的 Bug: Gradient Checkpointing + Thread-Local

这个 bug 花了最长时间修复，也是最有技术深度的。

**表面症状**: `residual + out` shape mismatch —— residual 来自正确的 hidden_states，但 attention output 大小不对。

**调试过程**:
1. Round 1: 以为是 hooks 重复安装 → 改为只安装一次 → 部分修复（过了更多 step）
2. Round 2: 发现 grad checkpoint backward 时 `_tree_ctx` 是错的 → 用 registry 按 packed_len 查找 → 不可靠（不同 microbatch 可能有相同 packed_len）
3. Round 3: 发现 `packed_seq_params` 通过闭包在 grad checkpoint 中正确传递 → 把 metadata 绑到它上面 + SelfAttention hook 恢复 `_tree_ctx` → **彻底修复**

**核心教训**: 在有 gradient checkpointing 的系统中，不能依赖 thread-local / global state 传递 per-data metadata。必须将 metadata 绑到随数据流传递的对象上。

---

## 7. 端到端验证

### 7.1 配置

```yaml
# examples/qwen2.5-0.5B-agentic/tree_attn_test_sokoban.yaml
pretrain: Qwen2.5-0.5B-Instruct
actor_train:
  model_args:
    use_tree_attention: true
    attn_implementation: fa2
  training_args:
    per_device_train_batch_size: 2
    gradient_accumulation_steps: 8
  strategy_args:
    strategy_name: megatron_train
    strategy_config:
      recompute_granularity: full  # gradient checkpointing
max_steps: 20
train_env_manager:
  num_env_groups: 2
  group_size: 8  # 8 条 trajectory 共享前缀
  tags: [SimpleSokoban]
```

### 7.2 结果

| 指标 | 结果 |
|------|------|
| 总 step 数 | 20/20 ✅ |
| 运行时错误 | 0 |
| Shape mismatch 警告 | 0 |
| Val score (step 0) | -2.25 |
| Val score (step 10) | -1.95 |
| 每 step 训练时间 | ~1.5s |

### 7.3 环境

- **硬件**: 2× NVIDIA H200 (143GB each)
- **软件**: megatron-core 0.15.0, TransformerEngine 2.13.0, flash_attn 2.7.4, PyTorch 2.6.0
- **GPU 分配**: GPU 0 = actor_train (Megatron), GPU 1 = actor_infer (vLLM)

---

## 8. 性能 Benchmark

### 8.1 Qwen2.5-0.5B-Instruct Forward Pass

使用 `scripts/tree_attention_megatron_benchmark.py` 测试：

| Config | N (branches) | Prefix Len | Standard (ms) | Tree (ms) | Speedup |
|--------|-------------|------------|---------------|-----------|---------|
| Short prefix | 4 | 68 | 23.5 | 11.9 | **1.97x** |
| Short prefix | 8 | 68 | 14.1 | 11.1 | **1.27x** |
| Long prefix | 4 | 272 | 14.9 | 11.8 | **1.26x** |
| Long prefix | 8 | 272 | 15.7 | 12.1 | **1.29x** |

> **注**: 这是单次 forward pass 的 benchmark。实际训练中的加速取决于 prefix sharing ratio 和 batch 构成。

### 8.2 加速分析

- **Token 节省**: `(N-1) × P` tokens per batch（前缀不再重复计算）
- **内存开销**: KV expansion 使每个 attention 层的 K,V 增加 `(N-1) × P` tokens
- **最佳场景**: 长前缀 + 少分支（prefix_ratio 高）
- **最差场景**: 短前缀 + 多分支（KV expansion 开销超过节省）

---

## 9. 文件清单

### 9.1 核心代码

| 文件 | 行数 | 说明 |
|------|------|------|
| `roll/utils/tree_attention_megatron.py` | 575 | 核心模块：detect, linearize, hooks, attention, unpack |
| `roll/distributed/strategy/megatron_strategy.py` | +100 行 | 集成：inner_forward_step, loss_wrapper, hooks cleanup |
| `roll/configs/model_args.py` | +11 行 | 配置：`use_tree_attention` 字段 |

### 9.2 测试

| 文件 | 测试数 | 说明 |
|------|--------|------|
| `tests/utils/test_tree_attention_megatron.py` | 11 | 单元测试 + GPU 端到端 |

### 9.3 早期原型（保留作参考）

| 文件 | 说明 |
|------|------|
| `roll/utils/tree_attention.py` | v1: Two-pass KV cache |
| `roll/utils/tree_attention_v2.py` | v2: Single-pass 4D mask |

### 9.4 Benchmark 与文档

| 文件 | 说明 |
|------|------|
| `scripts/tree_attention_megatron_benchmark.py` | Megatron forward pass benchmark |
| `scripts/tree_attention_perf_benchmark.py` | 性能对比 benchmark |
| `scripts/tree_attention_precision_report.py` | 数值精度报告 |
| `docs_roll/tree_attention_design.md` | 设计文档 |
| `docs_roll/related_work.md` | 相关工作综述 |
| `examples/qwen2.5-0.5B-agentic/tree_attn_test_sokoban.yaml` | 端到端测试配置 |

### 9.5 Git

- **Branch**: `feature/training-time-tree-attention`
- **Commits**:
  - `b351299`: fix: tree attention hook lifecycle for multi-microbatch + gradient checkpointing

---

## 10. 未来工作

### 10.1 性能优化

- [ ] **Triton kernel 替代 KV expansion**: 避免 prefix K,V 重复占内存，直接在 kernel 中读取共享的 prefix KV
- [ ] **长序列场景 benchmark**: 更长 prefix → 更大加速比（当前测试 prefix 较短）
- [ ] **多 tree group 支持**: 当前要求整个 batch 属于同一 group，未来可支持 batch 中多个 group 各自做 tree attention

### 10.2 正确性验证

- [ ] **Loss 曲线对比**: 对比 tree attention 和标准训练的 loss 曲线，验证数值等价性
- [ ] **长训练验证**: 100+ step 训练验证稳定性
- [ ] **DeepSpeed 支持**: 当前只支持 Megatron strategy，可扩展到 DeepSpeed

### 10.3 学术方向

- [ ] **与 Tree Training (2511.00413) 的差异化**: 我们的 KV expansion 方案 vs 论文的 dense mask + gradient restoration
- [ ] **Agentic RL 特有优势**: 多轮交互环境中 prefix sharing ratio 更高
- [ ] **Profile 分析**: 使用 ncu/nsys 分析瓶颈，量化 compute vs memory 的 tradeoff

---

## 附录 A: 关键数据结构

### TreeAttentionMetadata

```python
@dataclass
class TreeAttentionMetadata:
    prefix_len: int              # P
    suffix_lengths: List[int]    # [S0, S1, ..., S_{N-1}]
    num_branches: int            # N
    total_len: int               # P + sum(Si)
    cu_seqlens_q: Tensor         # [0, P, P+S0, P+S0+S1, ...]
    cu_seqlens_kv: Tensor        # [0, P, P+P+S0, P+P+S0+P+S1, ...]
    max_seqlen_q: int
    max_seqlen_kv: int
    original_seq_len: int        # padded seq_len of original batch
```

### Hook 注册表

```
_tree_ctx (threading.local):
  .meta: TreeAttentionMetadata | None     # 当前 microbatch 的 metadata
  .cu_seqlens_padded: Tensor | None       # 当前 microbatch 的 padded cu_seqlens
  .meta_registry: dict[int, tuple]        # packed_len → (meta, cu_pad) 查找表

PackedSeqParams (附加属性):
  ._tree_meta: TreeAttentionMetadata      # 绑定到数据流，随 grad ckpt 闭包传递
  ._tree_cu_pad: Tensor                   # 绑定到数据流
```

---

## 附录 B: 开发环境注意事项

1. **容器重启后 `/proc` 可能丢失** — 不是 Ray 的问题，是容器 procfs 未挂载。CUDA 需要 `/proc/driver/nvidia/` 才能初始化。
2. **避免 `ray stop --force`** — 可能导致 `/proc` 丢失。
3. **megatron-core 版本** — 需要 0.15.0+，0.13.1 缺少 TESpecProvider。
4. **TransformerEngine** — 必须预装，运行时编译需要 cuDNN headers。
