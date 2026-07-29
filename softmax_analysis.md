
## Softmax 算子优化分析

### 1. 算子简介

Softmax 是深度学习中最常用的算子之一，作用是把一行任意数值转换成概率分布。计算步骤为：**找最大值（数值稳定）→ 求指数 → 求指数和 → 每个指数除以和**。最终所有输出值都在 0 到 1 之间，且加起来等于 1。

在大模型（如 LLaMA）的推理过程中，Softmax 用于注意力机制的最后一步，将注意力分数转换为权重概率。

---

### 2. 原始代码分析

原始代码来自 `llama.cpp` 仓库的 `softmax_f32` 核函数。核心逻辑分为三步：

```cpp
// 第一步：找最大值
float max_val = -INFINITY;
for (int col = tid; col < ncols; col += block_size) {
    const float v = x[row * ncols + col];
    if (v > max_val) max_val = v;
}
// 共享内存归约找全局最大值（此处省略折半归约代码）

// 第二步：求指数和
float sum_exp = 0.0f;
for (int col = tid; col < ncols; col += block_size) {
    sum_exp += expf(x[row * ncols + col] - max_val);
}
// 共享内存归约求全局指数和（此处省略折半归约代码）

// 第三步：归一化
for (int col = tid; col < ncols; col += block_size) {
    dst[row * ncols + col] = expf(x[row * ncols + col] - max_val) / sum_exp;
}
```

---

3. 发现的性能瓶颈

瓶颈 1：三步全部存在重复访问全局显存。
三步操作都使用 for (int col = tid; col < ncols; col += block_size) 循环，每个线程跳着取多个数据，每次都从很慢的全局显存读取。整个算子对同一块数据重复读取了三次，严重浪费带宽。

瓶颈 2：共享内存数组存在 Bank Conflict 风险。
__shared__ float max_shared[block_size] 和 __shared__ float sum_shared[block_size] 都没有 +1。通常 block_size 是 32 的倍数（如 512），在折半归约时，大量线程会同时访问同一个 Bank 的不同地址，造成排队冲突。

瓶颈 3：存在重复计算。
第二步和第三步都计算了 expf(x[row * ncols + col] - max_val)，同一个指数值被算了两次。

---

4. 优化方案

优化 1：重构取数方式，实现合并访问，消除重复计算。
把三步的 for 循环全部去掉，改为每个线程只取自己负责的那一个数。计算一次 expf(val - max_val) 后，既用于归约求指数和，也用于最后的归一化，避免重复计算。

```cpp
// 改前（三步各有一个 for 循环，重复取数、重复计算）
for (int col = tid; col < ncols; col += block_size) { ... }

// 改后（每个线程只取一次，算一次指数，复用）
float val = (tid < ncols) ? x[row * ncols + tid] : -INFINITY;
// ... 归约找最大值 ...
float exp_val = (tid < ncols) ? expf(val - max_val) : 0.0f;
// ... 归约求指数和 ...
if (tid < ncols) dst[row * ncols + tid] = exp_val / sum_exp;
```

优化 2：给共享内存数组加 +1，避免 Bank Conflict。
两处共享内存数组都增加一个填充位，打破 32 的倍数关系，让线程均匀分散到 32 个 Bank 上。

```cpp
// 改前
__shared__ float max_shared[block_size];
__shared__ float sum_shared[block_size];

// 改后
__shared__ float max_shared[block_size + 1];
__shared__ float sum_shared[block_size + 1];
```

---

5. 优化效果预期

根据在 RTX 4090 上对同类算子的性能基准测试：

优化策略 实测效果
合并访问 vs 非合并访问 快约 9.5 倍
避免 Bank Conflict vs 有冲突 快约 10.9 倍
共享内存归约 vs 全局内存归约 快约 6.4 倍

本次优化用到了 合并访问 + 共享内存归约 + 避免 Bank Conflict + 消除重复计算 四把刀，综合预期 GPU 计算时间可缩短 3-5 倍。

---

6. 用到的优化刀总结

优化刀 在本次改动中的体现
合并访问 把三个 for 循环改成单次取数，相邻线程访问相邻地址
共享内存归约 两次归约（找最大值、求指数和）均在共享内存中完成
避免 Bank Conflict 共享内存数组从 [block_size] 改为 [block_size + 1]
消除重复计算 expf(val - max_val) 只算一次，复用于归约和归一化

---

7. 与 RMS Norm 的对比

维度 RMS Norm Softmax
归约次数 1 次（求平方和） 2 次（找最大值 + 求指数和）
特殊计算 开根号 sqrtf 求指数 expf
归一化方式 除以 RMS 值 除以指数和
用到的优化刀 合并访问 + 共享内存归约 + 避免 Bank Conflict 合并访问 + 共享内存归约 + 避免 Bank Conflict + 消除重复计算

---

8. 个人技能总结

· 熟练使用 CUDA C/C++ 进行算子开发与性能优化。
· 能独立分析真实算子代码（如 llama.cpp），识别合并访问、共享内存、Bank Conflict、重复计算等性能瓶颈。
· 掌握 cudaEvent 精确计时、共享内存归约、避免 Bank Conflict 等性能调优技术。
· 能完成从分析到优化的完整闭环，并输出规范的分析文档。

```

---
