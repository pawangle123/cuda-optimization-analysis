

RMS Norm 算子优化分析

1. 算子简介

RMS Norm（Root Mean Square Normalization，均方根归一化）是大模型（如 LLaMA）中高频使用的归一化算子。它对每一行输入数据进行计算，将数值拉回到一个标准范围内，防止模型训练和推理时数值不稳定。

计算步骤为：求平方和 → 求平均值 → 开根号 → 每个数除以 RMS 值。

---

2. 原始代码分析

原始代码来自 llama.cpp 仓库的 rms_norm_f32 核函数。核心逻辑如下：

```cpp
// 第一步：每个兵跳着取数据，求平方和
for (int col = tid; col < ncols; col += block_size) {
    tmp += x[row * ncols + col] * x[row * ncols + col];
}

// 第二步：共享内存归约求平方和
__shared__ float sum[block_size];
sum[tid] = tmp;
__syncthreads();
for (int s = block_size / 2; s > 0; s >>= 1) {
    if (tid < s) sum[tid] += sum[tid + s];
    __syncthreads();
}

// 第三步：开根号求 RMS 值
float rms = sqrtf(sum[0] / (float)ncols + eps);

// 第四步：归一化（再次跳着取、跳着写）
for (int col = tid; col < ncols; col += block_size) {
    dst[row * ncols + col] = x[row * ncols + col] / rms;
}
```

---

3. 发现的性能瓶颈

瓶颈 1：重复多次访问全局显存。
第一步和第四步的 for (int col = tid; col < ncols; col += block_size) 循环，让每个线程跳着取多个数据，每次都从很慢的全局显存读取，整个算子对同一块数据重复读取了两次。

瓶颈 2：共享内存数组存在 Bank Conflict 风险。
__shared__ float sum[block_size] 中没有 +1。通常 block_size 是 32 的倍数（如 512），在折半归约时，大量线程会同时访问同一个 Bank 的不同地址，造成排队冲突。

---

4. 优化方案

优化 1：重构取数方式，实现合并访问。
把跳着取数的 for 循环，改为每个线程只取自己负责的那一个数，相邻线程访问相邻内存地址。

```cpp
// 改前
for (int col = tid; col < ncols; col += block_size) {
    tmp += x[row * ncols + col] * x[row * ncols + col];
}

// 改后
float val = (tid < ncols) ? x[row * ncols + tid] : 0.0f;
float sum_sq = val * val;
```

优化 2：给共享内存数组加 +1，避免 Bank Conflict。
数组宽度从 512 变成 513，打破 32 的倍数关系，让线程均匀分散到 32 个 Bank 上。

```cpp
// 改前
__shared__ float sum[block_size];

// 改后
__shared__ float sum[block_size + 1];
```

---

5. 优化效果预期

根据在 RTX 4090 上对同类算子的性能基准测试：

优化策略 实测效果
合并访问 vs 非合并访问 快约 9.5 倍
避免 Bank Conflict vs 有冲突 快约 10.9 倍
共享内存归约 vs 全局内存归约 快约 6.4 倍

本次优化用到了 合并访问 + 共享内存归约 + 避免 Bank Conflict 三把刀，综合预期 GPU 计算时间可缩短 3-5 倍。

---

6. 用到的优化刀总结

优化刀 在本次改动中的体现
合并访问 把 for 循环改成 val = x[row * ncols + tid]，每个线程只访问一次全局显存
共享内存归约 全班数据搬进共享内存进行折半归约，避免反复读取全局显存
避免 Bank Conflict 共享内存数组从 [block_size] 改为 [block_size + 1]，打破 32 的倍数

---

7. 个人技能总结

· 熟练使用 CUDA C/C++ 进行算子开发与性能优化。
· 能独立分析真实算子代码（如 llama.cpp），识别合并访问、共享内存、Bank Conflict 等性能瓶颈。
· 掌握 cudaEvent 精确计时、共享内存归约、避免 Bank Conflict 等性能调优技术。
· 能在 NVIDIA GPU 上完成从分析到优化的完整闭环。
