> **johnkai AI 教學系列 — 第四篇**

# LLM 推理優化的完整旅程：怎麼讓模型又快又便宜

---

# 第 0 章：這份文件到底要幹嘛

Doc2 帶你走過了一段文字穿越 LLM 的完整旅程——從 Tokenization 到 Embedding 到 Transformer Block 到 Sampling，每一步都精確追蹤了資料的形狀和矩陣的維度。

但那個過程有一個隱含的假設：**一次只處理一個請求，不管速度和成本。**

現實中，一個 LLM 服務可能每秒收到幾千個請求。一個 70B 的模型光是權重就佔 140 GB——一張 GPU 根本放不下。即使模型能放下，逐 Token 生成的 autoregressive loop 讓 GPU 大部分時間在等資料、而不是在算。

這份文件拆解的就是：**怎麼解決這些問題——讓推理又快又便宜。**

每個優化技術都從一個具體的瓶頸出發：裝不下→量化、算太慢→FlashAttention、快取太大→KV Cache 壓縮、逐步太慢→推測解碼、利用率低→動態 Batching、一張不夠→多 GPU 並行。

前置知識：讀完文件二《LLM Token 生成的完整旅程》，特別是 KV Cache、Prefill/Decode 階段、Compute-Bound 與 Memory-Bound 的概念。

---

# 第 1 章：你會學到什麼

這份文件會讓你真的看懂以下這些東西：

- 為什麼把 FP16 權重量化到 INT4 能省 75% 記憶體，而品質損失幾乎可忽略
- 為什麼 FlashAttention 不是「近似注意力」而是「精確注意力的 IO 優化」
- 為什麼 MLA（Multi-head Latent Attention）能把 KV Cache 壓縮 93%
- 為什麼推測解碼能保證輸出與原始模型完全一致
- 為什麼 PagedAttention 能讓 GPU 記憶體利用率從 ~50% 提升到 ~95%
- 為什麼 Tensor Parallelism 適合節點內、Pipeline Parallelism 適合節點間
- 為什麼一張 RTX 4090 就能跑 70B 模型（量化之後）

這份文件會用這些故事來教你：

1. **一張 GPU 的極限挑戰**
   會帶你走過：
   - 模型太大裝不下（量化壓縮）→ 注意力計算太慢（IO 感知優化）→ KV Cache 隨序列爆炸（壓縮技術）→ 逐 Token 生成太慢（推測解碼）→ 單請求利用率太低（動態 Batching）→ 一張 GPU 不夠用（多卡並行）

讀完故事後，你會看到這些真實案例：

1. **部署一個 70B 模型的完整方案**
   你應該不靠額外說明就能看懂：
   - 用什麼量化格式、多少張 GPU、怎麼切分模型
   - 每個優化技術帶來多少加速和記憶體節省
   - 不同配置下的 throughput 和 latency 的 trade-off

---

# 第 2 章：推理的瓶頸在哪？

Doc2 介紹過 Prefill 階段是 Compute-Bound、Decode 階段是 Memory-Bound。這一章深入分析這兩種瓶頸。

---

## GPU 的兩種資源

```
GPU 有兩種關鍵資源：

1. 算力（Compute）
   衡量單位：FLOPS（每秒浮點運算次數）
   A100：312 TFLOPS（FP16）
   H100：989 TFLOPS（FP16）

2. 記憶體頻寬（Memory Bandwidth）
   衡量單位：GB/s（每秒能讀取多少資料）
   A100：2,039 GB/s
   H100：3,352 GB/s
```

一個計算任務的瓶頸取決於**算術強度（Arithmetic Intensity）**：

```
算術強度 = FLOPs / Bytes

即：每讀取一個 byte 的資料，需要做多少次浮點運算。
```

### Roofline 模型

```
                         ┌─── 算力上限（312 TFLOPS）
Performance              │
(TFLOPS)                 │
  ↑                      │
  │              ╱───────┤
  │             ╱        │
  │            ╱         │  ← Compute-Bound
  │           ╱          │    （算力是瓶頸）
  │          ╱           │
  │         ╱            │
  │        ╱  ← Memory-  │
  │       ╱    Bound     │
  │      ╱   （頻寬是    │
  │     ╱     瓶頸）     │
  │    ╱                 │
  │   ╱                  │
  └──┴────────┴──────────┴───→ 算術強度（FLOPs/Byte）
              ↑
          拐點 = 算力 / 頻寬
          A100：312T / 2039 ≈ 153 FLOPs/Byte

拐點的含義：
  算術強度 < 153 → Memory-Bound（GPU 在等資料）
  算術強度 > 153 → Compute-Bound（GPU 在忙計算）
```

### Prefill vs Decode 的算術強度

```
Prefill（一次處理所有輸入 Token）：
  矩陣乘法：[seq_len × d] × [d × d]
  FLOPs ≈ 2 × seq_len × d × d
  Bytes ≈ d × d × 2（讀取權重矩陣，FP16）

  算術強度 ≈ 2 × seq_len × d² / (d² × 2) = seq_len

  seq_len = 1024 → 算術強度 ≈ 1024 >> 153 → Compute-Bound ✓
  GPU 在全速計算，頻寬不是瓶頸。

Decode（每步只處理 1 個 Token）：
  矩陣乘法：[1 × d] × [d × d]
  FLOPs ≈ 2 × 1 × d × d = 2d²
  Bytes ≈ d × d × 2（同樣要讀整個權重矩陣）

  算術強度 ≈ 2d² / (d² × 2) = 1

  算術強度 = 1 << 153 → Memory-Bound ✓
  GPU 幾乎全部時間在等資料從 HBM 搬到計算單元。
  計算單元只用了 1/153 ≈ 0.65% 的能力。
```

**這就是 Decode 階段慢的根本原因**：每生成一個 Token，GPU 都要讀取整個模型的權重（幾十 GB），但只做一次向量-矩陣乘法。GPU 的算力幾乎完全浪費了。

---

## 優化策略的分類

```
不同的瓶頸需要不同的優化策略：

Memory-Bound（Decode 階段）：
  → 減少要讀的資料量：量化（權重從 FP16 → INT4，讀取量減半再減半）
  → 減少重複讀取：KV Cache（Doc2 已介紹）
  → 一次讀取服務多個請求：Batching
  → 一次猜多個 Token：推測解碼

Compute-Bound（Prefill 階段）：
  → 減少計算量：稀疏注意力、線性注意力（Doc2 的 DeltaNet）
  → 優化計算效率：FlashAttention（減少 IO，讓計算不被打斷）
  → 堆更多算力：多 GPU 並行

兩者都受限的：
  → KV Cache 的記憶體佔用（同時影響容量和頻寬）
  → 模型太大放不下一張 GPU（既影響儲存也影響計算切分）
```

> Decode 階段的瓶頸是記憶體頻寬——GPU 花大部分時間在讀資料而不是算。Prefill 階段的瓶頸是算力——真正在全速計算。理解這個區分是所有推理優化的基礎。

---

# 第 3 章：把模型塞進更小的記憶體——量化

## 量化的動機

```
Llama 3.1 70B（FP16）：
  權重：70B × 2 bytes = 140 GB
  一張 A100（80 GB）：放不下
  一張 H100（80 GB）：放不下
  需要至少 2 張 GPU

如果能把權重壓到 INT4（每個參數 0.5 byte）：
  權重：70B × 0.5 bytes = 35 GB
  一張 A100（80 GB）：放得下！還剩 45 GB 給 KV Cache 和 Activation

量化不只省記憶體——在 Decode 階段（Memory-Bound），
讀取量從 140 GB 降到 35 GB = 速度提升近 4 倍。
```

---

## 數值格式一覽

```
┌──────────┬──────┬───────────┬────────────────┬────────────────────┐
│ 格式     │ Bits │ 每參數大小 │ 70B 模型大小   │ 精度               │
├──────────┼──────┼───────────┼────────────────┼────────────────────┤
│ FP32     │ 32   │ 4 bytes   │ 280 GB         │ ~7 位有效數字      │
│ FP16     │ 16   │ 2 bytes   │ 140 GB         │ ~3-4 位有效數字    │
│ BF16     │ 16   │ 2 bytes   │ 140 GB         │ ~2-3 位有效數字    │
│ FP8      │ 8    │ 1 byte    │ 70 GB          │ ~2 位有效數字      │
│ INT8     │ 8    │ 1 byte    │ 70 GB          │ 256 個離散值       │
│ INT4     │ 4    │ 0.5 bytes │ 35 GB          │ 16 個離散值        │
│ NF4      │ 4    │ 0.5 bytes │ 35 GB          │ 16 個非均勻離散值  │
└──────────┴──────┴───────────┴────────────────┴────────────────────┘
```

---

### Weight-Only 量化 vs Weight-Activation 量化

```
Weight-Only（W4A16）：
  只量化權重，啟動值（Activation）保持 FP16
  推理時：
    1. 從記憶體讀取 INT4 權重
    2. 反量化到 FP16
    3. 跟 FP16 的 Activation 做矩陣乘法
  好處：記憶體省、Decode 更快（因為讀取量少）
  壞處：反量化有額外計算開銷

Weight-Activation（W8A8）：
  權重和啟動值都量化到 INT8
  推理時：
    1. 讀取 INT8 權重
    2. 把 FP16 Activation 量化到 INT8
    3. 用 INT8 × INT8 硬體加速做矩陣乘法
  好處：可以用 INT8 Tensor Core（更快的硬體）
  壞處：啟動值量化可能損失更多精度

2025-2026 的共識：
  W4A16 → 適合 Decode（Memory-Bound，減少讀取量最重要）
  W8A8 → 適合 Prefill（Compute-Bound，用 INT8 硬體加速更重要）
```

---

### 量化的數學原理

**均勻量化（Uniform Quantization）**

```
把浮點數映射到等間隔的整數：

量化：q = round(x / s) + z
反量化：x̂ = (q - z) × s

其中：
  s = 縮放因子（scale）
  z = 零點偏移（zero-point）

以 absmax INT8 量化為例（對稱量化，z = 0）：
  一組權重 x = [0.312, -0.876, 0.045, 0.567, -0.234]
  |max| = 0.876
  s = |max| / 127 = 0.876 / 127 = 0.006898
  q = round(x / s) = round([45.2, -127.0, 6.5, 82.2, -33.9])
    = [45, -127, 7, 82, -34]

  儲存：5 個 INT8 值 + 1 個 FP16 scale = 5 + 2 = 7 bytes
  原始：5 個 FP16 = 10 bytes
  壓縮比：30%

反量化：
  x̂ = q × s = [0.310, -0.876, 0.048, 0.566, -0.235]
  vs 原始   = [0.312, -0.876, 0.045, 0.567, -0.234]
  誤差很小 ✓
```

**分組量化（Group Quantization）**

```
問題：如果一組權重中有 outlier（異常大的值），
scale 會被 outlier 主導，其他正常值的量化精度變差。

  x = [0.01, 0.02, -0.03, 0.01, ..., 15.7]
                                      ↑ outlier
  s = 15.7 / 127 = 0.124
  正常值 0.01 / 0.124 = 0.08 → round(0.08) = 0 → 量化為 0！

解法：把權重分成小組，每組有獨立的 scale：

  Group Size = 128（每 128 個權重一組）
  每組計算自己的 |max| 和 s
  正常值的組不會被 outlier 影響

  代價：每 128 個權重多存一個 FP16 的 scale
  額外開銷：2 bytes / 128 = 0.016 bytes/參數（可忽略）
```

---

### GPTQ（Gradient Post-Training Quantization）

**為什麼需要它**

簡單的 round-to-nearest 量化對 INT4 來說精度損失太大。GPTQ 用更聰明的方式做量化，讓 INT4 的效果接近 FP16。

**核心思想：逐列量化 + 誤差補償**

```
問題設定：
  原始權重矩陣 W（FP16），形狀 [d_out × d_in]
  量化後的矩陣 Ŵ（INT4）
  目標：最小化 ||WX - ŴX||²（讓量化後的輸出盡可能接近原始輸出）
  X 是校準資料集的啟動值

GPTQ 的做法（基於 Optimal Brain Quantization）：

  逐列量化：一次量化一列（一個輸出維度）

  對每一列 i：
    1. 用 round-to-nearest 量化第 i 列
    2. 計算量化引入的誤差 δ = w_i - ŵ_i
    3. 把這個誤差「分攤」到還未量化的列上：
       w_j += δ × H⁻¹_ij / H⁻¹_ii （j > i）
       H = Xᵀ X（Hessian 矩陣，衡量每個權重的重要性）

  直覺：
    量化第 i 列不可避免地引入誤差。
    但你可以微調後面還沒量化的列，來「抵消」這個誤差。
    Hessian 矩陣告訴你怎麼分攤最有效——
    重要的權重（對輸出影響大的）分到更多的補償。
```

**GPTQ 的效果**

```
以 Llama 2 70B 為例：
  FP16 PPL（Perplexity）：3.32
  GPTQ INT4（group_size=128）PPL：3.37
  簡單 round-to-nearest INT4 PPL：4.15

  GPTQ 的精度損失 < 2%，而簡單量化損失 > 25%。

量化時間：
  校準資料：128 條文本
  70B 模型：~4 小時（單張 A100）
  只需要做一次，之後所有推理都用量化後的模型。
```

---

### AWQ（Activation-Aware Weight Quantization）

**為什麼需要它**

AWQ 比 GPTQ 更快（校準時間短）、效果相當或更好。

**核心思想：保護重要權重**

```
GPTQ 的思路：量化後補償誤差
AWQ 的思路：量化前就保護重要的權重

「重要的權重」怎麼定義？
  → 看啟動值（Activation）。
  如果某個權重對應的啟動值經常很大，
  說明這個權重對輸出的影響很大 → 它是「重要的」。

AWQ 的做法：
  1. 用校準資料跑一遍 Forward，統計每個權重通道的啟動值大小
  2. 對重要通道的權重乘以一個 scale（放大），
     對不重要的權重除以 scale（縮小）
  3. 再做標準量化

  為什麼放大能保護？
    量化的精度 ∝ |值| / 2^bits
    值越大，量化的相對誤差越小。
    放大重要權重 → 它們的量化精度提高
    縮小不重要權重 → 它們的精度降低，但影響不大

  數學上：
    原始：y = W × x
    AWQ：y = (W × diag(s)) × (diag(1/s) × x)
    = W' × x'

    W' = W × diag(s)：權重被 scale 調整
    x' = diag(1/s) × x：啟動值被反向調整
    整體結果不變，但量化誤差的分佈變得更有利。
```

**AWQ vs GPTQ 比較**

```
┌──────────────┬──────────────┬──────────────┐
│              │ GPTQ         │ AWQ          │
├──────────────┼──────────────┼──────────────┤
│ 核心思路     │ 量化後補償    │ 量化前保護    │
│ 校準時間     │ 較慢（需逐列） │ 較快         │
│ 精度（INT4） │ 優秀          │ 優秀（略好） │
│ 推理速度     │ 快            │ 快           │
│ 記憶體       │ 相同          │ 相同         │
│ 主流地位     │ 2023-2024     │ 2025-2026    │
└──────────────┴──────────────┴──────────────┘
```

---

### 量化的實務 Trade-off

```
┌──────────┬──────────────┬──────────────┬──────────────┐
│ 量化方案 │ 記憶體節省    │ 速度提升      │ 品質損失      │
├──────────┼──────────────┼──────────────┼──────────────┤
│ FP16     │ 基準         │ 基準          │ 無            │
│ FP8/W8A8 │ 50%          │ Prefill 1.5-2×│ < 1%         │
│ INT8/W8  │ 50%          │ Decode 1.5-2× │ < 1%         │
│ INT4/W4  │ 75%          │ Decode 2-4×   │ 1-3%         │
│ INT3     │ 81%          │ Decode 2.5-5× │ 3-10%        │
│ INT2     │ 87%          │ Decode 3-6×   │ 10-30%       │
└──────────┴──────────────┴──────────────┴──────────────┘

INT4 是 2026 年的甜蜜點：
  75% 記憶體節省 + 可接受的精度損失
  GPTQ/AWQ 讓 INT4 的品質接近 FP16

INT8 是「幾乎無損」的安全選擇：
  如果品質至關重要（醫療、金融），INT8 比 INT4 更穩妥
```

> 量化是推理優化的第一道防線——在 Memory-Bound 的 Decode 階段，減少權重的讀取量就是最直接的加速。INT4（GPTQ/AWQ）在 75% 記憶體節省的同時把品質損失控制在 3% 以內，是 2026 年的主流選擇。

---

# 第 4 章：讓注意力算得更快——FlashAttention

## 標準注意力的記憶體問題

回憶 Doc2 的注意力計算：

```
Attention(Q, K, V) = Softmax(Q × Kᵀ / √d_k) × V

對於序列長度 N、頭維度 d_k：
  Q：[N × d_k]
  K：[N × d_k]
  V：[N × d_k]
  Q × Kᵀ：[N × N] ← 問題在這裡！
```

**N × N 的注意力矩陣是記憶體殺手**：

```
N = 4,096（常見上下文長度）：
  N² = 16,777,216 × 2 bytes (FP16) = 32 MB / per head
  32 個頭 → 32 × 32 MB = 1 GB
  可以接受。

N = 32,768：
  N² × 2 bytes × 32 heads = 64 GB
  一張 A100 80GB 幾乎全給注意力矩陣了。

N = 131,072（128K 上下文）：
  N² × 2 bytes × 32 heads = 1 TB
  完全不可能存進 GPU 記憶體。
```

標準做法是把 [N × N] 矩陣實體化（materialize）到 GPU 的 HBM（高頻寬記憶體）中，然後做 Softmax 和跟 V 的乘法。每一步都要在 HBM 和計算單元之間來回搬資料。

---

## GPU 記憶體層次

```
GPU 有多層記憶體，速度差異巨大：

┌──────────────┬──────────┬───────────────┐
│ 記憶體層級    │ 大小      │ 頻寬          │
├──────────────┼──────────┼───────────────┤
│ 暫存器       │ ~KB      │ ~TB/s         │
│ SRAM（L1/L2）│ ~20 MB   │ ~19 TB/s      │
│ HBM          │ 40-80 GB │ ~2 TB/s       │
│ CPU RAM      │ 幾百 GB  │ ~50 GB/s      │
└──────────────┴──────────┴───────────────┘

SRAM 比 HBM 快 ~10 倍，但小 ~1000 倍。
HBM 比 CPU RAM 快 ~40 倍。

標準注意力的問題：
  1. 把 Q、K 從 HBM 讀到計算單元
  2. 算 S = Q × Kᵀ，把 S 寫回 HBM（因為 S 太大，放不進 SRAM）
  3. 從 HBM 讀 S，做 Softmax，把 P 寫回 HBM
  4. 從 HBM 讀 P 和 V，算 O = P × V，寫回 HBM

  每個中間結果都要在 HBM 來回搬運——這就是瓶頸。
```

---

### FlashAttention 的核心思想

**關鍵洞察：永遠不需要把 N × N 矩陣存到 HBM。**

FlashAttention 用**分塊計算（Tiling）** + **在線 Softmax（Online Softmax）** 來解決這個問題。

**分塊計算**

```
不一次算整個 N × N 矩陣，而是分成小塊：

把 Q 分成 T_r 塊，每塊大小 B_r
把 K、V 分成 T_c 塊，每塊大小 B_c

選擇 B_r 和 B_c 使得每塊能放進 SRAM：
  一塊 Q：[B_r × d_k]
  一塊 K：[B_c × d_k]
  一塊 V：[B_c × d_k]
  中間結果 S_block：[B_r × B_c]

  只要 B_r × d_k + B_c × d_k + B_c × d_k + B_r × B_c ≤ SRAM 大小
  所有計算都在 SRAM 內完成，不需要把中間結果寫回 HBM。
```

**但有一個問題——Softmax 需要看完整的一行**

```
Softmax(S_i) = exp(S_ij) / Σ_j exp(S_ij)

分母的 Σ_j exp(S_ij) 需要遍歷第 i 行的所有元素。
如果只看了一塊，你不知道分母是多少。
```

**在線 Softmax（Online Softmax）**

```
FlashAttention 用遞增更新的方式解決這個問題：

初始化：
  O = [0]（輸出累加器）
  l = [0]（分母累加器）
  m = [-∞]（最大值追蹤器）

對每一塊 K_j、V_j：
  1. 算局部注意力分數：S_block = Q_block × K_jᵀ / √d_k
  2. 找局部最大值：m_new = max(m_old, max(S_block))
  3. 更新縮放因子：
     舊結果的修正 = exp(m_old - m_new)
     新塊的 exp = exp(S_block - m_new)
  4. 更新分母：l_new = l_old × exp(m_old - m_new) + Σ exp(S_block - m_new)
  5. 更新輸出：
     O_new = O_old × (l_old × exp(m_old - m_new) / l_new)
           + (exp(S_block - m_new) / l_new) × V_j

每處理一塊，就更新累加器。
最後一塊處理完，O 就是精確的注意力輸出——跟標準實現的結果完全一致。
```

**為什麼要追蹤最大值 m？**

```
數值穩定性。如果直接算 exp(S_ij)，S_ij 可能很大（比如 30），
exp(30) ≈ 10¹³ → FP16 溢位。

減去最大值 m 後，所有的 exp 參數 ≤ 0，exp 的結果在 (0, 1] 之間。
每處理一塊就更新 m，確保全程數值穩定。
```

---

### FlashAttention 的效果

```
記憶體節省：
  標準注意力：O(N²) — 需要存整個 N × N 矩陣
  FlashAttention：O(N) — 只存輸入和輸出，中間結果在 SRAM 中即時消費

  N = 128K 時：
    標準：1 TB 的注意力矩陣
    Flash：幾十 MB

速度提升：
  因為 HBM 讀寫次數大幅減少：
    標準：讀寫 O(N² × d) 的資料
    Flash：讀寫 O(N² × d² / SRAM_size) 的資料

  實測加速：1.5-3× 取決於序列長度
  序列越長，加速越明顯

計算結果：
  數學上完全精確——不是近似注意力！
  唯一的差異：浮點運算的順序不同，
  可能導致 FP16 層級的微小數值差異（最後幾位小數），
  但這不影響模型品質。
```

### FlashAttention 的演進

```
FlashAttention-1（2022）：
  核心思想，分塊 + 在線 Softmax
  速度 2-4×，記憶體 5-20×

FlashAttention-2（2023）：
  更好的 work partitioning（在 thread block 和 warp 之間更均衡）
  速度在 FA-1 基礎上再提升 ~2×

FlashAttention-3（2024）：
  利用 H100 的 FP8 Tensor Core 和非同步操作
  進一步壓榨硬體能力

2026 年，FlashAttention 已經內建在所有主流推理框架中
（PyTorch, vLLM, TensorRT-LLM），不需要手動啟用。
```

> FlashAttention 不是「近似注意力」——它算的是完全精確的注意力，只是用分塊計算和在線 Softmax 避免了把 N × N 矩陣存到慢速 HBM。這讓長序列注意力的記憶體從 O(N²) 降到 O(N)，速度提升 2-3 倍。

---

# 第 5 章：讓 KV Cache 不爆炸

Doc2 已經詳細介紹了 KV Cache 的原理和增長計算。這裡聚焦在**怎麼壓縮它**。

## KV Cache 問題的量級

```
回顧 Doc2 的計算（Llama 3.1 8B）：
  每生成 1 個 Token，KV Cache 增加 128 KB
  生成 128K Token → KV Cache ≈ 16 GB

  對於 70B 模型（Llama 3.1 70B，80 個 KV 頭，128 維）：
  每 Token：80 層 × 2(K+V) × 8 頭 × 128 維 × 2 bytes = 320 KB
  128K Token：320 KB × 128K = 40 GB

  一張 80 GB GPU：模型 35 GB（INT4）+ KV Cache 40 GB = 75 GB
  幾乎沒有餘量給第二個請求——吞吐量極低。
```

所以壓縮 KV Cache 不只是省記憶體——它直接決定了**一張 GPU 能同時服務多少個請求**。

---

### GQA（Grouped Query Attention）

Doc2 已經介紹過 GQA 的基本概念。這裡補充它對 KV Cache 的具體壓縮效果：

```
MHA（Multi-Head Attention）：32 個 Q 頭，32 個 KV 頭
  KV Cache / Token = 32 × 2 × 128 × 2 = 16,384 bytes = 16 KB / 層

GQA（Grouped Query Attention）：32 個 Q 頭，8 個 KV 頭
  KV Cache / Token = 8 × 2 × 128 × 2 = 4,096 bytes = 4 KB / 層
  壓縮比：4×

MQA（Multi-Query Attention）：32 個 Q 頭，1 個 KV 頭
  KV Cache / Token = 1 × 2 × 128 × 2 = 512 bytes = 0.5 KB / 層
  壓縮比：32×（但品質可能下降）

GQA 是目前主流的折衷方案——品質接近 MHA，KV Cache 只有 1/4。
Llama 3.1 的所有版本（8B/70B/405B）都用 GQA。
```

---

### MLA（Multi-head Latent Attention）

**為什麼需要它**

GQA 通過減少 KV 頭數壓縮快取，但頭數不能無限減——減到 MQA（1 個頭）時品質會下降。

MLA 用一個完全不同的思路：**不減頭數，減維度。**

**核心思想：低秩壓縮**

```
標準 MHA 的 KV Cache：
  每個 Token 存 K 和 V，各有 n_head × d_head 個值
  例如：32 頭 × 128 維 = 4,096 維的 K + 4,096 維的 V

MLA 的做法：
  不存完整的 K 和 V。
  而是存一個壓縮後的 latent vector c（維度遠小於 K+V）。

  Forward 時：
    1. 用投影矩陣 W_DKV 把隱藏向量壓縮到低維：
       c = W_DKV × h    [d_c] = [d_c × d] × [d]
       d_c << d（例如 d=4096, d_c=512）

    2. 存 c 到 KV Cache（而不是完整的 K 和 V）

    3. 需要計算注意力時，從 c 還原 K 和 V：
       K = W_UK × c    [n_head × d_head] = [n_head × d_head × d_c] × [d_c]
       V = W_UV × c    同上

  KV Cache 大小比較（每 Token 每層）：
    MHA：(32 × 128 + 32 × 128) × 2 = 16,384 bytes
    GQA（8 KV 頭）：(8 × 128 + 8 × 128) × 2 = 4,096 bytes
    MLA（d_c = 512）：512 × 2 = 1,024 bytes

  MLA 的 KV Cache 是 MHA 的 1/16，是 GQA 的 1/4。
```

**MLA 的代價**

```
推理時需要做反投影（W_UK × c 和 W_UV × c）來還原 K 和 V。
這增加了計算量——但在 Memory-Bound 的 Decode 階段，
減少記憶體讀取的收益 >> 多出的計算開銷。

MLA 被 DeepSeek-V2/V3 使用，KV Cache 壓縮了 93%，
吞吐量提升近 6 倍。
```

---

### Sliding Window Attention（滑動窗口注意力）

**核心思想：只注意最近的 Token**

```
標準 Causal Attention：每個 Token 注意它前面所有的 Token
  KV Cache 隨序列長度線性增長

Sliding Window Attention：每個 Token 只注意前面 W 個 Token
  W = 窗口大小（如 4,096）

  注意力範圍：
  標準：Token i 注意 Token 0, 1, 2, ..., i-1
  窗口：Token i 注意 Token max(0, i-W), ..., i-1

  KV Cache 大小固定為 W，不再隨序列增長：
    W = 4096 → KV Cache 永遠不超過 4096 Token 的量

  對比：
    128K 序列的 KV Cache：
    標準：128K Token 的 KV → 隨序列線性增長
    窗口（W=4096）：4096 Token 的 KV → 固定大小
    壓縮比：32×
```

**品質代價**

```
超出窗口的 Token 完全不被注意到。
如果模型需要引用文本開頭的資訊（如「第一段提到的數字」），
純滑動窗口可能做不到。

實務中的解法（Hybrid 架構，Doc2 的 Qwen 方案）：
  大部分層用滑動窗口或線性注意力（便宜，處理局部）
  少數層用完整注意力（貴，處理全局）
  這樣既控制了 KV Cache，又保留了長距離能力。
```

---

### KV Cache 壓縮技術比較

```
┌──────────────┬────────────┬────────────┬────────────────────┐
│ 技術          │ 壓縮比     │ 品質損失   │ 代表模型            │
├──────────────┼────────────┼────────────┼────────────────────┤
│ MHA          │ 1×（基準） │ 無         │ 原始 Transformer    │
│ GQA          │ 4-8×       │ < 1%       │ Llama 3, GPT-4     │
│ MQA          │ 32×        │ 1-3%       │ Falcon, PaLM        │
│ MLA          │ 16×        │ < 1%       │ DeepSeek-V2/V3     │
│ Sliding Window│ 取決於 W  │ 長距離退化  │ Mistral, Gemma     │
│ 線性注意力    │ 固定大小   │ 精度退化    │ Qwen3.5（Hybrid）  │
│ KV Cache 量化 │ 2-4×      │ 1-5%       │ 各框架支援          │
└──────────────┴────────────┴────────────┴────────────────────┘
```

> KV Cache 是長序列推理的最大記憶體瓶頸。GQA（減頭數）、MLA（低秩壓縮）、Sliding Window（限制範圍）從不同角度壓縮快取。2026 年的趨勢是混合方案——用 MLA 或線性注意力處理大部分層，用完整注意力處理少數關鍵層。

---

# 第 6 章：一次猜多個 Token——推測解碼

## Decode 的根本瓶頸

Doc2 介紹過，Decode 階段每次只生成 1 個 Token，但每次都要讀取整個模型的權重。這是 Memory-Bound 的根源——**每讀一次幾十 GB 的權重，只產出一個 Token。**

```
70B 模型（INT4，35 GB 權重）：
  生成 1 個 Token：讀 35 GB 權重 → 做 1 次矩陣乘法 → 產出 1 Token
  生成 100 個 Token：讀 35 GB × 100 = 3.5 TB 的資料

  A100 頻寬 2 TB/s → 100 Token 至少需要 1.75 秒
  → 每秒 ~57 Token（理論上限）

有沒有辦法「讀一次權重，產出多個 Token」？
```

---

### 推測解碼（Speculative Decoding）

**核心思想**

```
用一個小模型（Draft Model）快速猜 K 個 Token，
然後用大模型（Target Model）一次性驗證這 K 個 Token。

為什麼這有效？
  小模型（1B）比大模型（70B）快 10-50 倍。
  猜 K 個 Token 的成本很低。

  大模型一次驗證 K 個 Token：
  就是做一次 Prefill——輸入 K 個 Token，同時算所有位置的機率。
  Prefill 是 Compute-Bound 的，可以充分利用 GPU 算力。

  所以：讀一次 70B 權重 → 驗證 K 個 Token → 產出多個 Token。
```

**具體流程**

```
假設 Draft Model 猜了 K = 5 個 Token：
  d₁, d₂, d₃, d₄, d₅

Target Model 一次性驗證：
  把 [d₁, d₂, d₃, d₄, d₅] 當作輸入，做一次 Forward Pass
  得到每個位置的機率分佈：P₁, P₂, P₃, P₄, P₅

逐位檢查：
  位置 1：Target 在位置 0 的預測分佈 P₁
          Draft 猜的是 d₁
          如果 d₁ 符合 Target 的分佈 → 接受 ✓
  位置 2：Target 在位置 1 的預測分佈 P₂
          Draft 猜的是 d₂
          如果 d₂ 符合 → 接受 ✓
  位置 3：Target 在位置 2 的預測分佈 P₃
          Draft 猜的是 d₃
          如果 d₃ 不符合 → 拒絕 ✗
          從 P₃ 重新取樣一個 Token → 作為位置 3 的輸出

  結果：接受了 d₁, d₂，拒絕了 d₃
  本輪產出 3 個 Token（d₁, d₂, 和從 P₃ 取樣的新 Token）
  而不是只產出 1 個 Token。
```

**為什麼保證輸出完全一致？**

```
關鍵：驗證時用的是修正後的取樣（Modified Rejection Sampling）。

如果 Draft 猜的 Token d 的機率在 Target 分佈中為 p_target(d)，
在 Draft 分佈中為 p_draft(d)：

  接受機率 = min(1, p_target(d) / p_draft(d))

  直覺：
  - 如果 Target 比 Draft 更喜歡這個 Token → 100% 接受
  - 如果 Target 不太喜歡 → 按比例概率接受

  被拒絕時，從修正分佈中取樣：
  p_corrected(x) ∝ max(0, p_target(x) - p_draft(x))

  數學上可以證明：這個過程的最終輸出分佈
  === 完全由 Target Model 獨立生成的分佈。

  不是「近似相同」，是「數學上完全等價」。
```

**加速效果**

```
加速取決於「接受率」——Draft Model 猜對的比例。

接受率取決於：
  - Draft Model 跟 Target Model 的相似度
  - 生成內容的可預測性（常見句子接受率高，創意寫作接受率低）

典型加速：
  接受率 70%（K=5）：平均每輪產出 3.5 個 Token
  接受率 85%（K=5）：平均每輪產出 4.25 個 Token
  加速比：2-3×

Draft Model 的選擇：
  同家族的小模型（如 Llama 3.1 8B → Llama 3.1 1B）
  或者模型自身的淺層子網路（Self-Speculative Decoding）

2026 年實測：
  Llama 3.1 70B + 1B Draft → 2.5-3× 加速
  Apple Mirror Speculative Decoding → 2.8-5.8× 加速
```

> 推測解碼是 Decode 階段最優雅的加速方案——用小模型猜、大模型驗，把 Memory-Bound 的逐步生成轉換成 Compute-Bound 的批次驗證。最關鍵的是：輸出分佈在數學上與原始模型完全等價，沒有品質損失。

---

# 第 7 章：讓一張 GPU 服務更多人——Continuous Batching 與 PagedAttention

## 靜態 Batching 的浪費

```
最簡單的 Batching：把多個請求湊成一個 Batch 一起算。

但 LLM 的輸出長度是不確定的：

  請求 1：「翻譯 hello」 → 輸出 2 Token
  請求 2：「寫一篇文章」 → 輸出 500 Token
  請求 3：「1+1=?」 → 輸出 1 Token

靜態 Batching：等所有請求都完成才能處理新的。

  ┌────────────────────────────────────────────┐
  │ 請求 1 ██                                   │ ← 完成後在等
  │ 請求 2 █████████████████████████████████████ │
  │ 請求 3 █                                    │ ← 完成後在等
  └────────────────────────────────────────────┘
             ↑                                 ↑
           開始                         全部結束才能接新的

  請求 1 和 3 早就完成了，但 GPU 在等請求 2。
  大量的算力和記憶體被浪費。
```

---

### Continuous Batching（連續批次處理）

**核心思想**

```
不等所有請求完成——完成一個就立刻移出，空出的位置立刻插入新請求。

  ┌────────────────────────────────────────────┐
  │ 請求 1 ██                                   │
  │ 請求 2 █████████████████████████████████████ │
  │ 請求 3 █                                    │
  │ 請求 4    ███████████（請求 3 完成後插入）    │
  │ 請求 5      ████████████（請求 1 完成後插入） │
  │ 請求 6              ████████（4 完成後插入）  │
  └────────────────────────────────────────────┘

  GPU 永遠在滿負荷工作——沒有空閒的等待時間。
```

**Iteration-Level Scheduling**

```
在每一步 Decode 迭代中：
  1. 檢查有沒有已完成的請求（遇到 EOS 或 max_length）
  2. 移出已完成的請求，釋放它們的 KV Cache 記憶體
  3. 如果有等待中的新請求且記憶體足夠 → 插入新請求
  4. 對當前 Batch 中的所有請求做一步 Decode

  這讓 GPU 利用率從靜態 Batching 的 50-60% 提升到 90%+。
```

---

### PagedAttention——KV Cache 的虛擬記憶體

**為什麼需要它**

Continuous Batching 讓 GPU 更忙了，但**記憶體管理**成了新的瓶頸：

```
問題：KV Cache 的大小在請求開始時是未知的。

  請求開始時：不知道會生成多少 Token → 不知道 KV Cache 要多大

傳統做法：預分配最大長度
  每個請求預留 max_seq_len × KV_size_per_token 的記憶體
  比如：max_seq_len = 4096 → 預留 4096 × 128 KB = 512 MB

  但大多數請求遠用不到 4096 Token。
  如果平均生成 200 Token → 記憶體利用率 = 200/4096 = 4.9%
  95% 的預留記憶體被浪費了！
```

**PagedAttention 的靈感**

```
作業系統怎麼管理 CPU 記憶體？→ 虛擬記憶體 + 分頁。

  不預分配連續的大塊記憶體
  而是把記憶體分成固定大小的「頁」（Page）
  需要多少分配多少
  頁可以不連續——用頁表映射

PagedAttention 把同樣的概念用在 KV Cache 上。
```

**具體機制**

```
KV Cache 被分成固定大小的 Block：
  每個 Block 存 B 個 Token 的 KV 向量（B 通常 = 16 或 32）

  Block 大小 = B × 2(K+V) × n_kv_heads × d_head × 2 bytes

  Llama 3.1 8B（B=16）：
  每個 Block = 16 × 2 × 8 × 128 × 2 = 65,536 bytes = 64 KB

GPU 記憶體被預先劃分成一個 Block Pool：
  80 GB GPU，假設 40 GB 給 KV Cache
  Block 數量 = 40 GB / 64 KB = 655,360 個 Block

每個請求維護一個 Block Table（頁表）：
  ┌──────────┬────────────┐
  │ 邏輯位置  │ 物理 Block │
  ├──────────┼────────────┤
  │ Token 0-15  │ Block 42    │
  │ Token 16-31 │ Block 1,337 │
  │ Token 32-47 │ Block 98    │
  │ ...       │ ...        │
  └──────────┴────────────┘

  Block 在物理記憶體中不需要連續！
  需要時從 Pool 中分配新 Block。
  請求完成時歸還 Block。
```

**PagedAttention 的效果**

```
記憶體利用率：
  傳統預分配：~5%（大量浪費）
  PagedAttention：~95%+（只浪費最後一個 Block 的碎片）

  浪費量 = 每個請求最多浪費 (B-1) 個 Token 的空間
  B = 16 → 最多浪費 15 Token 的 KV Cache ≈ 可忽略

實際效果：
  同一張 GPU，能同時服務的請求數增加 2-4 倍
  吞吐量（Token/s）提升 2-4 倍

共享前綴（Prefix Sharing）：
  多個請求如果有相同的前綴（如同一個 System Prompt），
  它們可以共享對應的 KV Cache Block——不需要每個請求都存一份。
  這對 Chat 場景（所有請求共享相同 System Prompt）效果顯著。
```

---

### vLLM 的完整優化棧

```
vLLM 是目前最主流的 LLM 推理服務框架，集成了所有上述技術：

  PagedAttention     → 高效 KV Cache 記憶體管理
  Continuous Batching → 動態請求調度
  FlashAttention      → 高效注意力計算
  量化支援            → GPTQ / AWQ / FP8
  推測解碼            → Draft Model 加速
  Tensor Parallelism  → 多 GPU 並行

  效果：比 HuggingFace Transformers 的推理快 2-4 倍
       比靜態 Batching 的吞吐量高 2-4 倍
```

> Continuous Batching 讓 GPU 永遠忙碌，PagedAttention 讓記憶體不浪費。兩者結合把一張 GPU 的有效吞吐量提升了 2-4 倍。PagedAttention 的本質是把作業系統的虛擬記憶體概念搬到 KV Cache 上——按需分配、碎片歸零、可共享。

---

# 第 8 章：一張 GPU 不夠——多 GPU 並行

當模型太大一張 GPU 放不下（即使量化後），或者需要更高的吞吐量時，就需要**多 GPU 並行**。

---

## 三種基本並行策略

### Data Parallelism（資料並行, DP）

```
每張 GPU 都有一份完整的模型。
不同的請求分配到不同的 GPU。

  GPU 0：模型副本 A ← 處理請求 1, 4, 7, ...
  GPU 1：模型副本 B ← 處理請求 2, 5, 8, ...
  GPU 2：模型副本 C ← 處理請求 3, 6, 9, ...

好處：
  - 吞吐量線性增長（3 張 GPU → 3× 吞吐量）
  - 每個請求的延遲不變
  - 沒有 GPU 之間的通訊開銷

壞處：
  - 每張 GPU 都要裝得下完整模型
  - 70B INT4 = 35 GB → 一張 A100 80GB 裝得下
  - 70B FP16 = 140 GB → 一張 GPU 裝不下 → 不適用

適用場景：
  模型能放進單張 GPU，需要提升吞吐量
```

### Tensor Parallelism（張量並行, TP）

```
把每一層的矩陣切分到多張 GPU。

以 W_Q [4096 × 4096] 切到 4 張 GPU 為例：

  GPU 0：W_Q[:, 0:1024]     [4096 × 1024]
  GPU 1：W_Q[:, 1024:2048]  [4096 × 1024]
  GPU 2：W_Q[:, 2048:3072]  [4096 × 1024]
  GPU 3：W_Q[:, 3072:4096]  [4096 × 1024]

每張 GPU 各算一部分，然後通過 AllReduce 通訊合併結果。

計算過程：
  輸入 x [seq_len × 4096] → 廣播到所有 GPU

  GPU 0：y₀ = x × W_Q₀ → [seq_len × 1024]
  GPU 1：y₁ = x × W_Q₁ → [seq_len × 1024]
  GPU 2：y₂ = x × W_Q₂ → [seq_len × 1024]
  GPU 3：y₃ = x × W_Q₃ → [seq_len × 1024]

  AllReduce（或 AllGather）：合併 [y₀, y₁, y₂, y₃] → [seq_len × 4096]

好處：
  - 可以處理一張 GPU 放不下的模型
  - 每個請求的延遲降低（每張 GPU 算的量更少）

壞處：
  - 每一層都需要 AllReduce 通訊（延遲開銷）
  - 通訊量 ∝ hidden_size × batch_size
  - 需要高速互聯（NVLink 600 GB/s，跨節點 InfiniBand ~400 GB/s）

AllReduce 的成本：
  每個 Transformer Block 需要 2 次 AllReduce（Attention 後 + FFN 後）
  每次 AllReduce 的通訊量 ≈ 2 × hidden_size × batch_size × sizeof(dtype)

  70B 模型，TP=4：
  每層 2 × 2 × 8192 × 1 × 2 bytes ≈ 64 KB（單請求）
  80 層 × 64 KB = 5 MB / Token

  NVLink 600 GB/s → 延遲 ≈ 8 μs / 層 → 80 層 ≈ 640 μs
  這個通訊開銷在低 batch size 時佔比顯著。
```

### Pipeline Parallelism（管線並行, PP）

```
把不同的層分配到不同的 GPU。

  80 層模型，4 張 GPU：
  GPU 0：第 0-19 層
  GPU 1：第 20-39 層
  GPU 2：第 40-59 層
  GPU 3：第 60-79 層

  資料像流水線一樣流過：
  GPU 0 算完前 20 層 → 傳給 GPU 1 → GPU 1 算 20-39 層 → ...

好處：
  - 通訊量小（只在層的邊界傳一次）
  - 通訊量 = hidden_size × batch_size × sizeof(dtype)
  - 不需要超高速互聯——適合跨節點

壞處：
  - Pipeline Bubble（管線氣泡）
    GPU 0 算第 1 層時，GPU 1/2/3 在等 → 空閒
    GPU 3 算最後一層時，GPU 0/1/2 在等 → 空閒
    利用率理論上限 = (P-1)/P（P=4 → 75%）

  微批次（Micro-batching）可以緩解氣泡：
    把 Batch 切成多個 Micro-batch，交錯處理
    GPU 0 處理 micro-batch 2 時，GPU 1 在處理 micro-batch 1
    利用率提升到 ~90%+
```

---

## 實務配置原則

```
┌──────────────────────────────────────────────────────┐
│                 多 GPU 並行配置指南                    │
├──────────────────────────────────────────────────────┤
│ 節點內（同一台機器，有 NVLink）：                      │
│   優先 Tensor Parallelism                             │
│   因為 AllReduce 需要高頻寬，NVLink 提供 600 GB/s     │
│                                                      │
│ 節點間（不同機器，InfiniBand）：                       │
│   優先 Pipeline Parallelism                           │
│   因為只在層邊界通訊一次，對頻寬要求低                 │
│                                                      │
│ 常見配置：                                            │
│   8 GPU × 1 節點：TP = 8                              │
│   8 GPU × 2 節點：TP = 8, PP = 2                      │
│   8 GPU × 4 節點：TP = 8, PP = 4                      │
│                                                      │
│ MoE 模型額外選項：                                    │
│   Expert Parallelism（EP）：把不同的 Expert 放在       │
│   不同的 GPU 上。因為每個 Token 只啟動少數 Expert，     │
│   通訊量相對較小。                                    │
└──────────────────────────────────────────────────────┘
```

```
混合並行的例子——部署 Llama 3.1 405B：

  405B FP16 = 810 GB → 至少需要 11 張 A100 80GB
  405B INT4 = 203 GB → 至少需要 3 張 A100 80GB

  推薦配置（INT4，追求低延遲）：
    TP = 4（4 張 GPU 在一個節點內）
    每張 GPU：~51 GB 模型 + KV Cache 空間

  推薦配置（FP16，追求品質）：
    2 個節點 × 8 GPU：TP = 8, PP = 2
    每張 GPU：~51 GB 模型 + KV Cache 空間
```

> 多 GPU 並行是處理大模型的最後手段。節點內用 Tensor Parallelism（高頻寬 AllReduce），節點間用 Pipeline Parallelism（低通訊量）。實務中通常混合使用，配置取決於模型大小、GPU 數量和互聯頻寬。

---

# 第 9 章：案例——部署一個 70B 模型的完整方案

讓我們把前面所有技術組裝起來，部署 Llama 3.1 70B 作為一個線上服務。

---

## 方案 A：低成本方案（2 張 A100 80GB）

```
量化：AWQ INT4
  模型大小：70B × 0.5 = 35 GB
  每張 GPU：~17.5 GB 模型

並行策略：TP = 2
  每層矩陣切成兩半
  AllReduce 通過 NVLink（節點內）

KV Cache 預算：
  每張 GPU 剩餘：80 - 17.5 - 5(其他) = 57.5 GB
  每個 Token KV Cache（GQA，8 KV 頭，80 層，INT4 量化快取）：
    80 × 2 × 8 × 128 × 0.5 bytes ÷ 2 (TP) = 40 KB / Token / GPU
  最大容量：57.5 GB / 40 KB ≈ 1,500K Token

  如果每個請求平均 2K Token（輸入+輸出）：
  最大併發：1,500K / 2K ≈ 750 個請求

注意力：FlashAttention（自動啟用）

Batching：Continuous Batching（vLLM 自動處理）

KV Cache 管理：PagedAttention（Block Size = 16）

推測解碼：Llama 3.1 8B 作為 Draft Model
  額外記憶體：8B × 0.5 = 4 GB（INT4）
  預期加速：2-2.5×

總延遲（生成 200 Token）：
  不用推測解碼：
    Prefill（1K 輸入）：~50 ms
    Decode（200 Token）：~200 × 15 ms = 3 秒
    TTFT：~50 ms，總延遲：~3 秒

  用推測解碼：
    Prefill：~55 ms（Draft + Target）
    Decode（200 Token，接受率 70%）：~1.3 秒
    總延遲：~1.4 秒
```

---

## 方案 B：高品質方案（8 張 H100 80GB）

```
量化：FP8（W8A8）
  模型大小：70B × 1 = 70 GB
  每張 GPU：~8.75 GB 模型

並行策略：TP = 8
  模型分佈在 8 張 GPU 上

KV Cache 預算：
  每張 GPU 剩餘：80 - 8.75 - 5 = 66.25 GB
  每個 Token KV Cache（FP16）：
    80 × 2 × 8 × 128 × 2 bytes ÷ 8 (TP) = 40 KB / Token / GPU
  最大容量：66.25 GB / 40 KB ≈ 1,700K Token
  最大併發（2K Token / 請求）：~850 個請求

推測解碼：不需要（H100 算力足夠，Decode 本身已快）

總延遲（生成 200 Token）：
  Prefill（1K 輸入）：~10 ms（H100 算力 + TP=8）
  Decode（200 Token）：200 × 6 ms = 1.2 秒
  TTFT：~10 ms，總延遲：~1.2 秒

吞吐量：~2000-4000 Token/s（所有請求合計）
```

---

## 方案比較

```
┌──────────────┬────────────────────┬────────────────────┐
│              │ 方案 A（2×A100）   │ 方案 B（8×H100）   │
├──────────────┼────────────────────┼────────────────────┤
│ GPU 成本     │ ~$4/hr             │ ~$24/hr            │
│ 量化         │ AWQ INT4           │ FP8 W8A8           │
│ 品質         │ 接近 FP16（~97%）  │ 接近 FP16（~99%）  │
│ TTFT         │ ~50 ms             │ ~10 ms             │
│ Decode 速度  │ ~35 Token/s/req    │ ~165 Token/s/req   │
│ 最大併發     │ ~750 請求          │ ~850 請求          │
│ 推測解碼     │ 用（2.5× 加速）   │ 不需要             │
│ 適用場景     │ 成本敏感、離線處理 │ 低延遲、高品質要求 │
└──────────────┴────────────────────┴────────────────────┘

結論：
  方案 A 用 1/6 的成本達到了 ~80% 的性能
  方案 B 用 6 倍成本提供最低延遲和最高品質
  大部分應用場景選方案 A + 推測解碼就足夠
```

---

## 每種技術的貢獻

```
把所有優化技術疊加在一起，看看效果：

基線：70B FP16，單張 GPU（如果能放下），無優化
  → 大約 5 Token/s

+ 量化 INT4（4× 記憶體節省，~2× Decode 加速）
  → ~10 Token/s

+ FlashAttention（2× 注意力加速，支持長序列）
  → ~12 Token/s

+ Continuous Batching（2-4× 吞吐量）
  → 總吞吐量 ~30-50 Token/s（多個請求合計）

+ PagedAttention（2-4× 併發量）
  → 總吞吐量 ~80-120 Token/s

+ 推測解碼（2-3× 單請求加速）
  → 單請求 ~25 Token/s，總吞吐量 ~200+ Token/s

+ Tensor Parallelism TP=4（~3× 單請求加速）
  → 單請求 ~60 Token/s

全部疊加：
  從基線的 5 Token/s → 數百 Token/s 的吞吐量
  效率提升 ~50-100 倍
```

> 部署一個 70B 模型不是「選一個優化技術」——而是把量化、FlashAttention、KV Cache 管理、Batching、推測解碼、多 GPU 並行全部疊加起來。每個技術解決一個具體瓶頸，疊加後實現 50-100 倍的效率提升。

---

# 第 10 章：收尾

## 壓縮公式

完整版：LLM 推理優化是在六個維度上榨乾硬體的過程——量化壓縮模型（裝得下）、FlashAttention 優化注意力 IO（算得快）、GQA/MLA/Sliding Window 壓縮 KV Cache（存得下）、推測解碼並行驗證多個 Token（猜得準）、Continuous Batching + PagedAttention 最大化 GPU 利用率（服務得多）、Tensor/Pipeline Parallelism 堆多張 GPU（扛得住）。

口訣版：壓模型、快注意力、縮快取、猜多步、塞滿 GPU、多卡並行——六招疊加，效率百倍。

---

## 專有名詞速查表

### 算術強度（Arithmetic Intensity）
單位：FLOPs/Byte。每讀取一個 byte 的資料需要做多少次運算。決定瓶頸是算力還是頻寬。
- 故事中：Prefill（算術強度高，Compute-Bound）vs Decode（算術強度低，Memory-Bound）
- 案例中：理解為什麼量化對 Decode 加速最大

### Roofline 模型
分析 GPU 性能瓶頸的模型。拐點 = 算力/頻寬，低於拐點是 Memory-Bound，高於是 Compute-Bound。
- 故事中：所有推理優化的理論基礎

### 量化（Quantization）
用更少的 bit 表示權重或啟動值。從 FP16（16 bit）壓到 INT4（4 bit）→ 記憶體省 75%。
- 故事中：推理優化的第一道防線
- 案例中：70B INT4 只需 35 GB，一張 A100 放得下

### GPTQ（Gradient Post-Training Quantization）
逐列量化 + Hessian 誤差補償。INT4 品質接近 FP16。
- 故事中：比簡單量化精度損失少 10 倍以上

### AWQ（Activation-Aware Weight Quantization）
根據啟動值大小保護重要權重。比 GPTQ 校準更快、品質略好。
- 故事中：2025-2026 年的主流量化方法

### FlashAttention
用分塊計算 + 在線 Softmax 避免實體化 N × N 注意力矩陣。數學精確、記憶體 O(N)、速度 2-3×。
- 故事中：不是近似注意力，是精確注意力的 IO 優化
- 案例中：內建在所有主流框架中，自動啟用

### MLA（Multi-head Latent Attention）
用低秩投影壓縮 KV Cache 到 latent vector。壓縮比 16×，吞吐量提升 6×。
- 故事中：不減頭數，減維度
- 案例中：DeepSeek-V2/V3 的核心技術

### Sliding Window Attention（滑動窗口注意力）
每個 Token 只注意前面 W 個 Token。KV Cache 大小固定，不隨序列增長。
- 故事中：犧牲長距離能力換取固定 KV Cache 大小

### 推測解碼（Speculative Decoding）
小模型猜 K 個 Token、大模型一次驗證。輸出分佈在數學上與原始模型完全等價。
- 故事中：把 Memory-Bound 的逐步生成轉換成 Compute-Bound 的批次驗證
- 案例中：70B + 1B Draft → 2.5-3× 加速

### Continuous Batching（連續批次處理）
完成一個請求就立刻插入新請求，GPU 永遠滿負荷。
- 故事中：解決靜態 Batching 的空閒等待問題

### PagedAttention
把 KV Cache 分成固定大小的 Block，按需分配。靈感來自作業系統的虛擬記憶體分頁。
- 故事中：記憶體利用率從 ~5% 提升到 ~95%
- 案例中：vLLM 的核心技術

### Tensor Parallelism（張量並行, TP）
把每層的矩陣水平切分到多張 GPU，每張 GPU 算一部分，AllReduce 合併結果。
- 故事中：適合節點內（需要高頻寬 NVLink）
- 案例中：TP=8 讓 70B 模型分佈在 8 張 GPU 上

### Pipeline Parallelism（管線並行, PP）
把不同層分配到不同 GPU，資料像流水線流過。通訊量小，適合跨節點。
- 故事中：Pipeline Bubble 是主要問題，微批次可緩解

### Data Parallelism（資料並行, DP）
每張 GPU 都有完整模型副本，處理不同的請求。吞吐量線性增長但模型要能放下單張 GPU。
- 故事中：最簡單的並行策略，適合模型放得下的情況

### Expert Parallelism（EP）
MoE 模型專用：把不同的 Expert 放在不同的 GPU 上。
- 故事中：每個 Token 只啟動少數 Expert → 通訊量較小

---

## 文件資訊
- 最後更新日期：2026-03-23
- 延伸閱讀：
  - FlashAttention 論文：FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness (Dao et al., 2022)
  - PagedAttention 論文：Efficient Memory Management for Large Language Model Serving with PagedAttention (Kwon et al., 2023)
  - 推測解碼：Fast Inference from Transformers via Speculative Decoding (Leviathan et al., 2023)
  - GPTQ：GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers (Frantar et al., 2022)
  - AWQ：AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration (Lin et al., 2023)
  - MLA：DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model (DeepSeek-AI, 2024)
  - vLLM：vllm.ai
  - Roofline Model：S. Williams et al., "Roofline: An Insightful Visual Performance Model" (2009)
- 關鍵字：Quantization、INT4、INT8、GPTQ、AWQ、FlashAttention、Tiling、Online Softmax、KV Cache、GQA、MLA、Sliding Window、Speculative Decoding、Draft Model、Continuous Batching、PagedAttention、vLLM、Tensor Parallelism、Pipeline Parallelism、Data Parallelism、Expert Parallelism、Roofline、Arithmetic Intensity、Memory-Bound、Compute-Bound
- 關鍵知識：
  - Decode 是 Memory-Bound（算術強度 ≈ 1），Prefill 是 Compute-Bound（算術強度 ≈ seq_len）
  - INT4 量化是 2026 年的甜蜜點：75% 記憶體節省、品質損失 < 3%
  - FlashAttention 是精確計算的 IO 優化，不是近似
  - 推測解碼的輸出在數學上與原始模型完全等價
  - PagedAttention 把 KV Cache 記憶體利用率從 5% 提到 95%
  - 所有技術疊加可實現 50-100 倍效率提升
