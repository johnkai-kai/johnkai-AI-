> **johnkai AI 教學系列 — 第二篇**

# LLM Token 生成的完整旅程：從輸入文字到輸出回應的每一步

---

# 第 0 章：這份文件到底要幹嘛

你在 ChatGPT 的輸入框打了一句「什麼是機器學習？」，按下 Enter，幾秒後螢幕上開始一個字一個字地跳出回答。

這份文件追蹤的就是這段旅程——你的文字從進入模型到回答跳出來之間，到底經過了什麼。每一步都會拆到數學層級，讓你看到資料的形狀怎麼變、矩陣怎麼乘、機率怎麼算。

為了讓你同時看到「經典架構」和「前沿架構」的差異，全文使用兩個真實模型做平行追蹤：

| | CASE A：經典 Dense 架構 | CASE B：前沿 MoE + Hybrid 架構 |
|--|--|--|
| **模型** | Llama 3.1 8B（Meta, 2024） | Qwen3.5-35B-A3B（Alibaba, 2026） |
| **注意力** | 32 層全部用 Softmax Attention | 30 層線性注意力 + 10 層 Softmax |
| **FFN** | 每層一個完整 Dense FFN | 每層 256 個 Expert，每 Token 只啟動 9 個 |
| **總參數** | 8B（全部活躍） | 35B 總量，每 Token 只用 ~3B |

前置知識：讀完文件一《從傳統機器學習到 Transformer》，或者已經理解向量、矩陣乘法、注意力機制（Q/K/V）、Transformer Block 的基本組件。

---

# 第 1 章：你會學到什麼

這份文件會讓你真的看懂以下這些東西：

- 為什麼同一句中文在不同模型中會被切成不同數量的 Token
- 為什麼 Token ID 只是一個整數，模型卻能理解它的「意義」
- 為什麼 MoE（混合專家）架構能用更少的計算做到更好的效果
- 為什麼有些模型不用傳統的注意力機制，改用線性注意力
- 為什麼 LLM 的輸出本質上是一個機率分佈，而不是一個確定的答案
- 為什麼 Temperature、Top-p、Min-p 這些參數會影響生成品質
- 為什麼你按下 Enter 後要等一下才看到第一個字（TTFT）
- 為什麼 KV Cache 對生成速度至關重要
- 為什麼 Dense 模型和 MoE 模型在記憶體和速度上有截然不同的表現

這份文件會用這些故事來教你：

1. **一段文字穿越 LLM 的完整旅程**
   會帶你走過：
   - 文字被拆成碎片（分詞）→ 碎片變成向量（嵌入）→ 向量穿過幾十層 Transformer Block（注意力 + FFN）→ 最後一層的向量被投影成分數（Logits）→ 分數被處理成機率（Temperature / Top-p）→ 從機率中選出一個 Token → 拼回去重複整個過程

讀完故事後，你會看到這些真實案例：

1. **端到端追蹤「什麼是機器學習？」**
   你應該不靠額外說明就能看懂：
   - 這句話在 Llama 和 Qwen 中分別被切成幾個 Token
   - 每一步的資料形狀怎麼變化
   - Dense 架構和 MoE 架構在每一層的計算差異
   - 為什麼 Qwen 的 KV Cache 只有 Llama 的 1/5

---

# 第 2 章：文字怎麼變成碎片？

你輸入了「什麼是機器學習？」。但模型不懂文字——它只懂數字。所以第一步是把文字拆成模型能處理的最小單位，然後給每個單位一個編號。

這個過程叫**分詞**（**Tokenization**）。

---

### 分詞器（Tokenizer）

**為什麼需要它**

電腦只能處理數字。你需要一個工具把人類的文字轉換成一串整數，讓模型能處理。

**它是什麼**

分詞器做兩件事：

```
步驟 1：切割 — 把文字拆成一個個片段（Token）
步驟 2：查表 — 把每個片段對應到一個整數（Token ID）
```

Token 不是「字」也不是「詞」——它是分詞器根據統計規律決定的切割單位。同一段文字在不同模型的分詞器下可能被切成完全不同的 Token。

**【雙 CASE 對比】同一句話的分詞差異**：

```
原始文字：「什麼是機器學習？」

═══════════════════════════════════════════════════
CASE A — Llama 3.1 8B（詞彙表大小 = 128,256）
═══════════════════════════════════════════════════

  Llama 的分詞器以英文語料為主訓練，中文 Token 較少。
  中文常被逐字切割：

  "什" "麼" "是" "機" "器" "學" "習" "？"
  → 8 個 Token

═══════════════════════════════════════════════════
CASE B — Qwen3.5-35B-A3B（詞彙表大小 = 248,320）
═══════════════════════════════════════════════════

  Qwen 的分詞器包含大量中文詞彙，常見詞組合併為單一 Token：

  "什麼" "是" "機器學習" "？"
  → 4 個 Token（Llama 的一半！）

═══════════════════════════════════════════════════
影響
═══════════════════════════════════════════════════
  1. Token 數量直接決定計算量 — Token 越少，推理越快
  2. 注意力矩陣大小 = Token 數² → 8² = 64 vs 4² = 16（差 4 倍）
  3. 同樣的上下文長度（128K Token），Qwen 能裝入更多中文
```

---

### BPE 演算法（Byte Pair Encoding）

**為什麼需要它**

分詞器怎麼決定「把什麼切成一個 Token」？你不能手動定義——語言太多樣了。你需要一個自動建立詞彙表的演算法。

**它是什麼**

BPE 的核心思想極簡：**反覆合併最常出現的相鄰片段**。

```
訓練過程（在大量文本上執行）：

1. 起點：所有單一位元組（0-255），共 256 個基礎符號
2. 統計語料中所有相鄰符號對的出現頻率
3. 合併頻率最高的那對，產生一個新符號，加入詞彙表
4. 重複步驟 2-3，直到詞彙表大小達到目標

Llama 的目標：128,256 個 Token
Qwen 的目標：248,320 個 Token
```

**具體範例**：

```
語料：low low low low lowest newest

初始（逐字元）：l o w  l o w  l o w  l o w  l o w e s t  n e w e s t

第 1 輪：最常出現的相鄰對 = (l, o)，出現 5 次
  合併 → 新符號 "lo"
  詞彙表加入 "lo"

第 2 輪：最常出現的 = (lo, w)，出現 5 次
  合併 → 新符號 "low"

第 3 輪：最常出現的 = (e, s)，出現 2 次
  合併 → 新符號 "es"

...持續到詞彙表達到目標大小。
```

**結果**：高頻詞（"the", "is", "今天"）被合併成單一 Token，罕見詞被拆成多個 Token，極罕見的詞退化到逐位元組編碼。

---

### 詞彙表（Vocabulary）

**為什麼需要它**

BPE 訓練完成後，你得到的是一張固定的「Token↔ID」對照表——這就是詞彙表。

**它是什麼**

```
CASE A — Llama 3.1 8B 的詞彙表：128,256 個 Token
  - 以英文為主訓練，英文 Token 豐富
  - 中文 Token 少，常見中文詞需要拆成多個 Token

CASE B — Qwen3.5-35B-A3B 的詞彙表：248,320 個 Token
  - 詞彙表接近 Llama 的 2 倍
  - 大量中文、日文、韓文 CJK 詞彙
  - 還包含多模態 Token 和工具呼叫 Token

詞彙表越大的代價：
  嵌入矩陣更大（vocab_size × hidden_size）→ 佔更多記憶體
  LM Head 矩陣也更大 → 最後一步計算量增加
  但回報是：目標語言的分詞效率大幅提升
```

---

### 特殊 Token 與聊天模板（Chat Template）

**為什麼需要它**

你輸入的不只是「什麼是機器學習？」這句話。聊天系統會在你的文字外面包一層「模板」——加上角色標記（user / assistant）、對話邊界標記等，讓模型知道「這是使用者問的問題」和「你應該在這裡開始回答」。

**【雙 CASE】聊天模板**：

```
你的原始輸入：「什麼是機器學習？」

═══════════════════════════════════════════════════
CASE A — Llama 3（格式）
═══════════════════════════════════════════════════

  <|begin_of_text|><|start_header_id|>user<|end_header_id|>

  什麼是機器學習？<|eot_id|><|start_header_id|>assistant<|end_header_id|>

  模板化後：約 18 個 Token

═══════════════════════════════════════════════════
CASE B — ChatML（格式）
═══════════════════════════════════════════════════

  <|im_start|>user
  什麼是機器學習？<|im_end|>
  <|im_start|>assistant

  模板化後：約 12 個 Token（分詞效率更高 + 模板更簡潔）

═══════════════════════════════════════════════════
```

模板中的特殊 Token（如 `<|begin_of_text|>`、`<|im_start|>`）有自己的 Token ID，但它們不是普通的文字——它們是控制信號，告訴模型對話的結構。

**到這裡，你的文字已經變成了一串整數**：

```
CASE A：[128000, 128006, ..., 37507, 6390, ...]  共約 18 個 Token ID
CASE B：[248044, ..., 104642, 38960, ...]         共約 12 個 Token ID
```

> 分詞是 LLM 處理的第一步：文字被 BPE 分詞器切成 Token，每個 Token 對應一個整數 ID。不同模型的詞彙表不同，同一段文字可能被切成完全不同數量的 Token，直接影響計算效率。

---

# 第 3 章：碎片怎麼變成有意義的向量？

Token ID 只是一個整數——25580 就是 25580，它跟 25581 沒有數學上的「相似性」。但模型需要的是連續的、可以計算距離的向量。

---

### 嵌入層（Embedding Layer）

**為什麼需要它**

模型需要把離散的整數（Token ID）轉換成連續的高維向量。只有轉成向量，才能做矩陣乘法、算內積、衡量相似度。

**它是什麼**

嵌入層本質上就是一張巨大的查找表——一個 [vocab_size × hidden_size] 的矩陣。每一行對應一個 Token 的向量。

```
嵌入操作：x = EmbeddingMatrix[token_id]

就是拿 Token ID 當索引，去表中取出對應的那一行。不是計算，是查表。

【雙 CASE 維度】：

CASE A — Llama 3.1 8B：
  嵌入矩陣形狀：[128,256 × 4,096]
  每個 Token 被映射為 4,096 維的向量
  記憶體：128,256 × 4,096 × 2 bytes(FP16) ≈ 1.0 GB

CASE B — Qwen3.5-35B-A3B：
  嵌入矩陣形狀：[248,320 × 2,048]
  每個 Token 被映射為 2,048 維的向量
  記憶體：248,320 × 2,048 × 2 bytes ≈ 1.0 GB

CASE B 的 hidden_size（2,048）只有 CASE A（4,096）的一半。
但 CASE B 靠更多的 Expert 和更多的層來彌補維度的減少。
```

**經過嵌入層後，資料的形狀**：

```
CASE A：18 個 Token → 查表 → [18 × 4,096] 的矩陣
         每一行是一個 Token 的 4,096 維向量

CASE B：12 個 Token → 查表 → [12 × 2,048] 的矩陣
         每一行是一個 Token 的 2,048 維向量
```

---

### 位置編碼（RoPE — Rotary Position Embedding）

**為什麼需要它**

嵌入層只給了每個 Token 它自己的語義向量，但沒有告訴模型「這個 Token 在序列中的哪個位置」。同一個詞出現在句首和句尾應該有不同的表示。

**它是什麼**

文件一介紹了原始 Transformer 的正弦位置編碼（加法式）。現代 LLM 幾乎都改用 **RoPE**——旋轉位置編碼。

RoPE 不是把位置向量「加到」詞向量上，而是在注意力計算時用旋轉的方式注入位置。具體來說，它根據位置把 Q 和 K 向量旋轉一個角度——位置越遠的兩個 Token，旋轉角度差越大，內積值自然變小。

```
RoPE 的核心特性：
  1. 相對位置敏感：位置 5 和位置 8 的距離 = 位置 10 和位置 13 的距離
     → 同樣距離的詞對有相似的注意力模式
  2. 遠距離衰減：距離越遠的詞對，Q·K 的值自然越小
  3. 不需要額外參數：旋轉角度由公式計算，不用學習

【雙 CASE 差異】：

CASE A — Llama 3.1 8B：
  rope_theta = 500,000（控制旋轉頻率的基底）
  → 支援最長 128K tokens

CASE B — Qwen3.5-35B-A3B（Full Attention 層）：
  rope_theta = 10,000,000（基底大 20 倍）
  partial = 0.25（只對 25% 的維度做旋轉）
  → 支援最長 262K tokens，可擴展至 ~1M

CASE B 的 DeltaNet 層不使用 RoPE，改用 Causal Conv1D（因果卷積）注入位置。
```

**到這裡，你的資料已經從一串整數變成了一個矩陣**：

```
CASE A：[18 × 4,096]  — 18 個 Token，每個用 4,096 個數字描述
CASE B：[12 × 2,048]  — 12 個 Token，每個用 2,048 個數字描述
```

接下來，這個矩陣要穿過幾十層 Transformer Block，被反覆精煉。

> 嵌入層把離散的 Token ID 映射成連續的高維向量，RoPE 在注意力計算時注入位置資訊。經過這兩步，原始文字變成了一個可以被矩陣運算處理的數值矩陣。

---

# 第 4 章：向量怎麼被精煉？——Dense Transformer Block

現在你的資料是一個 [seq_len × hidden_size] 的矩陣。它要穿過 L 層 Transformer Block，每一層都會精煉這些向量——讓它們從「每個 Token 只知道自己」變成「每個 Token 融合了整個上下文的理解」。

我們先用 CASE A（Llama 3.1 8B）的 Dense 架構完整走一遍，然後在第 5 章介紹 CASE B 的 MoE + Hybrid 架構。

---

### 一個 Dense Transformer Block 的完整流程

```
CASE A — Llama 3.1 8B 的每一層（32 層全部結構相同）：

輸入 x（形狀：[18 × 4,096]）
│
├─────────────────────────────────────┐
│                                     │
▼                                     │
RMSNorm                               │
│                                     │
▼                                     │
GQA Self-Attention                    │（殘差連接）
│  32 個 Q 頭，8 個 KV 頭             │
│  head_dim = 128                     │
│  + RoPE 位置編碼                    │
│  + Causal Mask                      │
▼                                     │
+ ◄───────────────────────────────────┘  x₁ = Attn(RMSNorm(x)) + x
│
├─────────────────────────────────────┐
│                                     │
▼                                     │
RMSNorm                               │
│                                     │
▼                                     │
Dense SwiGLU FFN                      │（殘差連接）
│  4,096 → 14,336 → 4,096            │
▼                                     │
+ ◄───────────────────────────────────┘  x₂ = FFN(RMSNorm(x₁)) + x₁
│
▼
輸出（形狀：[18 × 4,096]）→ 傳給下一層
```

**輸入和輸出的形狀完全一樣**——每一層都是 [seq_len × hidden_size] 進去、[seq_len × hidden_size] 出來。改變的不是形狀，而是每個向量的「內容」——經過注意力和 FFN 處理後，向量攜帶了更豐富的語義資訊。

---

### RMSNorm（Root Mean Square Normalization）

**為什麼需要它**

向量在經過多次矩陣乘法後，數值可能爆大或趨近零。正規化把向量的數值範圍拉回到穩定的區間，讓後續的計算更穩定。

**它是什麼**

RMSNorm 只做一件事：把向量除以它的均方根。

```
對一個 d 維向量 x = [x₁, x₂, ..., x_d]：

  RMS = √( (x₁² + x₂² + ... + x_d²) / d )

  x̂ᵢ = xᵢ / (RMS + ε)    其中 ε ≈ 1e-5，防止除以零

  outputᵢ = γᵢ × x̂ᵢ      其中 γ 是可學習的縮放向量

CASE A 的 ε = 1e-5
CASE B 的 ε = 1e-6（更精確）
```

比標準 LayerNorm 更簡單——省略了「減均值」的步驟，計算更快，效果相當。

---

### GQA（Grouped Query Attention）

**為什麼需要它**

文件一介紹了多頭注意力：每個頭有獨立的 W_Q、W_K、W_V。但 K 和 V 會被存進 KV Cache（後面會解釋），KV 頭數越多，快取越大。GQA 讓多個 Q 頭共享一組 K/V 頭，在品質和記憶體之間取得平衡。

**它是什麼**

```
三種注意力的 KV 分配方式：

MHA（Multi-Head Attention）：
  每個 Q 頭有獨立的 KV 頭
  32 個 Q 頭 → 32 組 KV → KV Cache 最大

GQA（Grouped Query Attention）：   ← CASE A 使用
  多個 Q 頭共享一組 KV 頭
  32 個 Q 頭，8 組 KV → 每 4 個 Q 共享 1 組 KV → KV Cache 是 MHA 的 1/4

MQA（Multi-Query Attention）：
  所有 Q 頭共享同一組 KV
  32 個 Q 頭，1 組 KV → KV Cache 最小，但品質可能下降
```

**CASE A 的注意力完整維度追蹤**：

```
輸入 x：[18 × 4,096]

Q = x · W_Q：[18 × 4,096] × [4,096 × 4,096] = [18 × 4,096]
K = x · W_K：[18 × 4,096] × [4,096 × 1,024] = [18 × 1,024]
V = x · W_V：[18 × 4,096] × [4,096 × 1,024] = [18 × 1,024]

Q 的維度 = 32 頭 × 128 維/頭 = 4,096
K/V 的維度 = 8 頭 × 128 維/頭 = 1,024

拆成多頭：
  Q：[18 × 32 × 128]    （32 個 Q 頭）
  K：[18 ×  8 × 128]    （8 個 KV 頭）
  V：[18 ×  8 × 128]

每 4 個 Q 頭共享 1 組 KV 頭：
  Q 頭 0~3  ↔ KV 頭 0
  Q 頭 4~7  ↔ KV 頭 1
  ...
  Q 頭 28~31 ↔ KV 頭 7

對每個 Q 頭，計算注意力（以 Q 頭 0 為例）：
  scores = Q₀ · K₀ᵀ / √128
         = [18 × 128] × [128 × 18] / 11.3
         = [18 × 18] / 11.3

  + Causal Mask（下三角遮罩，未來位置 → -∞）

  weights = softmax(scores)    → [18 × 18]
  output₀ = weights · V₀      → [18 × 128]

32 個頭各自計算後，拼接：
  Concat(output₀, ..., output₃₁) → [18 × 4,096]

最後投影：
  output · W_O = [18 × 4,096] × [4,096 × 4,096] = [18 × 4,096]
```

---

### Dense SwiGLU FFN

**為什麼需要它**

注意力讓 Token 之間交換了資訊。FFN 對每個 Token 獨立地做非線性處理——「開完會後各自消化」。

**它是什麼**

SwiGLU 用三個權重矩陣加上門控機制：

```
FFN(x) = W_down · (SiLU(W_gate · x) ⊙ (W_up · x))

CASE A 的維度：
  W_gate：[4,096 × 14,336]    — 門控投影
  W_up：  [4,096 × 14,336]    — 上投影
  W_down：[14,336 × 4,096]    — 下投影

計算過程：
  gate = SiLU(x · W_gate)     [18 × 4,096] → [18 × 14,336]
  up   = x · W_up             [18 × 4,096] → [18 × 14,336]
  gated = gate ⊙ up           [18 × 14,336]（逐元素相乘）
  output = gated · W_down     [18 × 14,336] → [18 × 4,096]

門控的作用：SiLU(W_gate · x) 產生 0~x 之間的門控信號，
選擇性地放行或抑制 W_up · x 中的各個特徵。

單層 FFN 參數量 = 3 × 4,096 × 14,336 ≈ 176M
佔每層總參數的 ~81%
```

**殘差連接**把 FFN 的輸出加回輸入：x₂ = FFN(RMSNorm(x₁)) + x₁

> Dense Transformer Block 的完整流程：RMSNorm → GQA Attention → 殘差 → RMSNorm → SwiGLU FFN → 殘差。輸入和輸出形狀相同，改變的是向量攜帶的語義資訊。

---

# 第 5 章：為什麼一種架構不夠？——MoE 與 Hybrid Attention

CASE A 的 Llama 3.1 8B 是「純粹」的 Dense Transformer——每一層結構相同，每個 Token 使用全部 8B 參數。這種架構簡單、穩定，但有一個根本限制：

**要提升模型能力，你只能增加參數量。但增加參數量 = 增加每個 Token 的計算量。**

8B 參數 → 每個 Token 用 8B 參數計算。80B 參數 → 每個 Token 用 80B 參數計算。計算量隨參數量線性增長。

有沒有辦法讓模型擁有 35B 參數的知識，但每個 Token 只用 3B 參數的計算量？

這就是 **MoE（Mixture of Experts）** 的核心想法。

---

### MoE（混合專家機制）

**為什麼需要它**

Dense FFN 的所有參數對每個 Token 都啟動。但直覺上，處理「今天天氣很好」不需要動用「微積分公式」相關的參數。如果能讓模型「按需啟動」，就能用更少的計算獲得更大模型的知識。

**它是什麼**

MoE 把一個大型 FFN 拆成很多個小型的 **Expert**（專家），每個 Expert 就是一個獨立的小型 SwiGLU FFN。每個 Token 由一個 **Router**（路由器）決定只送去其中幾個 Expert 處理。

```
CASE B — Qwen3.5-35B-A3B 的 MoE 配置：

  Expert 總數：256 個
  每 Token 啟動數：8 個 routed + 1 個 shared = 9 個
  每個 Expert 結構：2,048 → 512 → 2,048（SwiGLU，3 個矩陣）

  單個 Expert 參數量：3 × 2,048 × 512 ≈ 3.1M
  單層 MoE 總參數量：257 × 3.1M ≈ 809M
  單層活躍參數量：9 × 3.1M ≈ 28M

  稀疏率：28M / 809M ≈ 3.5%
  → 每個 Token 只用了單層 FFN 3.5% 的參數！
```

**Router 的工作原理**：

```
Router 是一個簡單的線性層：

  scores = x · W_router    [2,048] × [2,048 × 256] → [256]

  256 個分數，每個代表「這個 Token 應該被這個 Expert 處理的程度」。

  選出分數最高的 8 個 Expert（Top-8 Routing）：
    top_8_indices = TopK(scores, k=8)
    top_8_weights = Softmax(scores[top_8_indices])

  每個選中的 Expert 獨立處理 Token：
    output_i = Expert_i(x)     （一個小型 SwiGLU FFN）

  加權聚合 + Shared Expert：
    final = Σ(weight_i × output_i) + SharedExpert(x)
```

**為什麼 Expert 的 intermediate_size 只有 512（比 hidden_size 2,048 還小）？**

在 Dense FFN 中，intermediate_size 通常是 hidden_size 的 3~4 倍（擴展）。但在 MoE 中，9 個 Expert 的輸出會加權求和，等效中間維度 ≈ 512 × 9 = 4,608，大約是 hidden_size 的 2.25 倍。而且不同 Expert 可能「專精」不同的特徵子空間，聚合後的表達能力反而更強。

**MoE 的記憶體悖論**：

```
計算速度由「活躍參數量」決定  → 3B，快
記憶體需求由「總參數量」決定  → 35B，大

為什麼？因為 Router 可能把任何 Token 路由到任何 Expert。
所有 256 個 Expert 的權重都必須載入 VRAM，隨時待命。

CASE A — Llama 3.1 8B（Dense）：
  模型權重（FP16）：~16 GB
  每 Token 計算量：~16 GFLOPs（用全部 8B 參數）

CASE B — Qwen3.5-35B-A3B（MoE）：
  模型權重（FP16）：~70 GB
  每 Token 計算量：~6 GFLOPs（只用 3B 活躍參數）

  → 記憶體大 4.4 倍，但計算量只有 38%
```

---

### 線性注意力（Gated DeltaNet）

**為什麼需要它**

標準 Softmax Attention 的計算量隨序列長度平方增長（O(n²)）。1000 個 Token 的注意力矩陣是 [1000 × 1000] = 一百萬個元素。10,000 個 Token → 一億個元素。長序列下，注意力計算成為瓶頸。

而且，Softmax Attention 需要 **KV Cache**（後面會詳細解釋）——快取所有歷史 Token 的 K 和 V 向量，記憶體隨序列長度線性增長。

有沒有辦法繞過 O(n²) 的計算和不斷增長的快取？

**它是什麼**

Gated DeltaNet 是一種**線性注意力**替代方案。它不計算完整的 [seq_len × seq_len] 注意力矩陣，而是維護一個固定大小的「狀態矩陣」，每讀一個新 Token 就更新這個狀態。

```
Softmax Attention（CASE A 的所有層 + CASE B 的 10 層）：
  每個新 Token → 跟所有歷史 Token 做內積 → O(n) per token
  需要 KV Cache，大小隨序列長度增長

DeltaNet（CASE B 的 30 層）：
  每個新 Token → 更新固定大小的狀態矩陣 → O(1) per token
  狀態矩陣大小固定，不隨序列長度增長

  狀態矩陣形狀：[head_dim × head_dim] = [128 × 128]
  每層 16 個 QK 頭 → 16 個狀態矩陣
  每層固定記憶體 = 16 × 128 × 128 × 2 bytes ≈ 0.5 MB（不變！）
```

**但 DeltaNet 有代價**：它把歷史資訊壓縮進固定大小的狀態中，必然會丟失一些細節。對於需要精確回溯很遠之前的某個 Token 的任務，它不如 Softmax Attention。

---

### Hybrid Attention——兩者混合

**為什麼需要它**

純 Softmax Attention：精確但記憶體貴、計算 O(n²)。
純線性注意力：高效但會丟失長距離細節。

解法：混合使用。大部分層用線性注意力（快、省記憶體），少部分層用 Softmax Attention（精確的「全域刷新」）。

**CASE B 的混合策略**：

```
40 層，3:1 交替佈局：

  Layer 0:  DeltaNet（線性）+ MoE
  Layer 1:  DeltaNet（線性）+ MoE
  Layer 2:  DeltaNet（線性）+ MoE
  Layer 3:  Full Attention（Softmax）+ MoE   ★ 全域刷新
  Layer 4:  DeltaNet（線性）+ MoE
  Layer 5:  DeltaNet（線性）+ MoE
  Layer 6:  DeltaNet（線性）+ MoE
  Layer 7:  Full Attention（Softmax）+ MoE   ★ 全域刷新
  ...（重複 10 個循環）

  總計：30 層 DeltaNet + 10 層 Full Attention = 40 層

直覺理解：
  DeltaNet 層像是「快速掃描」——高效地處理序列，但可能積累誤差
  Full Attention 層像是「定期校正」——精確地重新計算所有位置間的關係
  每 3 層快速掃描後做 1 次精確校正，平衡效率和精度
```

---

### Dense vs MoE + Hybrid：完整對照

```
┌──────────────────┬─────────────────────┬──────────────────────────┐
│                  │ CASE A — Dense      │ CASE B — MoE + Hybrid    │
│                  │ (Llama 3.1 8B)      │ (Qwen3.5-35B-A3B)       │
├──────────────────┼─────────────────────┼──────────────────────────┤
│ 層數             │ 32（全部相同）       │ 40（30 DeltaNet + 10 FA）│
│ hidden_size      │ 4,096               │ 2,048                    │
│ 總參數           │ 8B（全部活躍）       │ 35B（每 Token 活躍 3B）  │
│ 注意力類型       │ 全部 Softmax         │ 75% 線性 + 25% Softmax  │
│ FFN 類型         │ Dense SwiGLU         │ MoE（256 Expert, 9 活躍）│
│ KV Cache 層數    │ 32 層（全部需要）    │ 10 層（僅 Full Attn）    │
│ 固定狀態         │ 無                   │ 30 層 DeltaNet 狀態      │
│ 每 Token FLOPs   │ ~16 GFLOPs           │ ~6 GFLOPs                │
│ 模型權重(FP16)   │ ~16 GB               │ ~70 GB                   │
│ 最大上下文       │ 128K tokens          │ 262K tokens              │
└──────────────────┴─────────────────────┴──────────────────────────┘
```

> MoE 用稀疏計算實現「大模型知識、小模型計算量」的平衡；Hybrid Attention 用線性注意力壓縮大部分層的記憶體和計算、用少量 Softmax 層做精確校正。兩者結合，讓 35B 參數的模型以接近 3B 的速度運行。

---

# 第 6 章：穿過所有層——從淺層特徵到深層理解

你的資料矩陣現在要穿過所有 Transformer Block——CASE A 是 32 層，CASE B 是 40 層。

**不同深度的層學到不同層次的特徵**：

```
淺層（前幾層）：
  學到詞法、簡單語法
  「什麼」是疑問詞，「機器」是名詞

中間層：
  學到語義關係、複合概念
  「機器學習」是一個專有名詞，不是「機器」+「學習」

深層（後幾層）：
  學到任務理解、推理模式
  這是一個問答任務，需要給出定義性的回答
```

**CASE A 的完整流程**：

```
嵌入後：[18 × 4,096]

Layer 0:  RMSNorm → GQA Attention → 殘差 → RMSNorm → Dense FFN → 殘差
Layer 1:  （同上，參數不同）
...
Layer 31: （同上，參數不同）

最終 RMSNorm → 輸出：[18 × 4,096]

32 層 × (42M 注意力 + 176M FFN) ≈ 7.0B 參數
```

**CASE B 的完整流程**：

```
嵌入後：[12 × 2,048]

Layer 0 (DeltaNet):
  RMSNorm → Gated DeltaNet → 殘差 → RMSNorm → MoE(9/256) → 殘差
Layer 1 (DeltaNet):  同上
Layer 2 (DeltaNet):  同上
Layer 3 (Full Attn): ★
  RMSNorm → Gated Attention(Softmax, 16Q/2KV) → 殘差 → RMSNorm → MoE(9/256) → 殘差
Layer 4 (DeltaNet):  同 Layer 0
...（3:1 交替直到 Layer 39）

最終 RMSNorm → 輸出：[12 × 2,048]

40 層 × (~25M 注意力 + ~809M MoE) ≈ 33.4B 參數（活躍 ~2.1B）
```

穿過所有層後，每個位置的向量已經從「只知道自己是什麼詞」變成了「理解了整個句子的語境、知道該怎麼回應」。

> 資訊在逐層傳遞中被不斷精煉：淺層學詞法和語法，中間層學語義關係，深層學任務理解。最終輸出的向量攜帶了對整個輸入序列的完整理解。

---

# 第 7 章：從向量到文字——預測下一個 Token

經過 L 層 Transformer Block，你得到了一個 [seq_len × hidden_size] 的矩陣。但你要的不是一組向量——你要的是**下一個詞**。

---

### LM Head（語言模型頭）

**為什麼需要它**

你需要把 hidden_size 維的向量「翻譯」成「對詞彙表中每個 Token 的評分」。

**它是什麼**

LM Head 就是一次矩陣乘法，把 hidden_size 維的向量投影到 vocab_size 維。

```
步驟 1：最終 RMSNorm
  對最後一層的輸出做正規化

步驟 2：線性投影
  logits = h_final × W_LM_head

  自迴歸生成時，只需要最後一個位置的向量：

  CASE A：
    h_final：[1 × 4,096]（最後一個 Token 位置的向量）
    W_LM_head：[4,096 × 128,256]
    logits = [1 × 4,096] × [4,096 × 128,256] = [1 × 128,256]

    → 128,256 個分數，每個對應詞彙表中的一個 Token

  CASE B：
    h_final：[1 × 2,048]
    W_LM_head：[2,048 × 248,320]
    logits = [1 × 2,048] × [2,048 × 248,320] = [1 × 248,320]

    → 248,320 個分數
```

**Logits（原始分數）** 是任意的實數——可以是正的、負的、很大、很小。分數高的 Token 表示模型認為它更可能是下一個詞。

```
logits 範例（CASE A，只列幾個）：

  logits[25580] = 8.2     → "我" 的分數
  logits[12345] = 15.7    → 某個 Token 的分數（最高分）
  logits[38960] = -1.3    → "今天" 的分數
  logits[6390]  = 0.4     → "了" 的分數
  ...
  共 128,256 個分數
```

但這些原始分數不能直接用——它們沒有上下界，也不是機率。你需要一系列處理步驟把它們變成「可以取樣的機率分佈」。

> LM Head 是一次矩陣乘法，把 hidden_size 維的語義向量投影成 vocab_size 維的原始分數（logits）。每個分數對應詞彙表中的一個 Token，分數越高表示模型越認為這個 Token 應該是下一個。

---

# 第 8 章：把原始分數變成可用的機率——Logits 處理管線

LM Head 吐出了 128,256 個（或 248,320 個）原始分數。在選出下一個 Token 之前，這些分數要經過一條處理管線：

```
Raw Logits
    ↓
懲罰機制（Repetition / Frequency / Presence Penalty）
    ↓
Temperature Scaling（溫度縮放）
    ↓
Top-k Filtering
    ↓
Top-p Filtering（Nucleus Sampling）
    ↓
Min-p Filtering
    ↓
Softmax → 機率分佈
    ↓
Sampling（取樣）→ 輸出一個 Token
```

每個步驟都是可選的。如果全部關閉，就是純粹的 Softmax 後隨機取樣。

---

### 懲罰機制——抑制重複

LLM 有時會陷入重複迴圈（「我是我是我是...」）。懲罰機制在取樣前修改已出現過的 Token 的分數，降低它們被再次選中的機率。

**三種懲罰**：

```
1. Repetition Penalty（重複懲罰）— 乘法式，不計次數
   正 logit → 除以 rep_penalty（變小）
   負 logit → 乘以 rep_penalty（更負）
   出現 1 次和 10 次的懲罰相同
   常用值：1.05~1.15
   框架：llama.cpp、Ollama

2. Frequency Penalty（頻率懲罰）— 加法式，計次數
   logit = logit − freq_penalty × count(token)
   出現越多次扣越多
   常用值：0.3~0.8
   框架：OpenAI API

3. Presence Penalty（存在懲罰）— 加法式，不計次數
   logit = logit − pres_penalty × 𝟙[已出現過]
   出現過就扣固定分，不論幾次
   鼓勵引入新詞彙
   常用值：0.3~0.8
   框架：OpenAI API

一般只用其中一種，不需要同時使用三種。
```

---

### Temperature（溫度）——控制隨機程度

**為什麼需要它**

有時你想要精確、可預測的輸出（寫程式碼）。有時你想要有創意、有驚喜的輸出（寫小說）。Temperature 是控制這個旋鈕的。

**它是什麼**

```
logits_scaled = logits / T

T < 1.0 → 差距放大 → 分佈更尖銳 → 模型更確定
T = 1.0 → 原始分佈
T > 1.0 → 差距縮小 → 分佈更平坦 → 模型更隨機
T → 0   → 永遠選最高分的 Token（Greedy Decoding）
```

**數值範例**：

```
原始 logits = [5.0, 3.0, 1.0, -1.0, -3.0]

                Token A   Token B   Token C   Token D   Token E
T = 0.3         99.87%    0.13%     ~0%       ~0%       ~0%
T = 0.7         94.22%    5.46%     0.31%     ~0%       ~0%
T = 1.0         86.50%    11.70%    1.58%     0.21%     0.03%
T = 1.5         73.67%    19.48%    5.15%     1.35%     0.36%

低溫 → 贏家通吃
高溫 → 雨露均霑
```

---

### Top-k 過濾——硬性人數限制

```
只保留 logits 最高的 k 個 Token，其餘設為 -∞。
經過 Softmax 後被排除的 Token 機率歸零。

缺點：k 是固定的，不會根據分佈形狀調整。
  模型很確定時 k=50 太多（引入噪音）
  模型不確定時 k=50 太少（排除了合理選項）
```

---

### Top-p 過濾（Nucleus Sampling）——自適應

```
從機率最高的 Token 開始累加，直到累積機率 ≥ p，保留這個最小集合。

自適應優勢：
  模型很確定（第 1 名佔 95%）→ 只保留 1 個 Token
  模型不確定（前 15 名才湊到 90%）→ 保留 15 個 Token
  候選數量隨模型信心自動調整
```

---

### Min-p 過濾——相對門檻

```
只保留機率 ≥ min_p × P(最高機率 Token) 的 Token。

門檻隨最高機率動態浮動：
  最高機率 95% → 門檻 = 0.1 × 0.95 = 9.5%（嚴格）
  最高機率 10% → 門檻 = 0.1 × 0.10 = 1%（寬鬆）
```

---

### 推薦組合

```
精確任務（程式碼/翻譯）：Temperature 0.2 + Top-p 0.9
日常對話：Temperature 0.7 + Top-p 0.9
創意寫作：Temperature 1.0 + Top-p 0.95 + Min-p 0.05
確定性輸出：Temperature 0（Greedy）

少即是多：通常 Temperature + 一個過濾器就夠了。
Temperature 是最重要的旋鈕——先調它，再決定是否需要過濾。
```

> Logits 處理管線把原始分數逐步塑形成可取樣的機率分佈：懲罰抑制重複 → Temperature 控制隨機度 → Top-k/Top-p/Min-p 過濾掉不合理的候選 → Softmax 轉成機率。每個步驟都可獨立開關。

---

# 第 9 章：選出下一個 Token——取樣策略

經過整條處理管線，你手上有一個合法的機率分佈——所有值在 0~1 之間、加總為 1。現在要從中選出一個 Token。

---

### 貪婪解碼（Greedy Decoding）

最簡單：直接選機率最高的。

```
next_token = argmax(probabilities)

  優點：確定性，可重現
  缺點：單調、容易重複、缺乏多樣性

  適用：程式碼生成、事實問答、數學計算
```

---

### 隨機取樣（Stochastic Sampling）

根據機率分佈擲一顆「加權骰子」：

```
每個 Token 被選中的機率 = 它在分佈中的機率

機率分佈 [0.45, 0.25, 0.15, 0.10, 0.05]
→ 想像一把 1 公尺的尺：

  |---------- 飯 ---------|--- 麵 ---|-- 菜 -|粥-|湯|
  0                      0.45     0.70   0.85 0.95 1.0

生成一個 0~1 的隨機數，看落在哪個區間。
機率高的 Token 佔的區間長，更容易被選中。

  優點：多樣性、創造力、自然感
  缺點：偶爾選到低品質的 Token
  適用：對話、寫作、大多數 LLM 聊天場景
```

Temperature、Top-k、Top-p 不是取樣方法本身——它們是在「擲骰子之前」修改骰子的形狀。修改完後，最終都是同一個動作：擲骰子。

---

### Beam Search（束搜尋）

同時維護 B 條候選路徑，每步擴展所有路徑，只保留總分最高的 B 條。

```
每步計算量 ≈ Greedy 的 B 倍。
  輸出更「正確」但更「無聊」。

  現代 LLM 聊天幾乎不使用——計算太貴，輸出太保守。
  仍然適用於：翻譯、摘要、結構化輸出（JSON/SQL）。
```

> 取樣的本質是從機率分佈中選一個 Token。Greedy 永遠選最大值（確定性），隨機取樣按機率擲骰子（多樣性）。現代 LLM 聊天以隨機取樣為主。

---

# 第 10 章：自迴歸循環——從第一個字到完整回答

到目前為止，你已經走完了「輸入→分詞→嵌入→L 層 Transformer→LM Head→Logits 處理→取樣→一個 Token」的完整路徑。

但 LLM 一次只生成一個 Token。要生成完整的回答，需要把生成的 Token 拼回輸入，重新走一遍整個流程。這就是**自迴歸循環**（**Autoregressive Loop**）。

---

### Prefill 階段——處理你的輸入

你按下 Enter 後的第一件事：模型把你的所有輸入 Token 一次性送入 Transformer。

```
Prefill（預填充）的工作：

  1. 所有輸入 Token 同時通過所有 Transformer Block
  2. 在每一層建立 KV Cache（快取 K 和 V 向量）
  3. 取出最後一個位置的隱藏狀態
  4. 通過 LM Head → Logits → 取樣 → 第一個輸出 Token

CASE A：18 個 Token 同時通過 32 層
  每層注意力矩陣：[18 × 18] → 可以平行計算
  計算瓶頸：GPU 的算力（Compute-Bound）

CASE B：12 個 Token 同時通過 40 層
  DeltaNet 層：逐 Token 更新狀態（無注意力矩陣）
  Full Attn 層：[12 × 12] 注意力矩陣
```

**Prefill 後建立的 KV Cache**：

```
CASE A — 32 層全部快取：
  每層 KV = 2 × 8 頭 × 18 Token × 128 維 × 2 bytes = 73.7 KB
  32 層 = 73.7 KB × 32 ≈ 2.3 MB

CASE B — 只有 10 層需要快取：
  Full Attn 層：2 × 2 頭 × 12 Token × 256 維 × 2 bytes = 24.6 KB
  10 層 KV Cache = 24.6 KB × 10 ≈ 0.24 MB
  30 層 DeltaNet 固定狀態 = 16 × 128 × 128 × 2 bytes × 30 ≈ 15 MB

  CASE B 總計 ≈ 15.2 MB

注意：DeltaNet 的 15 MB 是固定的——不管輸入多長都是 15 MB。
CASE A 的 KV Cache 會隨輸入長度線性增長。

長輸入時差距更大：
  2000 Token 輸入：
    CASE A：2.3 MB × (2000/18) ≈ 262 MB
    CASE B：KV Cache 2.6 MB + 固定狀態 15 MB ≈ 18 MB
    CASE B 只需 CASE A 的 ~7%
```

---

### KV Cache——為什麼不用每次重算

**為什麼需要它**

自迴歸生成時，每一步只新增一個 Token。如果每一步都重新計算所有 Token 的 K 和 V，大量的計算是重複的——前面 Token 的 K/V 跟上一步完全一樣。

**它是什麼**

KV Cache 在 Prefill 時把每一層的 K 和 V 向量存起來。後續每生成一個新 Token，只需要計算新 Token 的 Q、K、V，然後把新的 K 和 V 追加到快取中。

```
第 1 步（Prefill）：
  計算 18 個 Token 的 K 和 V
  存入 KV Cache：K_cache = [18 × d_k], V_cache = [18 × d_v]

第 2 步（生成第 1 個 Token）：
  只計算新 Token 的 Q、K、V（1 個 Token）
  把新的 K、V 追加到快取：
    K_cache = [19 × d_k], V_cache = [19 × d_v]
  注意力：Q_new · K_cacheᵀ = [1 × d_k] × [d_k × 19] = [1 × 19]
  只需算 1 行注意力，不用重算整個 [19 × 19] 矩陣

第 3 步（生成第 2 個 Token）：
  K_cache = [20 × d_k]
  注意力：[1 × 20]

...

第 N 步：
  注意力：[1 × (18+N)]

每步只計算 1 個 Token 的前向傳播，復用所有歷史的 K/V。
```

**沒有 KV Cache**：第 N 步要重算整個 [(18+N) × (18+N)] 的注意力矩陣 → 浪費。
**有 KV Cache**：第 N 步只算 [1 × (18+N)] 的一行 → 高效。

**KV Cache 的代價**：隨著生成的 Token 越多，快取越大。

```
CASE A 的 KV Cache 增長：
  每生成 1 個 Token，快取增加：
    32 層 × 2(K+V) × 8 頭 × 128 維 × 2 bytes = 128 KB

  生成 200 個 Token → 128 KB × 200 = 25.6 MB
  生成 2000 個 Token → 128 KB × 2000 = 256 MB
  生成 128K Token → 128 KB × 128K ≈ 16 GB（VRAM 爆炸！）

CASE B 的 KV Cache 增長：
  每生成 1 個 Token，快取增加（只有 10 層 Full Attn）：
    10 層 × 2 × 2 頭 × 256 維 × 2 bytes = 20 KB

  生成 200 個 Token → 20 KB × 200 = 4 MB
  生成 2000 個 Token → 20 KB × 2000 = 40 MB
  生成 128K Token → 20 KB × 128K ≈ 2.5 GB

  KV Cache 是 CASE A 的 1/6，加上 DeltaNet 固定狀態 15 MB（不變）。
```

---

### Decode 階段——逐 Token 生成

Prefill 後進入 Decode 循環：

```
自迴歸循環：

  while True:
    1. 取出上一步生成的 Token
    2. 嵌入 → [1 × hidden_size]
    3. 通過 L 層 Transformer（只計算新 Token，讀 KV Cache）
    4. LM Head → logits → 處理管線 → 取樣 → 新 Token
    5. 如果新 Token 是 EOS 或達到最大長度 → 停止
    6. 否則 → 把新 Token 拼回序列，回到步驟 1
```

**Prefill vs Decode 的計算特性**：

```
Prefill：
  所有 Token 同時計算 → 大量平行工作 → Compute-Bound（計算瓶頸）
  GPU 的計算核心全部忙碌

Decode：
  每步只計算 1 個 Token → 幾乎沒有平行度 → Memory-Bound（記憶體瓶頸）
  瓶頸是從 VRAM 讀取權重和 KV Cache 的頻寬

這就是為什麼：
  Prefill 的速度看 GPU 的 TFLOPS（計算能力）
  Decode 的速度看 GPU 的 GB/s（記憶體頻寬）
```

**TTFT（Time to First Token）**：

```
TTFT = Prefill 時間 + 首次 Decode 時間

這是你按下 Enter 到看見第一個字出現的等待時間。

影響因素：
  輸入越長 → Prefill 計算量越大 → TTFT 越高
  100 Token 的 prompt → TTFT ≈ 20-50ms
  2000 Token 的 prompt → TTFT ≈ 100-200ms
  10000 Token 的 prompt → TTFT ≈ 500-1500ms

之後每個 Token 的生成速度 = Decode 速度（token/s）：
  CASE A（INT4, RTX 4090）：~237 tokens/s
  CASE B（INT4, RTX 4090）：~56 tokens/s

  CASE A 的 Decode 更快，因為模型權重更小（4 GB vs 18 GB），
  每步從 VRAM 讀取的資料量更少。
```

> 自迴歸循環的兩個階段本質不同：Prefill 一次處理所有輸入（Compute-Bound），Decode 逐步生成每個 Token（Memory-Bound）。KV Cache 是連接兩個階段的關鍵——它在 Prefill 時建立，在 Decode 時被反覆讀取，避免重複計算。

---

# 第 11 章：什麼時候停下來？

模型怎麼知道回答完了？三種停止條件：

---

### EOS Token（序列結束標記）

模型在訓練時見過每段文字的結尾都有 EOS Token，因此它學會了在「回答完了」的時候預測 EOS。

```
CASE A 的 EOS：<|eot_id|>（Token ID = 128009）
CASE B 的 EOS：<|im_end|>（Token ID = 248046）

當模型取樣出 EOS Token → 立即停止生成。
EOS 本身不會被輸出給使用者。
```

---

### 最大長度限制

```
即使模型不預測 EOS，也會在達到最大長度時強制停止。

  max_tokens / num_predict：你能設定的最大生成 Token 數
  max_position_embeddings：模型支援的最大序列長度
    CASE A：131,072（128K）
    CASE B：262,144（256K）

  如果輸入 + 輸出超過這個限制 → 強制停止
```

---

### 停止字串（Stop Strings）

```
你可以指定一組字串，當模型生成的文字包含這些字串時停止。

  例如：stop = ["\\n\\n", "。", "END"]

  模型生成 "...回答完畢。" → 偵測到 "。" → 停止

  常用於結構化輸出（JSON、程式碼區塊），
  確保模型在合適的邊界停止。
```

> LLM 透過三種機制停止生成：學會預測 EOS Token（最自然）、達到最大長度（安全網）、匹配停止字串（使用者控制）。

---

# 第 12 章：案例——端到端追蹤「什麼是機器學習？」

把所有環節串起來，完整追蹤同一個問題在兩個架構中的旅程。

```
═══════════════════════════════════════════════════════════════
階段 1：Tokenization
═══════════════════════════════════════════════════════════════

  CASE A（Llama）："什" "麼" "是" "機" "器" "學" "習" "？" → 8 Token
  CASE B（Qwen）："什麼" "是" "機器學習" "？" → 4 Token

  加上 Chat Template：
    CASE A：約 18 Token（含 BOS + 角色標記）
    CASE B：約 12 Token（分詞效率 + 簡潔模板）

═══════════════════════════════════════════════════════════════
階段 2：Embedding
═══════════════════════════════════════════════════════════════

  CASE A：18 Token → 查 [128,256 × 4,096] 表 → [18 × 4,096]
  CASE B：12 Token → 查 [248,320 × 2,048] 表 → [12 × 2,048]

═══════════════════════════════════════════════════════════════
階段 3：Prefill — 通過所有 Transformer Block
═══════════════════════════════════════════════════════════════

  CASE A — 32 層，每層結構相同：
    每層：RMSNorm → GQA(32Q/8KV, d=128) → 殘差
          → RMSNorm → Dense SwiGLU(4096→14336→4096) → 殘差
    注意力矩陣：[18 × 18] per head，32 層全部存 KV Cache
    輸出：[18 × 4,096]

  CASE B — 40 層，3:1 交替：
    DeltaNet 層(30)：Gated DeltaNet + MoE(9/256 Expert)
      不存 KV Cache，維護固定狀態 [16 × 128 × 128]
    Full Attn 層(10)：Softmax Attention(16Q/2KV) + MoE
      注意力矩陣：[12 × 12]，存 KV Cache
    輸出：[12 × 2,048]

  Prefill 後 KV 記憶體：
    CASE A：32 層 × KV(18 Token) ≈ 2.3 MB
    CASE B：10 層 KV(12 Token) + 30 層固定狀態 ≈ 15.2 MB
    （輸入很短，差異不大；長輸入時 CASE A 會爆炸性增長）

═══════════════════════════════════════════════════════════════
階段 4：LM Head
═══════════════════════════════════════════════════════════════

  CASE A：[18 × 4,096] → 取最後位置 → [1 × 4,096]
          × [4,096 × 128,256] → [128,256] logits

  CASE B：[12 × 2,048] → 取最後位置 → [1 × 2,048]
          × [2,048 × 248,320] → [248,320] logits

═══════════════════════════════════════════════════════════════
階段 5：Logits 處理 + 取樣
═══════════════════════════════════════════════════════════════

  設定：Temperature = 0.7, Top-p = 0.9

  兩個模型可能選出不同的第一個 Token：
    CASE A：選出 "機" → 1 個字
    CASE B：選出 "機器學習" → 整個詞（1 個 Token 就涵蓋！）

═══════════════════════════════════════════════════════════════
階段 6：Decode 循環
═══════════════════════════════════════════════════════════════

  CASE A — 每步 Decode：
    輸入 1 Token → 通過 32 層
    每層讀 KV Cache → 注意力 [1 × seq_len]
    → Dense FFN → 新 Token
    KV Cache 每步增長 128 KB
    速度：~237 t/s（INT4, RTX 4090）

  CASE B — 每步 Decode：
    輸入 1 Token → 通過 40 層
    DeltaNet 層(30)：更新固定狀態，不增長記憶體
    Full Attn 層(10)：讀 KV Cache → 注意力 [1 × seq_len]
    → MoE(Router → 9 Expert) → 新 Token
    KV Cache 每步只增長 20 KB
    速度：~56 t/s（INT4, RTX 4090）

  假設都生成 ~200 個 Token 後遇到 EOS → 停止

═══════════════════════════════════════════════════════════════
最終對比
═══════════════════════════════════════════════════════════════

                        CASE A              CASE B
  輸入 Token 數          18                  12
  模型權重(INT4)         3.8 GB              16.3 GB
  每 Token 計算量        ~16 GFLOPs          ~6 GFLOPs
  KV Cache(200 Token)    ~27 MB              ~4 MB + 15 MB 固定
  Decode 速度            ~237 t/s            ~56 t/s
  TTFT                   ~30 ms              ~80 ms
  生成品質               好（8B Dense）       更好（35B 知識量）

  CASE A 更快、更省記憶體 → 適合資源有限的場景
  CASE B 更聰明、更省 KV Cache → 適合需要品質和長上下文的場景
```

---

# 第 13 章：收尾

## 壓縮公式

**完整版**：LLM 的生成過程是一條管線——文字被 BPE 切成 Token，Token 查嵌入表變成向量，向量穿過幾十層 Transformer Block（注意力交換資訊 + FFN 處理資訊），最後一層的向量投影成詞彙表大小的分數，分數經過溫度、過濾、Softmax 變成機率，從機率中取樣出一個 Token。這個 Token 拼回輸入，整個流程重來一遍。如此循環直到模型說「完了」。

**口訣版**：切→查→疊→投→濾→選→拼→重來。

---

## 專有名詞速查表

### 分詞器（Tokenizer）
把原始文字拆成 Token 片段，再對應到整數 ID 的工具。
- 故事中：旅程的第一站——文字變成數字的入口
- 案例中：同一句話在 Llama（8 Token）和 Qwen（4 Token）中被切成不同數量

### BPE（Byte Pair Encoding, 位元組對編碼）
反覆合併高頻相鄰片段來建立詞彙表的演算法。
- 故事中：決定「什麼片段算一個 Token」的規則
- 案例中：Qwen 的 BPE 在中文上合併了更多整詞

### 嵌入矩陣（Embedding Matrix）
一張 [vocab_size × hidden_size] 的查找表，每行對應一個 Token 的向量。
- 故事中：Token ID 查表變向量的關鍵
- 案例中：Llama [128,256 × 4,096]，Qwen [248,320 × 2,048]

### RoPE（Rotary Position Embedding, 旋轉位置編碼）
通過旋轉 Q 和 K 向量注入位置資訊的方法。
- 故事中：讓模型知道每個 Token 在序列中的位置
- 案例中：Llama 所有層使用，Qwen 只在 Full Attention 層使用

### GQA（Grouped Query Attention, 分組查詢注意力）
多個 Q 頭共享一組 KV 頭，減少 KV Cache 大小。
- 故事中：在注意力品質和記憶體之間取得平衡
- 案例中：Llama 32Q/8KV，Qwen 16Q/2KV

### MoE（Mixture of Experts, 混合專家機制）
把 FFN 拆成多個小 Expert，每個 Token 只啟動其中幾個。
- 故事中：用稀疏計算實現「大模型知識、小模型計算量」
- 案例中：Qwen 每層 256 Expert，每 Token 啟動 9 個

### Router（路由器）
MoE 中決定每個 Token 送去哪幾個 Expert 的輕量線性層。
- 故事中：MoE 的「分診台」
- 案例中：W_router [2,048 × 256]，輸出 256 個分數

### DeltaNet（Gated Delta Network）
用固定大小狀態矩陣取代 KV Cache 的線性注意力機制。
- 故事中：用 O(1) 記憶體處理每個 Token，不隨序列增長
- 案例中：Qwen 的 30 層使用，固定狀態 [128 × 128] per head

### Logits（原始分數）
LM Head 輸出的未正規化分數，每個 Token 對應一個值。
- 故事中：模型對「下一個 Token 是什麼」的原始看法
- 案例中：Llama 128,256 個分數，Qwen 248,320 個分數

### Temperature（溫度）
控制機率分佈尖銳度的超參數：低溫更確定，高溫更隨機。
- 故事中：logits / T，改變分佈的形狀
- 案例中：T=0.7 是日常對話的常用值

### Top-p（Nucleus Sampling, 核取樣）
只保留累積機率達到 p 的最小 Token 集合。
- 故事中：自適應地根據模型信心調整候選數
- 案例中：Top-p=0.9 是最常用的設定

### KV Cache（鍵值快取）
快取每層 Attention 的 K 和 V 向量，避免重複計算。
- 故事中：自迴歸循環的效率關鍵
- 案例中：Llama 32 層全快取（128 KB/token），Qwen 10 層（20 KB/token）

### Prefill（預填充）
把所有輸入 Token 一次性送入模型，建立 KV Cache 的階段。
- 故事中：按下 Enter 後的第一件事
- 案例中：Compute-Bound，所有 Token 平行計算

### Decode（解碼）
逐步生成輸出 Token 的階段，每步只計算一個新 Token。
- 故事中：第一個字出現後，一個接一個跳出來的過程
- 案例中：Memory-Bound，速度取決於 VRAM 頻寬

### TTFT（Time to First Token, 首 Token 延遲）
從收到請求到輸出第一個 Token 的耗時。
- 故事中：你按下 Enter 後的「等待時間」
- 案例中：輸入越長 → Prefill 越久 → TTFT 越高

### EOS（End of Sequence, 序列結束標記）
模型預測出這個 Token 時停止生成。
- 故事中：模型說「我講完了」的信號
- 案例中：Llama 用 `<|eot_id|>`，Qwen 用 `<|im_end|>`

---

## 文件資訊

- 最後更新日期：2026-03-23
- 延伸閱讀：
  - 文件一《從傳統機器學習到 Transformer》——理解 Transformer 每個組件的設計動機
  - "Attention Is All You Need" (Vaswani et al., 2017) — https://arxiv.org/abs/1706.03762
  - The Illustrated GPT-2 — Jay Alammar — https://jalammar.github.io/illustrated-gpt2/
  - Andrej Karpathy "Let's build GPT from scratch" — https://www.youtube.com/watch?v=kCc8FmEb1nY
  - Hugging Face NLP Course — https://huggingface.co/learn/nlp-course/chapter1/4
  - 李沐《動手學深度學習》— https://zh.d2l.ai/
- 關鍵字：Token、Tokenization、BPE、Embedding、RoPE、Transformer Block、GQA、MoE、Router、Expert、DeltaNet、Hybrid Attention、Dense FFN、SwiGLU、LM Head、Logits、Temperature、Top-k、Top-p、Min-p、Softmax、Sampling、KV Cache、Prefill、Decode、TTFT、Autoregressive、EOS
- 關鍵知識：
  - 分詞器的詞彙表決定了文字被切成多少個 Token，直接影響計算效率
  - 嵌入層是查表操作，把離散的整數映射成連續的高維向量
  - Dense 架構每個 Token 用全部參數，MoE 架構每個 Token 只用一小部分
  - Hybrid Attention 混合線性注意力（高效）和 Softmax 注意力（精確）
  - Logits 處理管線把原始分數塑形成可取樣的機率分佈
  - KV Cache 是自迴歸生成的效率關鍵，也是長序列的記憶體瓶頸
  - Prefill 是 Compute-Bound（算力瓶頸），Decode 是 Memory-Bound（頻寬瓶頸）
  - MoE 模型的記憶體由總參數決定，速度由活躍參數決定
