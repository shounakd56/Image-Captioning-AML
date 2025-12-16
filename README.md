# Image Captioning with CNN–LSTM and CNN–Transformer Architectures

This repository presents a systematic study of **image captioning models** using **CNN encoders** (ResNet18, MobileNetV2) combined with **LSTM** and **Transformer** decoders.  
We analyze architectural choices, training regimes, decoding strategies, and explainability, with a strong focus on **caption quality**, **degenerate repetition**, and **efficiency trade-offs**.

---

## 📌 Project Overview

- **Task:** Image Captioning  
- **Encoders:** ResNet-18, MobileNetV2 (ImageNet-pretrained)  
- **Decoders:** LSTM, Transformer  
- **Training Modes:**  
  - End-to-end CNN fine-tuning  
  - Feature-cache (frozen CNN, compute-light)  
- **Decoding:** Greedy search, Beam search (beam=3)  
- **Evaluation:** BLEU-4, METEOR, caption length statistics, degenerate repetition analysis  

---

## 🧹 Preprocessing

### Images
- Resized to **224 × 224**
- ImageNet normalization applied

### Captions
- Word-level tokenization  
- Vocabulary built **only on training set**
- Vocabulary size: **2807**
- Special tokens: `<pad>`, `<bos>`, `<eos>`, `<unk>`
- Maximum caption length: **22 tokens**

### Saved Statistics
- Train / Validation / Test text statistics table
- Vocabulary coverage
- OOV percentage
- Caption length histograms

---

## 📊 Text Statistics

### Train Set
- **Total words:** 504,917  
- **OOV:** 0 (0.00%)  
- **Coverage:** 100.00%  
- **Mean length:** 14.00 ± 3.12  

### Validation Set
- **Total words:** 53,780  
- **OOV:** 468 (0.87%)  
- **Coverage:** 99.13%  
- **Mean length:** 12.20 ± 3.07  

### Test Set
- **Total words:** 65,979  
- **OOV:** 842 (1.28%)  
- **Coverage:** 98.72%  
- **Mean length:** 14.46 ± 3.79  

---

## 🧠 Model Architectures

### CNN Encoder
- Backbones: **ResNet18**, **MobileNetV2**
- Classification head removed
- Global Average Pooling used
- Training modes:
  - **Feature-cache:** CNN fully frozen
  - **Fine-tune:** All layers frozen except last block

---

### LSTM Decoder
- Embedding dimension: **512**
- LSTM layers: **2**
- Hidden size: **512**
- Image features projected to initial hidden state
- Training:
  - Teacher forcing
  - Cross-entropy loss
  - `<pad>` ignored via `ignore_index`
- Inference:
  - Greedy decoding
  - Beam search (beam=3)

---

### Transformer Decoder
- Layers: **4**
- Heads: **8**
- Model dimension: **512**
- Causal mask + key padding mask
- Image feature projected to a **short memory sequence**
  - **1-token memory**
  - **4-token memory**
- LayerNorm applied to memory tokens

---

## 📈 Results

| Encoder + Decoder | Mode | BLEU-4 | Caption Length (mean ± std) | Degenerate Repetitions (%) |
|------------------|------|--------|-----------------------------|----------------------------|
| ResNet18 + LSTM | CNN finetune | 0.1719 | 11.21 ± 2.11 | 0.00 |
| ResNet18 + Transformer | CNN finetune | 0.1916 | 15.33 ± 1.73 | 26.59 |
| MobileNet + LSTM | CNN finetune | 0.1998 | 11.05 ± 2.14 | 0.00 |
| MobileNet + Transformer | CNN finetune | 0.1906 | 16.05 ± 1.36 | 72.37 |
| ResNet18 + LSTM | Feature-cache | 0.1759 | 11.54 ± 2.24 | 0.00 |
| MobileNet + LSTM | Feature-cache | 0.1812 | 12.22 ± 1.69 | 23.96 |
| ResNet18 + Transformer (4-token) | Feature-cache | 0.1800 | 16.54 ± 1.66 | 26.94 |
| MobileNet + Transformer (4-token) | Feature-cache | 0.1854 | 16.25 ± 1.58 | 18.55 |
| ResNet18 + Transformer (1-token) | Feature-cache | 0.1728 | 16.55 ± 1.79 | 20.15 |
| MobileNet + Transformer (1-token) | Feature-cache | **0.2105** | 13.73 ± 2.11 | **77.24** |

---

## 🔬 Experiments & Extensions

### Backbone Swap: ResNet18 ↔ MobileNet

**ResNet18**
- LSTM: BLEU-4 = 0.1719, 0% repetition, 17 it/s
- Transformer: BLEU-4 = 0.1916, 26.59% repetition, 15 it/s

**MobileNet**
- LSTM: BLEU-4 = 0.1998, 0% repetition, 17 it/s
- Transformer: BLEU-4 = 0.1906, 72.37% repetition, 14–15 it/s

**Key Observations**
- MobileNet + LSTM produces the most coherent captions
- Transformers achieve longer captions but are prone to repetition
- Lightweight backbones amplify degeneration in Transformers

---

### Transformer Memory: 1-token vs 4-token

**ResNet18**
- 1-token: BLEU-4 = 0.1728, 20.15% repetition
- 4-token: BLEU-4 = 0.1800, 26.94% repetition

**MobileNet**
- 1-token: BLEU-4 = 0.2105, 77.24% repetition
- 4-token: BLEU-4 = 0.1854, 18.55% repetition

**Takeaway**
- Increasing memory length improves coherence
- 4-token memory significantly reduces repetition for MobileNet
- Best trade-off between accuracy and fluency achieved with **4-token memory**

---

## 🧪 Evaluation, Analysis & Explainability

### Metrics
- BLEU-4
- METEOR
- Caption length (mean ± std)
- Degenerate repetition:
  - ≥3 identical tokens consecutively

---

### Qualitative & Slice Analysis
- 10 successful + 10 failure cases visualized
- Error slices analyzed:
  - Bright vs dark images
  - Short (≤8 tokens) vs long (≥16 tokens) captions
- Per-slice BLEU-4 deltas plotted

---

### Explainability

#### Grad-CAM / Grad-CAM++
- Applied on last CNN block w.r.t.:
  - EOS token logit
  - Salient content-word logit
- Visualized:
  - 3 correct predictions
  - 3 failure cases

#### Text Importance
- **LSTM:** Token occlusion → Δlogit / Δloss
- **Transformer:** Averaged last-layer attention maps

---

### Mis-Caption Case Studies
- Deep analysis of 3 failure examples
- Identified issues:
  - Object hallucination
  - Over-repetition
  - Poor grounding
- Proposed non-advanced fixes

---

## 🚀 Key Takeaways

- **LSTM decoders** are robust and repetition-free
- **Transformers** offer richer captions but require memory design to avoid degeneration
- **4-token memory** is a strong design choice for lightweight CNN backbones
- Feature-cache training provides a strong compute–performance trade-off

---

## 📌 Future Work
- Coverage loss / repetition penalty
- Contrastive vision–language alignment
- Scheduled sampling
- CIDEr-optimized training
- Larger-scale Transformer decoders
