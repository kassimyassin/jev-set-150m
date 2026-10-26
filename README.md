---
license: apache-2.0
library_name: transformers
tags:
  - decision-index
  - jev-decision-index
  - modernbert
  - ettin
  - set-attention
  - permutation-equivariant
---

# `JEV-Set-150M` (`149,936,905` Parameters) for the Jev Decision Index (`v0.4.1`)

**`JEV-Set-150M`** is a single-checkpoint decision encoder (`149,936,905` total served parameters — `63,095` below the strict `150,000,000` ceiling) engineered for the [Jev Decision Index (`v0.4.1`)](https://huggingface.co/spaces/multimodalart/jev-decision-index) (`apolinario/decision-index` commit `25c6ec7b6b5f44a216c63c3a700de3a9c6d6699d`).

It resolves the **Scoring-Topology Dilemma** of existing $\le 150\text{M}$ models (`tasksource-decider-nano`, `Crivo-150M`, `Mola-150M`, `Dinah-1`, `PICK-150M`) by combining:
1. **Backbone (`149,014,272` BF16 params):** `cross-encoder/ettin-reranker-150m-v1` (22-layer ModernBERT/Ettin architecture, $d=768$, 12 heads, GeGLU $d_{\text{ff}}=1152$).
2. **Tree-Masked Shared-State Virtual-Pair Packing + 32k Chunked Windowing:** Encodes the shared `state` prefix (`[CLS] state [SEP]`) once at positions `0 .. P-1` and packs all $K$ candidate options (`Question: <instructions>\nOption: <o_k> [SEP]`) at reset virtual-pair positions `P .. P+L_k-1` with a 2D block-diagonal tree attention mask. Ultra-long states ($>6,144$ tokens) are split into overlapping 512-token-anchored windows up to `32,768` tokens so **0 requests are ever truncated or refused as `Unsupported`**.
3. **`EquivariantSetInteractionHead` (`922,633` FP32 params):** Projects pooled context and option representations to $d_{\text{set}} = 256$, applies an **8-head Permutation-Equivariant Set-Attention layer (zero positional encodings across options)**, and scores the comparative feature vector $[\mathbf{u}_k;\, \mathbf{c};\, \mathbf{u}_k \odot \mathbf{c};\, \mathbf{u}_k - \bar{\mathbf{u}}] \in \mathbb{R}^{1024}$ fused via a zero-initialized residual gate $\tanh(\gamma)$ with the pretrained reranker linear scorer, followed by cardinality-aware temperature scaling $T(K, r)$.

---

## 1. Exact Parameter Budget (`149,936,905 <= 150,000,000`)

| Component | Tensor Shapes | Exact Parameter Count | Precision |
| :--- | :--- | ---: | :---: |
| `backbone.embeddings` | `50,368 × 768` + `LayerNorm(768, bias=False)` | `38,683,392` | `BF16` |
| `backbone.layers.0` | `Wqkv(2304×768)` + `Wo(768×768)` + `LN(768)` + `Wi(2304×768)` + `Wo(768×1152)` | `5,014,272` | `BF16` |
| `backbone.layers.1..21` | `21 × (LN(768) + Wqkv + Wo + LN(768) + Wi + Wo)` | `105,315,840` | `BF16` |
| `backbone.final_norm` | `LayerNorm(768, bias=False)` | `768` | `BF16` |
| **Backbone Subtotal** | **`cross-encoder/ettin-reranker-150m-v1`** | **`149,014,272`** | `BF16` |
| `head.type_emb` | `Embedding(3, 256)` | `768` | `FP32` |
| `head.ctx_proj` + `opt_proj` | `2 × Linear(768, 256, bias=True)` | `393,728` | `FP32` |
| `head.ctx_norm` + `opt_norm` | `2 × LayerNorm(256, bias=True)` | `1,024` | `FP32` |
| `head.set_attn` + `set_norm` | `MultiheadAttention(256, 8, bias=True)` + `LayerNorm(256)` | `263,680` | `FP32` |
| `head.base_scorer` | `Linear(768, 1, bias=True)` | `769` | `FP32` |
| `head.scorer_mlp` | `Linear(1024, 256, bias=True)` + `GELU` + `Linear(256, 1, bias=True)` | `262,657` | `FP32` |
| `head` gates & temperatures | `set_gate(1)` + `log_temperature(3)` + `cardinality_slope(3)` | `7` | `FP32` |
| **Head Subtotal** | **`EquivariantSetInteractionHead`** | **`922,633`** | `FP32` |
| **TOTAL SERVED PARAMETERS** | | **`149,936,905`** | `BF16` + `FP32` |
| **Safety Margin Below 150M** | $150,000,000 - 149,936,905$ | **`63,095`** | — |

---

## 2. Quickstart & Local Verification

```powershell
# 1. Verify exact parameter count (149,936,905) and float32 permutation equivariance (< 1e-5)
python scripts/verify_params.py

# 2. Run unit & edge-case test suite (K=512 options, 32k context windowing, official validator)
python -m pytest tests/test_budget_and_equivariance.py tests/test_engine_edge_cases.py -v

# 3. Run offline smoke test of data prep, teacher distillation, training, souping, and official evaluation
python scripts/prepare_data.py --smoke-test --num-smoke 32
python scripts/generate_teacher_logits.py --input-jsonl data/processed/stage2_distill_train.jsonl --output-jsonl data/processed/stage2_distill_train.jsonl --mock-teachers
python scripts/train.py --config configs/stage1_ood.yaml --max-steps 4
python scripts/soup_and_calibrate.py --demo-calibration
python scripts/evaluate_official.py --smoke-test
```

---

## 3. Running on Google Colab Pro (`A100 80GB/40GB`, `L4 24GB`, `T4 16GB`)

All stage configuration files (`configs/stage1_ood.yaml`, `configs/stage2_distill.yaml`, `configs/stage3_equilibrium.yaml`) and `scripts/train.py` include automatic hardware profile detection (`detect_colab_gpu_profile`) that preserves the exact **`32`-request effective batch size** regardless of which GPU Colab Pro allocates:

| Colab Pro GPU | VRAM | `per_device_requests` (Stage 1 / Stage 2–3) | `gradient_accumulation_steps` | `gradient_checkpointing` | `TF32` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **NVIDIA A100-SXM4-80GB** | `80 GB` | `8` / `4` | `4` / `8` | `false` | `true` |
| **NVIDIA A100-SXM4-40GB** | `40 GB` | `4` / `2` | `8` / `16` | `false` / `true` | `true` |
| **NVIDIA L4** | `24 GB` | `2` / `1` | `16` / `32` | `true` | `true` |
| **NVIDIA T4** | `16 GB` | `1` / `1` | `32` / `32` | `true` | `false` |

### Colab Pro Execution Options
1. **Fast Pilot Notebook (Recommended First Step):** Upload and open [`notebooks/colab_pilot.ipynb`](./notebooks/colab_pilot.ipynb) in Google Colab Pro (`Runtime -> Change runtime type -> A100 GPU` or `L4 GPU`). Runs real weights verification, pilot data build, teacher logits, and measured Exp A/B.
2. **Full Pipeline Notebook:** Upload and open [`notebooks/colab_pro_pipeline.ipynb`](./notebooks/colab_pro_pipeline.ipynb) for full-scale multi-epoch training, 3-seed model souping, and official evaluation.
3. **One-Command Shell Launcher:**
   ```bash
   # Run fast pilot (zero synthetic data, real teachers, measured Exp A/B)
   bash scripts/run_colab_pro.sh pilot

   # Run full pipeline with Google Drive persistence
   bash scripts/run_colab_pro.sh
   ```

### Upload to Google Drive / Colab
A clean, upload-ready archive is prepared at:
`C:\Users\lenovo\.vscode\jev-set-150m-upload.zip` (~1.2 MB, excludes `.venv`, `.git`, `__pycache__`, and test caches).
To use on Google Colab:
1. Upload `jev-set-150m-upload.zip` to your Google Drive root or `/content/`.
2. In Colab:
   ```bash
   !unzip -q /content/drive/MyDrive/jev-set-150m-upload.zip -d /content/jev-set-150m
   %cd /content/jev-set-150m
   ```
3. Run `notebooks/colab_pilot.ipynb` or `notebooks/colab_pro_pipeline.ipynb`.

---

## 4. Moving Your Workspace & Chat Inside `jev-set-150m` in Antigravity

To move your active workspace into `C:\Users\lenovo\.vscode\jev-set-150m`:
1. **Option A (Multi-Root Workspace — Keeps Current Chat Active Immediately):**
   - In Antigravity / VS Code, click **`File -> Add Folder to Workspace...`** and select `C:\Users\lenovo\.vscode\jev-set-150m`.
   - This keeps your current conversation history open while giving the agent full workspace context in `jev-set-150m`.
2. **Option B (Dedicated Window + Persistent `PLAN.md`):**
   - Click **`File -> Open Folder...`** and select `C:\Users\lenovo\.vscode\jev-set-150m`.
   - The full 20-point specification and live Phase 1–4 checklist are saved in [`PLAN.md`](./PLAN.md). Starting a prompt in the new window with `"Read PLAN.md and continue from Phase 2"` immediately restores full context.

### Which Skill Mode Is Better: `/boost` vs. `/teamwork-preview`?
- **Use `/boost` for:** Tight mathematical/architectural implementation, debugging kernels, tuning loss functions (`losses.py`, `modeling_jev_set.py`), and single-script optimization where deep, unified reasoning without inter-agent coordination overhead is best.
- **Use `/teamwork-preview` for:** Parallel multi-component execution (such as scaffolding multiple modules + running verification suites in parallel, or running data preparation + teacher logit scripts + evaluation checks across subagents).

---

## 5. Official Evaluation (`apolinario/decision-index` @ `25c6ec7b`)

```bash
# Run official evaluation on the frozen v0.3/v0.4 public panel
python -m decision_index run \
  --engine src.jev_set.engine:JEVSetEngine \
  --option checkpoint_dir=checkpoints/jev-set-150m-final \
  --edition 0.3 \
  --out runs/jev-set-150m-v04

# Score the run with the official scorer
python -m decision_index score \
  --results runs/jev-set-150m-v04/results.jsonl \
  --edition 0.3 \
  --out runs/jev-set-150m-v04
```
