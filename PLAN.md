# Project Requirement Document

## Project: **InterpretabilityWorkbench**

### Subtitle: *Interactive Mechanistic‑Interpretability Workbench for LLMs*

---

### 1. Background & Motivation

Large language models (LLMs) routinely surprise practitioners with hidden capabilities and failure modes. Sparse‑autoencoder (SAE) work has shown that interpretable features exist, but existing tooling is **offline** and **static**—researchers must train SAEs, inspect logs, then re‑train if they wish to test an edit.\
**InterpretabilityWorkbench** converts this pipeline into a *live laboratory*: users can record activations, train SAEs, browse discovered features, and **hot‑patch** model weights on‑the‑fly to observe real‑time effects on token probabilities.

### 2. Goals & Non‑Goals

| Area                 | In Scope                                             | Out of Scope                                           |
| -------------------- | ---------------------------------------------------- | ------------------------------------------------------ |
| **Interpretability** | SAE feature discovery; provenance graph; token cloud | Gradient‑based attribution beyond Integrated Gradients |
| **Editing**          | LoRA edits toggled per feature                       | Full model retraining                                  |
| **Models**           | HF decoder‑only (7B‑13B) using 8‑bit or 4‑bit quant  | Proprietary Anthropic models                           |
| **UX**               | Web UI (FastAPI + React/Plotly)                      | Desktop native app                                     |

### 3. Success Criteria (KPIs)

| KPI                       | Target              | Measurement              |
| ------------------------- | ------------------- | ------------------------ |
| SAE recon loss (held‑out) | ≤ 0.15              | `eval.py` script         |
| UI click → logits update  | < 400 ms            | WebSocket timestamp diff |
| Feature provenance depth  | ≥ 2 upstream layers | UI trace panel           |

### 4. User Personas & Journeys

- **Alignment Researcher Alice** records activations on a Trojan‑finetuned Mistral‑7B, trains SAEs overnight, then uses the UI to locate a “trigger detector” feature and flips the patch toggle—verifying that malicious outputs disappear in real time.
- **ML Engineer Ben** imports a colleague’s `saes/` + `lora/` files, attaches them to a local Llama‑3‑8B, and exports a *safe LoRA* for prod deployment—without re‑training.

### 5. Functional Requirements

1. **Activation Recorder**
   - CLI: `microscope trace --model <ckpt> --layer 20 --out acts.parquet`
   - Streaming Parquet writes; supports dataset shards.
2. **SAE Trainer**
   - Lightning module; adjustable β‑sparsity loss; automatic LR finder.
3. **Feature Explorer UI**
   - Feature table (ID, sparsity, top tokens).
   - Token cloud (hover = highlight).
   - Logit bar‑chart updates via WebSocket after patch.
4. **Live LoRA Patching**
   - `lora_patch.py` builds rank‑r (4 ≤ r ≤ 16) edits for selected neurons.
   - Hot‑swap onto running model with no restart.
5. **Provenance Graph**
   - BFS traversal two layers upstream; D3.js force‑directed panel.
6. **Export / Import**
   - `.safetensors` for SAE weights; JSON manifest for LoRA patches.

### 6. Non‑Functional Requirements

- **Compute Footprint** – fits on 1×A100‑40 GB *or* 2×RTX‑4090 with 4‑bit QLoRA.
- **Security** – No activation data leaves host machine. Optional W&B disabled by default.
- **Licence** – MIT for code; model checkpoints under original licences.

### 7. Technical Architecture

```
microscope-x/
├── cli.py              # entry‑point (`trace|train|ui`)
├── trace.py            # forward hooks → Parquet
├── sae_train.py        # LightningModule + Trainer
├── lora_patch.py       # build/apply edits
├── server/
│   ├── api.py          # FastAPI routes
│   └── websockets.py   # bidirectional stream model↔UI
├── ui/
│   ├── public/         # React build
│   └── src/
└── tests/
```

*Data Flow*

1. **Trace** → Parquet
2. **Train** → SAE weights
3. **UI** fetches weights → displays features
4. User toggles patch → WebSocket → LoRA injected → Model forward scores returned


### 8. Sanity Checklist (why this is feasible)

| Concern                                                   | Feasibility check                                                                                                     | Existing component you can reuse                                                |
| --------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| **Recording mid-layer activations at scale**              | Hugging Face hooks + `torch.no_grad()`; streaming to Parquet keeps RAM low                                            | `transformer_lens.ActivationCache`, `datasets.arrow`                            |
| **Training sparse autoencoders without blowing GPU VRAM** | Train *one layer at a time* with 4-bit quantized frozen backbone; Lightning handles gradient checkpointing            | 2024 SAE repo from Nethermind & OpenAI uses exactly this trick on consumer GPUs |
| **Live LoRA hot-patching**                                | LoRA weights are just low-rank adapters; swapping them is a single `state_dict` load + re-register 2 small matrices   | `peft.inject_adapter_in_model()` (0.4 ms for 7-B model)                         |
| **Web-socket latency < 400 ms**                           | All the compute happens *before* you stream to the browser; the WS payload (vector of 50 logits + metadata) is < 4 kB | FastAPI+Uvicorn WS benchmark ≈ 5 k req/s on localhost                           |
| **Ram usage with 7-B parameter model + SAE + LoRA**       | 7-B at 4-bit ≈ 7 GB, SAE layer (\~50 MB) + LoRA (< 30 MB) ⇒ fits in 12 GB GPU; add safety margin with 24 GB card      | BitsAndBytes 4-bit + PEFT                                                       |
| **Token-probability bar chart updates**                   | One forward pass to get next-token logits (pre-softmax), send top-K; no generation loop needed                        | `model(input_ids, use_cache=False).logits[:, -1, :]`                            |
| **Multi-layer provenance graph**                          | Use `torch.fx` or attention-pattern map already exported by TransformerLens; render with D3.js                        | `transformer_lens.utils.graph_structure()`                                      |
| **Two-month delivery**                                    | Hooks & viewers exist; the novelty is gluing them + UI polish                                                         | 2–3 weeks MVP; remainder for UX & docs                                          |


### 9. Milestone Plan (8 Weeks)

| Week | Deliverable                                 |
| ---- | ------------------------------------------- |
| 1    | Recorder + minimal CLI                      |
| 2    | SAE trainer on single layer with unit tests |
| 3    | Multi‑layer support + recon metrics script  |
| 4    | UI prototype (feature table & token cloud)  |
| 5    | Live LoRA patch end‑to‑end demo             |
| 6    | Provenance graph & latency profiling        |
| 7    | CI, docs, Dockerfile                        |
| 8    | v0.1 release + demo video                   |

### 10. Risks & Mitigation

| Risk                    | Impact         | Mitigation                               |
| ----------------------- | -------------- | ---------------------------------------- |
| OOM during SAE training | Blocker        | Default to 4‑bit; gradient checkpointing |
| WS latency > target     | UX frustration | Co‑locate UI+model; compress payload     |
| Feature UI overwhelm    | Usability      | Pagination & search filters              |

### 11. Acceptance Criteria

- End‑to‑end tutorial (`tutorial.ipynb`) yields <400 ms patch latency.
- Unit + integration tests green on CI (CPU stub).
- README shows GIF of live editing.

