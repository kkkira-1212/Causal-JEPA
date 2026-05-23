# Causal-JEPA

Self-supervised causal structure discovery for multivariate time series. Combines a JEPA-style predictive objective with a differentiable causal graph (NOTEARS) over a frozen TTM r2 backbone.

Three cross-attention predictors (`P_self`, `P_pa`, `P_non`) compete: `P_pa` predicts target patches from parent-aggregated representations, `P_non` from non-parents. Their relative error drives the causal adjacency matrix `Theta` via a log-ratio adversarial loss. See `ISSUES.md` for architecture details and experiment log.

---

## Setup

```bash
pip install torch>=2.1.0 transformers>=4.40.0
pip install numpy pandas pyyaml scipy matplotlib safetensors huggingface_hub
pip install "tsfm_public[notebooks]"   # IBM Granite TTM
pip install muon                        # optimizer for Theta
```

TTM weights download automatically on first run (`ibm-granite/granite-timeseries-ttm-r2`, revision `512-96-ft-r2.1`).

---

## Data

Pre-generated dataset included at `data/scm_d5_n30_t1500_dag42.npz` (D=5, N=30 series, T=1500, dag_seed=42). To regenerate:

```bash
python data/generate_dataset.py --n_series 30 --num_vars 5 --num_timesteps 1500 --dag_seed 42
```

---

## Training

```bash
# Full Causal-JEPA
python train.py --dataset scm_file

# Plain JEPA baseline (no causal layer)
python train.py --dataset scm_file --baseline

# Theta fixed at Granger init
python train.py --dataset scm_file --fixed-graph

# ETTh1 (no ground-truth graph)
python train.py --dataset etth1
```

All hyperparameters are in `configs/default.yaml`.

---

## Repository Structure

```
configs/default.yaml          hyperparameters
data/
  cauker_scm.py               SCM dataset generator (CAUKER-based)
  generate_dataset.py         CLI to generate and save .npz datasets
  etth.py                     ETTh1/ETTh2 loader
losses/losses.py              L_JEPA, L_ADV, L_DAG, L_sparse
models/
  causal_jepa.py              top-level model
  causal_layer.py             CausalLayer + TemperatureScheduler
  encoder.py                  TTMEncoder (TTM r2 wrapper)
  predictors.py               PatchPredictors (P_self, P_pa, P_non)
  downstream.py               linear probe head
  granger_init.py             GPU-batched Granger init for Theta
vendor/CauKer/                GP kernel bank and DAG generator
train.py                      training + evaluation script
ISSUES.md                     architecture documentation and experiment log
```

---

## References

- [I-JEPA (Assran et al., 2023)](https://arxiv.org/abs/2301.08243)
- [NOTEARS (Zheng et al., 2018)](https://arxiv.org/abs/1803.01422)
- [CAUKER](https://github.com/ckassaad/causalkernel)
- [IBM Granite TTM](https://huggingface.co/ibm-granite/granite-timeseries-ttm-r2)
- [Muon optimizer](https://github.com/KellerJordan/Muon)
