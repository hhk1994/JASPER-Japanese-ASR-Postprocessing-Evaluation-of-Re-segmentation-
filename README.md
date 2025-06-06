# JASPER-Japanese-ASR-Postprocessing-Evaluation-of-Re-segmentation-
# Japanese Tokenizer Comparison

This project implements a complete pipeline for Japanese automatic speech recognition (ASR) followed by language model (LM) post-processing. It aims to compare how different tokenization strategies (character-level and Sudachi modes A/B/C) affect the performance of downstream LSTM-based language models in refining raw ASR output.

---

## 📁 Project Structure

```
.
├── Stage1_NeMo_transcription.py        # NeMo Conformer CTC ASR inference script
├── Japanese_Word_Segmentation.py       # Tokenization using SudachiPy
├── train_lstm_lm.py                    # LSTM LM training script
├── batch_lm_inference.py               # Inference using each LM
├── wer_eval_domains.py                 # WER evaluation per domain per mode
├── run_pipeline_chain.sh               # SLURM-linked training→inference→WER
├── run_stage1_transcribe_all.sh        # Submit 1 ASR job per domain
├── run_tokenization_all.sh             # Submit 1 tokenization job per mode
├── slurm_scripts/
│   ├── stage1_nemo_asr_one.sh          # Per-domain ASR job
│   ├── tokenize_job.sh                 # Per-mode tokenization job
│   ├── train_job.sh                    # Per-mode LM training job
│   ├── infer_job.sh                    # Per-mode LM inference job
│   └── wer_job.sh                      # Final WER evaluation job
├── logs/                               # SLURM logs
└── data/                               # All output and intermediate files
```

---

## 🔁 Pipeline Overview

### 1. **ASR Transcription (Stage 1)**

* Uses NeMo's `stt_ja_conformer_transducer_large`
* Runs one transcription job per domain
* Saves hypotheses to: `/mnt/data/asr_raw_hypotheses_<domain>.txt`

```bash
bash run_stage1_transcribe_all.sh
```

---

### 2. **Tokenization (Stage 2)**

* Tokenize all ASR hypotheses using different tokenization modes: `char`, `sudachi_A`, `sudachi_B`, `sudachi_C`
* Output files: `/mnt/data/train_<mode>.txt`

```bash
bash run_tokenization_all.sh
```

---

### 3. **LSTM Language Model Training**

* Trains one LSTM LM per tokenization mode
* Input: `/mnt/data/train_<mode>.txt`
* Output: `lstm_lm_<mode>.pth`

Triggered automatically by:

```bash
bash run_pipeline_chain.sh
```

---

### 4. **LM Inference (Post-processing)**

* Each trained LM corrects tokenized ASR hypotheses
* Input: tokenized ASR files
* Output: `/mnt/data/asr_lm_outputs_real/output_<mode>.txt`

---

### 5. **WER Evaluation (Per-Domain)**

* Computes domain-separated WER after LM post-processing
* Input: ground-truth + LM-refined hypotheses

```bash
python3 wer_eval_domains.py
```

---

## 🧪 Domain and Tokenization Modes

* **Domains**:

  * `navigation`, `messaging`, `music_media`, `General_Knowledge`, etc.
* **Tokenization Modes**:

  * `char`, `sudachi_A` (fine), `sudachi_B` (medium), `sudachi_C` (coarse)

---

## 📊 Outputs

| File                       | Description                    |
| -------------------------- | ------------------------------ |
| `asr_raw_hypotheses_*.txt` | Raw ASR output per domain      |
| `train_<mode>.txt`         | Tokenized text for LM training |
| `lstm_lm_<mode>.pth`       | Trained LSTM LM checkpoint     |
| `output_<mode>.txt`        | LM-refined transcription       |
| `logs/`                    | SLURM logs                     |

---

## 🚀 To Run Entire Pipeline

```bash
bash run_pipeline_chain.sh
```

This will train all LMs, run inference, and evaluate WER automatically using SLURM job chaining.

---

## 🧠 Future Enhancements

* Add BLEU/CER evaluation
* Beam search decoding in LM
* Evaluate on more diverse or user-specific domains
* Export WER to CSV/Excel for reporting

---

## 🛠 Dependencies

* [NVIDIA NeMo Toolkit](https://github.com/NVIDIA/NeMo)
* `sudachipy` + `sudachidict-core`
* `jiwer`, `torch`, `torchaudio`
* SLURM-compatible compute cluster

---

## 📬 Contact

For questions or suggestions, please open an issue or contact the maintainers.

