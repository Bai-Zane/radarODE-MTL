# radarODE-MTL Reproduction Guide

This guide gives a practical, command-line workflow to reproduce dataset preparation and model training in this repo.

It supports:

1. `rcg` mode (recommended, paper-aligned): preprocessed radar channels (`RCG`) + raw ECG
2. `adc` mode (experimental): raw radar ADC cube + raw ECG

The generated training files match repository expectations:

- `sst_seg_*.npy` with shape `(50, 71, 120)`
- `ecg_seg_*.npy` with variable cycle length
- `anchor_seg_*.npy` with shape `(200,)`

## 1) Environment Setup

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 2) Prepare Raw Input Files

Each trial file should contain ECG plus radar data.

### Option A: RCG Mode (Recommended)
This matches MMECG-style data where radar has already been converted to 50-channel cardiac signals.

Expected keys:

- `ECG`: 1D waveform
- `RCG`: 2D array `[time, channels]`
- optional: `id`, `physistatus`

### Option B: ADC Mode (Experimental)
Expected keys:

- `ECG`: 1D waveform
- `radar_adc`: 4D array `[frames, chirps, rx, adc_samples]` (complex or real+imag packed)

Note: ADC mode uses a reference extraction pipeline (range FFT + phase extraction + cardiac-band ranking). It is intended for reproducible processing, but exact behavior can differ from the original MMECG internal preprocessing.

## 3) Generate radarODE-MTL Segments

### 3.1 RCG Mode

```bash
python tools/prepare_radarode_dataset.py \
  --input ./raw_trials \
  --output ./Dataset \
  --radar-source rcg \
  --glob "*.mat" \
  --signal-fs 200 \
  --sst-time-fs 30 \
  --sst-window-sec 4 \
  --sst-freq-min 1 \
  --sst-freq-max 25 \
  --sst-freq-bins 71 \
  --expected-radar-channels 50 \
  --save-manifest
```

### 3.2 ADC Mode

```bash
python tools/prepare_radarode_dataset.py \
  --input ./adc_trials \
  --output ./Dataset \
  --radar-source adc \
  --adc-key radar_adc \
  --signal-fs 200 \
  --adc-slowtime-fs 200 \
  --adc-num-tx 3 \
  --expected-radar-channels 50 \
  --save-manifest
```

Output folder example:

```text
Dataset/
  obj1_NB_trial001/
    sst_seg_0.npy
    ecg_seg_0.npy
    anchor_seg_0.npy
    ...
```

## 4) Train radarODE-MTL

```bash
python Projects/radarODE_plus/main.py \
  --mode train \
  --dataset_path ./Dataset \
  --save_path ./Model_saved \
  --save_name EGA_default \
  --gpu_id 0 \
  --batch_size 22 \
  --n_epochs 200 \
  --weighting EGA \
  --arch HPS \
  --optim sgd \
  --lr 0.005 \
  --weight_decay 0.0005 \
  --momentum 0.937 \
  --scheduler cos \
  --eta_min 0.00005 \
  --T_max 100 \
  --aug_snr 100
```

## 5) Test with a Saved Model

```bash
python Projects/radarODE_plus/main.py \
  --mode test \
  --dataset_path ./Dataset \
  --save_path ./Model_saved \
  --load_path ./Model_saved/best_EGA_default.pt \
  --save_name EGA_default \
  --gpu_id 0 \
  --batch_size 22 \
  --aug_snr 100
```

## 6) Default Design Choices Used in This Pipeline

These defaults follow paper/repo behavior:

- ECG peak detection: NeuroKit2 (`ECG_R_Peaks`)
- single-cycle ECG target: variable-length cycle, later resampled to 200 in dataloader
- SST settings: Morlet synchrosqueezed CWT, frequency range `[1, 25] Hz`
- SST segment length: 4 seconds
- SST time downsampling: 30 Hz (`120` time bins for 4 seconds)
- radar channels: 50

## 7) Troubleshooting

- `Cannot find ECG/RCG key`: pass explicit `--ecg-key`, `--rcg-key`, or `--adc-key`.
- MATLAB v7.3 loading issue: ensure `mat73` is installed.
- Very few generated segments: check ECG quality and `--signal-fs` value.
- GPU OOM: reduce `--batch_size` or `--num_workers`.
- `TEST: nan ...` with tiny datasets: reduce `--batch_size` (dataloader uses `drop_last=True`).
