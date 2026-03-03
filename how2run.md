Implemented an end-to-end reproducible pipeline in this repo, based on the paper/repo assumptions (50 channels, SST in 1–25 Hz, 4 s windows, 30 Hz SST time axis, NeuroKit2 R-peak extraction).

**What I added**
- Full reproduction guide: [REPRODUCE.md](F:/project/radarODE-MTL/REPRODUCE.md)
- Dependency file: [requirements.txt](F:/project/radarODE-MTL/requirements.txt)
- Raw-data preprocessing CLI (RCG mode + raw ADC mode): [tools/prepare_radarode_dataset.py](F:/project/radarODE-MTL/tools/prepare_radarode_dataset.py)
- README entry pointing to the new pipeline: [README.md](F:/project/radarODE-MTL/README.md)

**Key fixes for usability**
- Training entrypoint now CLI-driven (removed hardcoded dataset/save paths and hardcoded training config): [main.py](F:/project/radarODE-MTL/Projects/radarODE_plus/main.py)
- Passed `aug_snr` correctly into dataset construction in training.
- Made dataset ID split robust for custom datasets (`--test_ids`, auto ID discovery).
- Fixed anchor decoder output mismatch (now consistent with `anchor_seg` length 200).
- CPU fallback in trainer (no forced CUDA-only): [trainer.py](F:/project/radarODE-MTL/LibMTL/trainer.py)
- Fixed test-mode crash when no epoch buffer is provided.
- Relaxed folder-ID matching in dataloaders for generated dataset naming:
  - [spectrum_dataset.py](F:/project/radarODE-MTL/Projects/radarODE_plus/spectrum_dataset.py)
  - [dataloader.py](F:/project/radarODE-MTL/Projects/radarODE_plus/dataloader.py)
- Removed unused hard dependency imports in encoder: [encoder.py](F:/project/radarODE-MTL/Projects/radarODE_plus/nets/encoder.py)

**How to run (quick path)**
1. Install deps:
```bash
pip install -r requirements.txt
```
2. Build dataset from raw trials:
```bash
python tools/prepare_radarode_dataset.py --input <raw_dir> --output ./Dataset --radar-source rcg
```
or (experimental ADC path):
```bash
python tools/prepare_radarode_dataset.py --input <raw_dir> --output ./Dataset --radar-source adc --adc-key radar_adc
```
3. Train:
```bash
python Projects/radarODE_plus/main.py --mode train --dataset_path ./Dataset --save_path ./Model_saved
```

**Verification done**
- `py_compile` checks passed on all modified Python files.
- Preprocessing CLI tested end-to-end on synthetic trial data; generated valid:
  - `sst_seg_*.npy` `(50,71,120)`
  - `ecg_seg_*.npy` variable length
  - `anchor_seg_*.npy` `(200,)`
- Model test-mode smoke run succeeded on generated synthetic dataset.