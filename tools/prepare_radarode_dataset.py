#!/usr/bin/env python
"""
Prepare radarODE-MTL training dataset from raw trial files.

Output format (per trial folder):
    sst_seg_<i>.npy     -> shape (50, 71, 120), float32
    ecg_seg_<i>.npy     -> shape (variable,), float32
    anchor_seg_<i>.npy  -> shape (200,), float32

Supported inputs:
1) RCG mode (recommended for MMECG-style data):
   - trial files containing preprocessed radar channels (RCG) and ECG
2) ADC mode (experimental):
   - trial files containing raw/near-raw ADC cube and ECG
   - performs a reference extraction pipeline to get 50 radar channels

Examples:
    python tools/prepare_radarode_dataset.py \
        --input ./raw_trials \
        --output ./Dataset \
        --radar-source rcg

    python tools/prepare_radarode_dataset.py \
        --input ./adc_trials \
        --output ./Dataset \
        --radar-source adc \
        --adc-key radar_adc
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy import signal
from scipy.interpolate import interp1d
from scipy.io import loadmat

try:
    import mat73
except ImportError:
    mat73 = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert raw radar + ECG trials into radarODE-MTL training segments."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input trial file or directory with trial files (.mat/.npz/.npy).",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output dataset root directory.",
    )
    parser.add_argument(
        "--radar-source",
        type=str,
        choices=["rcg", "adc"],
        default="rcg",
        help="Radar source type in input files.",
    )
    parser.add_argument(
        "--glob",
        type=str,
        default="*.mat",
        help="Glob pattern when --input is a directory (default: *.mat).",
    )
    parser.add_argument(
        "--ecg-key",
        type=str,
        default="ECG",
        help="Key name for ECG in input file.",
    )
    parser.add_argument(
        "--rcg-key",
        type=str,
        default="RCG",
        help="Key name for preprocessed radar channels in rcg mode.",
    )
    parser.add_argument(
        "--adc-key",
        type=str,
        default="radar_adc",
        help="Key name for ADC cube in adc mode.",
    )
    parser.add_argument(
        "--subject-id-key",
        type=str,
        default="id",
        help="Optional key for subject ID; used in output folder naming.",
    )
    parser.add_argument(
        "--status-key",
        type=str,
        default="physistatus",
        help="Optional key for physiological status; used in output folder naming.",
    )
    parser.add_argument(
        "--signal-fs",
        type=float,
        default=200.0,
        help="Sampling rate (Hz) of ECG and RCG signals (default 200).",
    )
    parser.add_argument(
        "--sst-time-fs",
        type=float,
        default=30.0,
        help="Target time-axis sampling rate (Hz) for SST (default 30).",
    )
    parser.add_argument(
        "--sst-window-sec",
        type=float,
        default=4.0,
        help="SST segment window length in seconds (default 4).",
    )
    parser.add_argument(
        "--sst-freq-min",
        type=float,
        default=1.0,
        help="Minimum SST frequency in Hz (default 1).",
    )
    parser.add_argument(
        "--sst-freq-max",
        type=float,
        default=25.0,
        help="Maximum SST frequency in Hz (default 25).",
    )
    parser.add_argument(
        "--sst-freq-bins",
        type=int,
        default=71,
        help="Number of output SST frequency bins (default 71).",
    )
    parser.add_argument(
        "--expected-radar-channels",
        type=int,
        default=50,
        help="Expected number of radar channels used by model (default 50).",
    )
    parser.add_argument(
        "--ecg-output-len",
        type=int,
        default=200,
        help="Anchor output length (default 200).",
    )
    parser.add_argument(
        "--anchor-sigma",
        type=float,
        default=2.5,
        help="Anchor Gaussian sigma in samples (default 2.5).",
    )
    parser.add_argument(
        "--min-cycle-len",
        type=int,
        default=80,
        help="Minimum ECG cycle length in samples (default 80).",
    )
    parser.add_argument(
        "--max-cycle-len",
        type=int,
        default=260,
        help="Maximum ECG cycle length in samples (default 260).",
    )
    parser.add_argument(
        "--save-manifest",
        action="store_true",
        help="Save a manifest JSON for each processed trial.",
    )
    parser.add_argument(
        "--adc-num-tx",
        type=int,
        default=3,
        help="ADC mode: number of TX antennas (default 3).",
    )
    parser.add_argument(
        "--adc-slowtime-fs",
        type=float,
        default=200.0,
        help="ADC mode: resulting slow-time sampling rate used for RCG extraction.",
    )
    parser.add_argument(
        "--adc-heartband-min",
        type=float,
        default=0.8,
        help="ADC mode: lower bound (Hz) of cardiac band for channel/range selection.",
    )
    parser.add_argument(
        "--adc-heartband-max",
        type=float,
        default=3.0,
        help="ADC mode: upper bound (Hz) of cardiac band for channel/range selection.",
    )
    return parser.parse_args()


def _require_neurokit2():
    try:
        import neurokit2 as nk  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "neurokit2 is required. Install dependencies first (pip install -r requirements.txt)."
        ) from exc
    return nk


def _require_ssqueezepy():
    try:
        from ssqueezepy import ssq_cwt  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "ssqueezepy is required. Install dependencies first (pip install -r requirements.txt)."
        ) from exc
    return ssq_cwt


def _safe_to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return np.asarray(x)
    return np.asarray(x)


def _normalize_string_value(x: Any) -> str:
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="ignore")
    if isinstance(x, np.ndarray):
        if x.size == 1:
            return _normalize_string_value(x.item())
        return "".join(str(v) for v in x.tolist())
    return str(x)


def load_trial_file(path: Path) -> Dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".npz":
        loaded = np.load(path, allow_pickle=True)
        return {k: loaded[k] for k in loaded.files}
    if suffix == ".npy":
        arr = np.load(path, allow_pickle=True)
        if isinstance(arr, np.ndarray) and arr.shape == () and isinstance(arr.item(), dict):
            return dict(arr.item())
        raise ValueError(f"{path} is .npy but not a serialized dict; use .mat/.npz or dict-like .npy.")
    if suffix == ".mat":
        try:
            return loadmat(path, squeeze_me=True, struct_as_record=False)
        except NotImplementedError:
            if mat73 is None:
                raise RuntimeError(
                    f"{path} appears to be MATLAB v7.3. Install 'mat73' to read it."
                )
            return mat73.loadmat(path)
    raise ValueError(f"Unsupported file extension: {suffix}")


def _extract_from_struct(data: Any, key: str) -> Any:
    if hasattr(data, key):
        return getattr(data, key)
    if isinstance(data, dict) and key in data:
        return data[key]
    return None


def resolve_trial_fields(
    loaded: Dict[str, Any],
    *,
    ecg_key: str,
    rcg_key: str,
    adc_key: str,
    subject_id_key: str,
    status_key: str,
    radar_source: str,
) -> Tuple[np.ndarray, np.ndarray, str, str]:
    """
    Returns:
        radar_signal, ecg_signal, subject_id, status
    """
    root = loaded
    if "data" in loaded:
        root = loaded["data"]

    ecg = _extract_from_struct(root, ecg_key)
    subject_id = _extract_from_struct(root, subject_id_key)
    status = _extract_from_struct(root, status_key)

    if radar_source == "rcg":
        radar = _extract_from_struct(root, rcg_key)
    else:
        radar = _extract_from_struct(root, adc_key)

    if ecg is None:
        ecg = loaded.get(ecg_key)
    if radar is None:
        radar = loaded.get(rcg_key if radar_source == "rcg" else adc_key)
    if subject_id is None:
        subject_id = loaded.get(subject_id_key, "unknown")
    if status is None:
        status = loaded.get(status_key, "UNK")

    if ecg is None:
        raise KeyError(f"Cannot find ECG key '{ecg_key}' in loaded trial.")
    if radar is None:
        missing = rcg_key if radar_source == "rcg" else adc_key
        raise KeyError(f"Cannot find radar key '{missing}' in loaded trial.")

    ecg_arr = _safe_to_numpy(ecg).astype(np.float32).squeeze()
    radar_arr = _safe_to_numpy(radar)

    if ecg_arr.ndim != 1:
        ecg_arr = ecg_arr.reshape(-1)

    sid = _normalize_string_value(subject_id)
    st = _normalize_string_value(status).upper()
    return radar_arr, ecg_arr, sid, st


def _as_complex(x: np.ndarray) -> np.ndarray:
    if np.iscomplexobj(x):
        return x
    if x.shape[-1] == 2:
        return x[..., 0] + 1j * x[..., 1]
    return x.astype(np.complex64)


def extract_rcg_from_adc(
    adc: np.ndarray,
    *,
    num_tx: int,
    expected_channels: int,
    slowtime_fs: float,
    band_min: float,
    band_max: float,
) -> np.ndarray:
    """
    Reference ADC -> RCG extraction.

    Expected ADC shape: [frames, chirps, rx, adc_samples]
    (real/imag packed or complex).
    """
    if adc.ndim != 4:
        raise ValueError(
            f"ADC input must be 4D [frames, chirps, rx, adc_samples], got shape {adc.shape}"
        )
    adc = _as_complex(adc)

    frames, chirps, rx, adc_samples = adc.shape
    usable_chirps = (chirps // num_tx) * num_tx
    if usable_chirps == 0:
        raise ValueError(f"Invalid chirp count {chirps} for num_tx={num_tx}.")

    adc = adc[:, :usable_chirps, :, :]
    chirps_per_tx = usable_chirps // num_tx
    adc = adc.reshape(frames, chirps_per_tx, num_tx, rx, adc_samples)

    # Range FFT on fast-time axis.
    rng = np.fft.fft(adc, axis=-1)
    rng = rng[:, :, :, :, : adc_samples // 2]

    # Flatten slow-time and virtual channels.
    # slow-time: frames * chirps_per_tx, virtual channels: num_tx * rx.
    slow = rng.reshape(frames * chirps_per_tx, num_tx * rx, adc_samples // 2)

    # Remove static clutter.
    slow = slow - slow.mean(axis=0, keepdims=True)

    # Convert to unwrapped phase over slow-time.
    phase = np.unwrap(np.angle(slow), axis=0)
    phase = signal.detrend(phase, axis=0, type="linear")

    # Rank (virtual-channel, range-bin) candidates by cardiac-band energy.
    n_t, n_v, n_r = phase.shape
    flattened = phase.reshape(n_t, n_v * n_r)
    freqs, psd = signal.welch(flattened, fs=slowtime_fs, axis=0, nperseg=min(512, n_t))
    mask = (freqs >= band_min) & (freqs <= band_max)
    score = psd[mask].sum(axis=0)
    if np.all(score == 0):
        score = psd.sum(axis=0)

    top_idx = np.argsort(score)[-expected_channels:]
    rcg = flattened[:, top_idx]

    # Ensure shape [time, expected_channels].
    if rcg.shape[1] < expected_channels:
        padded = np.zeros((rcg.shape[0], expected_channels), dtype=np.float32)
        padded[:, : rcg.shape[1]] = rcg
        rcg = padded
    elif rcg.shape[1] > expected_channels:
        rcg = rcg[:, :expected_channels]
    return rcg.astype(np.float32)


def ensure_rcg_shape(rcg: np.ndarray, expected_channels: int) -> np.ndarray:
    rcg = np.asarray(rcg).astype(np.float32)
    if rcg.ndim != 2:
        raise ValueError(f"RCG must be 2D [time, channels], got shape {rcg.shape}")

    # Try to fix orientation if channels appear on first axis.
    if rcg.shape[1] != expected_channels and rcg.shape[0] == expected_channels:
        rcg = rcg.T

    if rcg.shape[1] < expected_channels:
        pad = np.zeros((rcg.shape[0], expected_channels - rcg.shape[1]), dtype=np.float32)
        rcg = np.concatenate([rcg, pad], axis=1)
    elif rcg.shape[1] > expected_channels:
        rcg = rcg[:, :expected_channels]
    return rcg


def align_radar_to_ecg_length(rcg: np.ndarray, ecg_len: int) -> np.ndarray:
    if rcg.shape[0] == ecg_len:
        return rcg
    out = np.zeros((ecg_len, rcg.shape[1]), dtype=np.float32)
    x_old = np.linspace(0.0, 1.0, rcg.shape[0], endpoint=True)
    x_new = np.linspace(0.0, 1.0, ecg_len, endpoint=True)
    for c in range(rcg.shape[1]):
        out[:, c] = np.interp(x_new, x_old, rcg[:, c])
    return out


def _interp_freq_axis(
    sst: np.ndarray, freq: np.ndarray, freq_min: float, freq_max: float, out_bins: int
) -> np.ndarray:
    tgt = np.linspace(freq_min, freq_max, out_bins, endpoint=True)
    fn = interp1d(freq, sst, kind="linear", axis=0, bounds_error=False, fill_value="extrapolate")
    return fn(tgt)


def compute_full_sst(
    rcg: np.ndarray,
    *,
    signal_fs: float,
    target_time_fs: float,
    freq_min: float,
    freq_max: float,
    freq_bins: int,
    voices_per_octave: int = 10,
) -> np.ndarray:
    ssq_cwt = _require_ssqueezepy()
    n_t, n_c = rcg.shape
    target_t = int(round(n_t * target_time_fs / signal_fs))
    out = np.zeros((n_c, freq_bins, target_t), dtype=np.float32)

    for c in range(n_c):
        tx, _, ssq_freqs, *_ = ssq_cwt(
            rcg[:, c], fs=signal_fs, wavelet="morlet", nv=voices_per_octave
        )
        sst = np.abs(np.asarray(tx))
        freq = np.asarray(ssq_freqs)
        mask = (freq >= freq_min) & (freq <= freq_max)
        if not np.any(mask):
            raise RuntimeError(
                f"No SST frequencies in [{freq_min}, {freq_max}] for channel {c}."
            )
        sst = sst[mask]
        freq = freq[mask]
        if sst.shape[0] != freq_bins:
            sst = _interp_freq_axis(sst, freq, freq_min, freq_max, freq_bins)
        if sst.shape[1] != target_t:
            sst = signal.resample(sst, target_t, axis=1)
        s_min = float(np.min(sst))
        s_max = float(np.max(sst))
        if s_max - s_min > 1e-8:
            sst = (sst - s_min) / (s_max - s_min)
        else:
            sst = np.zeros_like(sst)
        out[c] = sst.astype(np.float32)
    return out


def detect_r_peaks(ecg: np.ndarray, fs: float) -> np.ndarray:
    nk = _require_neurokit2()
    cleaned = nk.ecg_clean(ecg, sampling_rate=fs, method="neurokit")
    _, info = nk.ecg_peaks(cleaned, sampling_rate=fs, method="neurokit")
    peaks = np.asarray(info.get("ECG_R_Peaks", []), dtype=np.int64)
    if peaks.size < 3:
        # fallback
        peaks, _ = signal.find_peaks(cleaned, distance=max(1, int(0.3 * fs)))
        peaks = np.asarray(peaks, dtype=np.int64)
    return peaks


def make_anchor(length: int, peak_idx: int, sigma: float) -> np.ndarray:
    x = np.arange(length, dtype=np.float32)
    anchor = np.exp(-0.5 * ((x - peak_idx) / sigma) ** 2)
    m = float(anchor.max())
    if m > 0:
        anchor /= m
    return anchor.astype(np.float32)


def _sanitize_name(x: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in x)
    return safe.strip("_") or "UNK"


def build_trial_folder_name(subject_id: str, status: str, trial_stem: str) -> str:
    sid = _sanitize_name(subject_id)
    st = _sanitize_name(status.upper())
    ts = _sanitize_name(trial_stem)
    # Keep "_<id>_" token for compatibility with existing dataloaders.
    return f"obj{sid}_{st}_{ts}_{sid}_"


def process_trial(
    trial_path: Path,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    loaded = load_trial_file(trial_path)
    radar_raw, ecg, subject_id, status = resolve_trial_fields(
        loaded,
        ecg_key=args.ecg_key,
        rcg_key=args.rcg_key,
        adc_key=args.adc_key,
        subject_id_key=args.subject_id_key,
        status_key=args.status_key,
        radar_source=args.radar_source,
    )

    if args.radar_source == "adc":
        rcg = extract_rcg_from_adc(
            radar_raw,
            num_tx=args.adc_num_tx,
            expected_channels=args.expected_radar_channels,
            slowtime_fs=args.adc_slowtime_fs,
            band_min=args.adc_heartband_min,
            band_max=args.adc_heartband_max,
        )
    else:
        rcg = ensure_rcg_shape(radar_raw, args.expected_radar_channels)

    rcg = align_radar_to_ecg_length(rcg, len(ecg))
    rpeaks = detect_r_peaks(ecg, args.signal_fs)
    if rpeaks.size < 3:
        raise RuntimeError(f"Insufficient R peaks detected in {trial_path.name}: {rpeaks.size}")

    full_sst = compute_full_sst(
        rcg,
        signal_fs=args.signal_fs,
        target_time_fs=args.sst_time_fs,
        freq_min=args.sst_freq_min,
        freq_max=args.sst_freq_max,
        freq_bins=args.sst_freq_bins,
    )

    out_trial_dir = Path(args.output) / build_trial_folder_name(
        subject_id=subject_id,
        status=status,
        trial_stem=trial_path.stem,
    )
    out_trial_dir.mkdir(parents=True, exist_ok=True)

    seg_len_t = int(round(args.sst_window_sec * args.sst_time_fs))
    half = seg_len_t // 2
    written = 0
    dropped_boundary = 0
    dropped_cycle = 0

    for i in range(1, len(rpeaks) - 1):
        left = (int(rpeaks[i - 1]) + int(rpeaks[i])) // 2
        right = (int(rpeaks[i]) + int(rpeaks[i + 1])) // 2
        if right <= left:
            dropped_cycle += 1
            continue
        ecg_cycle = ecg[left:right]
        if ecg_cycle.size < args.min_cycle_len or ecg_cycle.size > args.max_cycle_len:
            dropped_cycle += 1
            continue

        center_t = int(round(int(rpeaks[i]) * args.sst_time_fs / args.signal_fs))
        start_t = center_t - half
        end_t = start_t + seg_len_t
        if start_t < 0 or end_t > full_sst.shape[-1]:
            dropped_boundary += 1
            continue

        sst_seg = full_sst[:, :, start_t:end_t].astype(np.float32)
        if sst_seg.shape != (
            args.expected_radar_channels,
            args.sst_freq_bins,
            seg_len_t,
        ):
            dropped_boundary += 1
            continue

        r_rel = int(rpeaks[i]) - left
        r_rel = max(0, min(r_rel, ecg_cycle.size - 1))
        peak_idx = int(round((r_rel / max(1, ecg_cycle.size - 1)) * (args.ecg_output_len - 1)))
        anchor = make_anchor(args.ecg_output_len, peak_idx, args.anchor_sigma)

        np.save(out_trial_dir / f"sst_seg_{written}.npy", sst_seg)
        np.save(out_trial_dir / f"ecg_seg_{written}.npy", ecg_cycle.astype(np.float32))
        np.save(out_trial_dir / f"anchor_seg_{written}.npy", anchor)
        written += 1

    manifest = {
        "trial_file": str(trial_path),
        "output_dir": str(out_trial_dir),
        "subject_id": subject_id,
        "status": status,
        "radar_source": args.radar_source,
        "signal_fs": args.signal_fs,
        "sst_time_fs": args.sst_time_fs,
        "sst_window_sec": args.sst_window_sec,
        "sst_freq_range_hz": [args.sst_freq_min, args.sst_freq_max],
        "sst_freq_bins": args.sst_freq_bins,
        "ecg_output_len": args.ecg_output_len,
        "total_rpeaks": int(len(rpeaks)),
        "written_segments": int(written),
        "dropped_boundary": int(dropped_boundary),
        "dropped_cycle": int(dropped_cycle),
    }
    if args.save_manifest:
        with open(out_trial_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
    return manifest


def iter_input_files(input_path: Path, glob_pattern: str) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        files = sorted(input_path.glob(glob_pattern))
        if not files:
            # fallback: support mixed extensions quickly
            files = sorted(
                list(input_path.glob("*.mat"))
                + list(input_path.glob("*.npz"))
                + list(input_path.glob("*.npy"))
            )
        return files
    raise FileNotFoundError(f"Input path not found: {input_path}")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    files = iter_input_files(input_path, args.glob)
    if not files:
        raise RuntimeError(f"No input files matched under {input_path} with pattern {args.glob}")

    summaries: List[Dict[str, Any]] = []
    for fp in files:
        try:
            summary = process_trial(fp, args)
            summaries.append(summary)
            print(
                f"[OK] {fp.name}: segments={summary['written_segments']}, "
                f"dropped_cycle={summary['dropped_cycle']}, "
                f"dropped_boundary={summary['dropped_boundary']}"
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[FAIL] {fp.name}: {exc}")

    total = sum(x["written_segments"] for x in summaries)
    print(f"\nProcessed {len(summaries)}/{len(files)} trials, total segments: {total}")

    global_manifest = output_path / "prepare_manifest.json"
    with open(global_manifest, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)
    print(f"Saved summary: {global_manifest}")


if __name__ == "__main__":
    main()
