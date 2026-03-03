#!/usr/bin/env python
"""
从原始试验文件准备 radarODE-MTL 训练数据集。

输出格式（每个试验文件夹）：
    sst_seg_<i>.npy     -> 形状 (50, 71, 120), float32
    ecg_seg_<i>.npy     -> 形状 (可变,), float32
    anchor_seg_<i>.npy  -> 形状 (200,), float32

支持的输入：
1) RCG 模式（推荐用于 MMECG 风格数据）：
   - 包含预处理雷达通道（RCG）和 ECG 的试验文件
2) ADC 模式（实验性）：
   - 包含原始/近原始 ADC 立方体和 ECG 的试验文件
   - 执行参考提取流程以获得 50 个雷达通道

示例：
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
        description="将原始雷达 + ECG 试验转换为 radarODE-MTL 训练片段。"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="输入试验文件或包含试验文件的目录 (.mat/.npz/.npy)。",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="输出数据集根目录。",
    )
    parser.add_argument(
        "--radar-source",
        type=str,
        choices=["rcg", "adc"],
        default="rcg",
        help="输入文件中的雷达源类型。",
    )
    parser.add_argument(
        "--glob",
        type=str,
        default="*.mat",
        help="当 --input 为目录时的 glob 模式（默认: *.mat）。",
    )
    parser.add_argument(
        "--ecg-key",
        type=str,
        default="ECG",
        help="输入文件中 ECG 的键名。",
    )
    parser.add_argument(
        "--rcg-key",
        type=str,
        default="RCG",
        help="rcg 模式下预处理雷达通道的键名。",
    )
    parser.add_argument(
        "--adc-key",
        type=str,
        default="radar_adc",
        help="adc 模式下 ADC 立方体的键名。",
    )
    parser.add_argument(
        "--subject-id-key",
        type=str,
        default="id",
        help="受试者 ID 的可选键；用于输出文件夹命名。",
    )
    parser.add_argument(
        "--status-key",
        type=str,
        default="physistatus",
        help="生理状态的可选键；用于输出文件夹命名。",
    )
    parser.add_argument(
        "--signal-fs",
        type=float,
        default=200.0,
        help="ECG 和 RCG 信号的采样率（Hz）（默认 200）。",
    )
    parser.add_argument(
        "--sst-time-fs",
        type=float,
        default=30.0,
        help="SST 目标时间轴采样率（Hz）（默认 30）。",
    )
    parser.add_argument(
        "--sst-window-sec",
        type=float,
        default=4.0,
        help="SST 片段窗口长度（秒）（默认 4）。",
    )
    parser.add_argument(
        "--sst-freq-min",
        type=float,
        default=1.0,
        help="最小 SST 频率（Hz）（默认 1）。",
    )
    parser.add_argument(
        "--sst-freq-max",
        type=float,
        default=25.0,
        help="最大 SST 频率（Hz）（默认 25）。",
    )
    parser.add_argument(
        "--sst-freq-bins",
        type=int,
        default=71,
        help="输出 SST 频率 bins 数量（默认 71）。",
    )
    parser.add_argument(
        "--expected-radar-channels",
        type=int,
        default=50,
        help="模型使用的预期雷达通道数（默认 50）。",
    )
    parser.add_argument(
        "--ecg-output-len",
        type=int,
        default=200,
        help="锚点输出长度（默认 200）。",
    )
    parser.add_argument(
        "--anchor-sigma",
        type=float,
        default=2.5,
        help="锚点高斯 sigma（样本）（默认 2.5）。",
    )
    parser.add_argument(
        "--min-cycle-len",
        type=int,
        default=80,
        help="最小 ECG 周期长度（样本）（默认 80）。",
    )
    parser.add_argument(
        "--max-cycle-len",
        type=int,
        default=260,
        help="最大 ECG 周期长度（样本）（默认 260）。",
    )
    parser.add_argument(
        "--save-manifest",
        action="store_true",
        help="为每个处理的试验保存清单 JSON。",
    )
    parser.add_argument(
        "--adc-num-tx",
        type=int,
        default=3,
        help="ADC 模式：TX 天线数量（默认 3）。",
    )
    parser.add_argument(
        "--adc-slowtime-fs",
        type=float,
        default=200.0,
        help="ADC 模式：用于 RCG 提取的慢时间采样率。",
    )
    parser.add_argument(
        "--adc-heartband-min",
        type=float,
        default=0.8,
        help="ADC 模式：通道/范围选择的心脏频带下限（Hz）。",
    )
    parser.add_argument(
        "--adc-heartband-max",
        type=float,
        default=3.0,
        help="ADC 模式：通道/范围选择的心脏频带上限（Hz）。",
    )
    return parser.parse_args()


def _require_neurokit2():
    try:
        import neurokit2 as nk  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "需要 neurokit2。请先安装依赖（pip install -r requirements.txt）。"
        ) from exc
    return nk


def _require_ssqueezepy():
    try:
        from ssqueezepy import ssq_cwt  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "需要 ssqueezepy。请先安装依赖（pip install -r requirements.txt）。"
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
        raise ValueError(f"{path} 是 .npy 但不是序列化的字典；请使用 .mat/.npz 或类字典的 .npy。")
    if suffix == ".mat":
        try:
            return loadmat(path, squeeze_me=True, struct_as_record=False)
        except NotImplementedError:
            if mat73 is None:
                raise RuntimeError(
                    f"{path} 似乎是 MATLAB v7.3 格式。请安装 'mat73' 来读取它。"
                )
            return mat73.loadmat(path)
    raise ValueError(f"不支持的文件扩展名: {suffix}")


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
    返回:
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
        raise KeyError(f"在加载的试验中找不到 ECG 键 '{ecg_key}'。")
    if radar is None:
        missing = rcg_key if radar_source == "rcg" else adc_key
        raise KeyError(f"在加载的试验中找不到雷达键 '{missing}'。")

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
    参考 ADC -> RCG 提取。

    预期的 ADC 形状: [frames, chirps, rx, adc_samples]
    (实部/虚部打包或复数)。
    """
    if adc.ndim != 4:
        raise ValueError(
            f"ADC 输入必须是 4D [frames, chirps, rx, adc_samples], 得到形状 {adc.shape}"
        )
    adc = _as_complex(adc)

    frames, chirps, rx, adc_samples = adc.shape
    usable_chirps = (chirps // num_tx) * num_tx
    if usable_chirps == 0:
        raise ValueError(f"chirp 数量 {chirps} 对于 num_tx={num_tx} 无效。")

    adc = adc[:, :usable_chirps, :, :]
    chirps_per_tx = usable_chirps // num_tx
    adc = adc.reshape(frames, chirps_per_tx, num_tx, rx, adc_samples)

    # 在快时间轴上进行距离 FFT。
    rng = np.fft.fft(adc, axis=-1)
    rng = rng[:, :, :, :, : adc_samples // 2]

    # 展平慢时间和虚拟通道。
    # 慢时间: frames * chirps_per_tx, 虚拟通道: num_tx * rx。
    slow = rng.reshape(frames * chirps_per_tx, num_tx * rx, adc_samples // 2)

    # 去除静态杂波。
    slow = slow - slow.mean(axis=0, keepdims=True)

    # 转换为慢时间上的展开相位。
    phase = np.unwrap(np.angle(slow), axis=0)
    phase = signal.detrend(phase, axis=0, type="linear")

    # 按心脏频带能量对 (虚拟通道, 距离 bin) 候选进行排序。
    n_t, n_v, n_r = phase.shape
    flattened = phase.reshape(n_t, n_v * n_r)
    freqs, psd = signal.welch(flattened, fs=slowtime_fs, axis=0, nperseg=min(512, n_t))
    mask = (freqs >= band_min) & (freqs <= band_max)
    score = psd[mask].sum(axis=0)
    if np.all(score == 0):
        score = psd.sum(axis=0)

    top_idx = np.argsort(score)[-expected_channels:]
    rcg = flattened[:, top_idx]

    # 确保形状为 [time, expected_channels]。
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
        raise ValueError(f"RCG 必须是 2D [time, channels], 得到形状 {rcg.shape}")

    # 如果通道出现在第一轴，尝试修复方向。
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
                f"通道 {c} 在 [{freq_min}, {freq_max}] 中没有 SST 频率。"
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
        # 回退方案
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
    # 保留 "_<id>_" 标记以兼容现有的数据加载器。
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
        raise RuntimeError(f"在 {trial_path.name} 中检测到的 R 峰不足: {rpeaks.size}")

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
            # 回退：快速支持混合扩展名
            files = sorted(
                list(input_path.glob("*.mat"))
                + list(input_path.glob("*.npz"))
                + list(input_path.glob("*.npy"))
            )
        return files
    raise FileNotFoundError(f"找不到输入路径: {input_path}")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    files = iter_input_files(input_path, args.glob)
    if not files:
        raise RuntimeError(f"在 {input_path} 下使用模式 {args.glob} 没有匹配的输入文件")

    summaries: List[Dict[str, Any]] = []
    for fp in files:
        try:
            summary = process_trial(fp, args)
            summaries.append(summary)
            print(
                f"[OK] {fp.name}: 片段={summary['written_segments']}, "
                f"丢弃周期={summary['dropped_cycle']}, "
                f"丢弃边界={summary['dropped_boundary']}"
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[失败] {fp.name}: {exc}")

    total = sum(x["written_segments"] for x in summaries)
    print(f"\n处理了 {len(summaries)}/{len(files)} 个试验, 总片段数: {total}")

    global_manifest = output_path / "prepare_manifest.json"
    with open(global_manifest, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)
    print(f"保存摘要: {global_manifest}")


if __name__ == "__main__":
    main()
