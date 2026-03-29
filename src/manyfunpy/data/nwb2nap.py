import json
import re
from pathlib import Path
from typing import Any
import numpy as np
import pynapple as nap
from natsort import natsorted
from scipy.interpolate import interp1d

# SpikeGLX NIDQ analog input channel definitions.
MIC_CHAN = 1
SPEAKER_CHAN = 2
PDIODE_CHAN = 4


def convert_nwb_to_nap(
    nwb, 
    ks_suffix: str = "_KS4_Th=8",
    min_num_spikes: int = 0,
    ) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Convert NWB data to nap objects and metadata."""
    from manyfunpy.data.audio import compute_mel_spectrogram

    # Load NWB
    if isinstance(nwb, str) or isinstance(nwb, Path):
        nwb_nap = nap.load_file(nwb)
    elif isinstance(nwb, nap.NWBFile):
        nwb_nap = nwb
    else:
        raise ValueError(f"Invalid NWB file type: {type(nwb)}")
    print(nwb_nap)

    # Load metadata
    raw_nwb = nwb_nap.nwb
    subject_meta = raw_nwb.processing["management_sheets"]["subjects"].to_dataframe().iloc[0].to_dict()
    recording_meta = raw_nwb.processing["management_sheets"]["recordings"].to_dataframe().iloc[0].to_dict()
    metadata = {
        "subject_meta": subject_meta,
        "recording_meta": recording_meta,
    }

    # Load all IntervalSets
    nap_objects = {}
    for key in nwb_nap.keys():
        if isinstance(nwb_nap[key], nap.IntervalSet):
            nap_objects[key] = nwb_nap[key]

    # Build analog objects
    anin = process_anin(nwb_nap["TimeSeriesNIDQ"], nwb_nap.get("Denoised Mic (.wav)", None))
    mel_speaker = compute_mel_spectrogram(anin["speaker"], "speaker")
    mel_mic = compute_mel_spectrogram(anin["mic"], "mic")
    nap_objects.update({
        "mel_speaker": mel_speaker,
        "mel_mic": mel_mic,
    })

    # Build speech feature objects
    if "intensity" in nwb_nap:
        intensity = nap.TsdFrame(
            t=nwb_nap["intensity"].times(),
            d=nwb_nap["intensity"].values,
            columns=["env", "peakEnv", "peakRate"],
        )
        nap_objects.update({"intensity": intensity})
    
    if "pitch" in nwb_nap:
        from manyfunpy.data.pitch import enrich_pitch
        pitch = enrich_pitch(
            nwb_nap["pitch"],
            stim_intervals=nap_objects["mfa_stim"],
            prod_intervals=nap_objects["mfa_prod"],
        )
        nap_objects.update({"pitch": pitch})

    if "artics" in nwb_nap:
        from manyfunpy.data.artic import enrich_artic
        artic = enrich_artic(nwb_nap["artics"])
        nap_objects.update({"artic": artic})

    if "artics_new" in nwb_nap:
        from manyfunpy.data.artic import build_artic2
        artic2 = build_artic2(nwb_nap["artics_new"])
        nap_objects.update({"artic2": artic2})

    # Build spike times
    spike_times = build_spike_times(nwb_nap, ks_suffix, min_num_spikes)
    num_units = len(spike_times)
    spike_times.set_info({
        "rec_id": np.repeat(raw_nwb.identifier, num_units),
        "region": np.repeat(recording_meta["Region"], num_units),
    })
    nap_objects["spike_times"] = spike_times

    return nap_objects, metadata


def process_anin(nidq: nap.TsdFrame, mic_denoised: nap.Tsd | None = None) -> nap.TsdFrame:
    """Process the analog input data."""
    from manyfunpy.data.audio import estimate_sample_rate, highpass_speech

    t_nidq = nidq.times()
    fs_nidq = estimate_sample_rate(t_nidq)

    if mic_denoised is not None:
        # Resample denoised mic to NIDQ timestamps (matches ReplaceMicAudio in SEMaker.m)
        t_mic = mic_denoised.times()
        interp_mic = interp1d(
            t_mic,
            mic_denoised.values,
            kind="linear",
            fill_value=0.0,
            bounds_error=False,
        )
        mic_signal = interp_mic(t_nidq)
    else:
        mic_signal = nidq[:, MIC_CHAN].values

    # Read the hardcoded channels
    speaker = highpass_speech(nidq[:, SPEAKER_CHAN].values, fs_nidq)
    mic = highpass_speech(mic_signal, fs_nidq)
    data = [speaker, mic]
    columns = ["speaker", "mic"]

    # Add the photodiode channel
    if nidq.values.shape[1] > PDIODE_CHAN:
        data.append(nidq.values[:, PDIODE_CHAN])
        columns.append("photodiode")

    # Build patched anin
    anin = nap.TsdFrame(t=t_nidq, d=np.column_stack(data), columns=columns)

    return anin


def build_spike_times(nwb_nap, ks_suffix: str, min_num_spikes: int = 0) -> nap.TsGroup:
    # Select one sorting group per probe.
    selected_keys = select_ks_keys(nwb_nap, ks_suffix)

    # Get probe TsGroup objects
    probes = [nwb_nap[key] for key in selected_keys]

    # Merge probes
    merged = nap.TsGroup.merge_group(*probes, reset_index=True, reset_time_support=True)

    # Drop low-spike units.
    num_spikes = np.array([len(st) for st in merged.values()], dtype=int)
    mask = num_spikes >= min_num_spikes
    merged = merged[mask]

    return merged

def select_ks_keys(nwb_nap, ks_suffix: str) -> list[str]:
    # 1) Keep only spike-sorting groups ("TsGroup") that are probe-specific.
    # Example key: "catgt_NP69_B1_g0_imec1_KS4_Th=8".
    tsgroup_keys = []
    suffix_pattern = re.escape(ks_suffix)
    for key in nwb_nap.keys():
        if isinstance(nwb_nap[key], nap.TsGroup) and re.search(fr"imec\d+{suffix_pattern}$", key):
            tsgroup_keys.append(key)

    if not tsgroup_keys:
        raise ValueError("No TsGroup probe candidates were found in the NWB file.")

    def _candidate_rank(key: str) -> tuple[int, str]:
        # Prefer most-processed data for each probe:
        #   mc_ (motion-corrected) > catgt_ > original.
        # Lexicographic key tie-break keeps the choice deterministic.
        if "mc_" in key:
            priority = 0
        elif "catgt_" in key:
            priority = 1
        else:
            priority = 2
        return (priority, key)

    # 2) Find unique probes and sort naturally: imec2 comes before imec10.
    probes = natsorted({re.search(r"imec\d+", key).group(0) for key in tsgroup_keys})
    selected = []
    for probe in probes:
        # 3) Candidate keys for this probe only.
        candidates = [key for key in tsgroup_keys if probe in key]

        # 4) Pick one key for this probe using priority rank.
        selected.append(sorted(candidates, key=_candidate_rank)[0])

    return selected


def save_nap_dataset(
    output_dir: str | Path,
    nap_objects: dict[str, Any] | None = None,
    metadata: dict[str, dict[str, Any]] | None = None,
):
    """Save nap objects to npz files and metadata to json files."""
    from manyfunpy.data.mnap import save_nap_objects
    output_dir = Path(output_dir)

    if nap_objects is not None:
        save_nap_objects(nap_objects, output_dir)
    
    if metadata is not None:
        for name, meta in metadata.items():
            with open(output_dir / f"{name}.json", "w", encoding="utf-8") as stream:
                json.dump(meta, stream, indent=2)

