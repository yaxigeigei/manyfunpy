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
DEFAULT_SURFACE_LOCATION = 7660.0


def convert_nwb_to_nap(
    nwb, 
    ks_suffix: str = "_KS4_Th=8",
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
    spike_times = build_spike_times(
        nwb_nap,
        ks_suffix,
        surface_location=recording_meta["Surface location"],
    )
    spike_times.set_info({
        "region": np.repeat(recording_meta["Region"], len(spike_times)),
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


def build_spike_times(nwb_nap, ks_suffix: str, surface_location=None) -> nap.TsGroup:
    
    # Select one sorting group per probe
    selected_keys = select_ks_keys(nwb_nap, ks_suffix)

    # Extract info
    rec_id = nwb_nap.nwb.identifier
    surface_location = parse_surface_location(surface_location)
    
    # Loop through probes
    probes = []
    for sort_name in selected_keys:
        probe = nwb_nap[sort_name]
        probe_index = int(re.search(r"imec(\d+)", sort_name).group(1))

        probe_meta = probe.metadata.copy().drop(columns=["rate"])
        probe_meta["rec_id"] = np.repeat(rec_id, len(probe_meta))
        probe_meta["probe_index"] = np.repeat(probe_index, len(probe_meta))
        probe_meta["sort_name"] = np.repeat(sort_name, len(probe_meta))
        
        # Derive unique cluster IDs
        cluster_ids = probe_meta["unit_name"].to_numpy(dtype=int)
        unique_cluster_ids = convert_to_unique_cluster_ids(
            cluster_ids=cluster_ids,
            rec_id=rec_id, 
            probe_index=probe_index,
            )
        probe_meta.index = unique_cluster_ids
        
        # Convert cortical depth
        probe_meta = convert_cortical_depth(probe_meta, surface_location, probe_index)
        
        # Rebuild the probe with unique cluster IDs
        probe_group = {}
        for old_key, new_key in zip(probe.keys(), unique_cluster_ids):
            probe_group[int(new_key)] = probe[old_key]
        probe = nap.TsGroup(probe_group, time_support=probe.time_support, metadata=probe_meta)

        probes.append(probe)
    
    # Merge probes
    merged = nap.TsGroup.merge_group(*probes, reset_index=False, reset_time_support=True)
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

def convert_to_unique_cluster_ids(cluster_ids: np.ndarray, rec_id: str, probe_index: int) -> np.ndarray:
    """Convert cluster IDs to unique cluster IDs."""
    rec_base_num = get_rec_base_num(rec_id)
    probe_base_num = probe_index * 1e4
    unique_cluster_ids = rec_base_num + probe_base_num + cluster_ids
    return unique_cluster_ids

def get_rec_base_num(rec_id: str) -> int:
    """Get the recording-specific cluster-ID base (e.g. 'NP1_B1' -> 10100000)."""
    match = re.fullmatch(r"NP(\d+)_B(\d+)", rec_id)
    subject_number = int(match.group(1))
    block_number = int(match.group(2))
    return int(subject_number * 1e7 + block_number * 1e5)


def convert_cortical_depth(unit_meta, surface_location, probe_index: int):
    """Convert distance-to-tip coordinates to cortical depth."""
    if "depth" not in unit_meta.columns:
        print("Skipping cortical depth conversion because spike_times.metadata has no 'depth' column.")
        return unit_meta
    if surface_location is None:
        print("Skipping cortical depth conversion because no surface location was provided.")
        return unit_meta

    surface_location = np.atleast_1d(np.asarray(surface_location, dtype=float))
    if surface_location.size == 1:
        surface_value = surface_location[0]
    else:
        surface_value = surface_location[probe_index]

    distance_to_tip = unit_meta["depth"].to_numpy(dtype=float, copy=True)
    unit_meta["distance_to_tip"] = distance_to_tip
    unit_meta["cortical_depth"] = -(distance_to_tip - surface_value)

    return unit_meta

def parse_surface_location(surface_location) -> np.ndarray:
    """Normalize surface locations to a numeric probe array."""
    # Use the default when the input is missing
    if surface_location is None:
        return np.array([DEFAULT_SURFACE_LOCATION], dtype=float)

    # Parse string-encoded values
    if isinstance(surface_location, str):
        try:
            surface_location = eval(surface_location, {"__builtins__": {}}, {"NaN": np.nan, "nan": np.nan})
        except Exception:
            print(f"Cannot parse surface location from '{surface_location}'. Using default value of {DEFAULT_SURFACE_LOCATION}.")
            surface_location = DEFAULT_SURFACE_LOCATION
    surface_location = np.atleast_1d(np.asarray(surface_location, dtype=float))

    # Fill missing entries or fall back entirely
    is_missing = np.isnan(surface_location)
    if np.any(is_missing):
        print(f"Filling missing surface locations with default value of {DEFAULT_SURFACE_LOCATION}.")
        surface_location[is_missing] = DEFAULT_SURFACE_LOCATION

    return surface_location


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

