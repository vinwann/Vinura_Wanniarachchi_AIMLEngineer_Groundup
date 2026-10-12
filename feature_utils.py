import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import signal
from IPython.display import display


# Unit / domain conversions
def convert_ms2_to_mms2(data: np.ndarray) -> np.ndarray:
    return data * 1000.0


def to_g(data: np.ndarray) -> np.ndarray:
    return data / 9.80665


def integrate_data(data: np.ndarray, sampling_rate: float) -> np.ndarray:
    return np.cumsum(data) / sampling_rate


# Filters
def bandpass_filter(
    data: np.ndarray,
    low_cutoff: int,
    high_cutoff: int,
    sampling_rate: int,
) -> np.ndarray:
    sos = signal.butter(
        6,
        [low_cutoff, high_cutoff],
        btype="bandpass",
        fs=sampling_rate,
        output="sos",
    )
    return np.asarray(signal.sosfilt(sos, data))


def highpass_filter(
    data: np.ndarray,
    low_cutoff: int,
    sampling_rate: int,
) -> np.ndarray:
    sos = signal.butter(
        6, low_cutoff, btype="highpass", fs=sampling_rate, output="sos"
    )
    return np.asarray(signal.sosfilt(sos, data))


# helpers
def remove_settling_time(
    data: np.ndarray,
    low_cutoff: int,
    original_length: int,
    sampling_rate: int,
) -> np.ndarray:
    fmax = sampling_rate / 2.56
    duration = (original_length / 2) / fmax
    multiplier = 2 if duration >= 2 else 1
    start = int(
        multiplier / low_cutoff * sampling_rate,
    )
    start = min(start, len(data) // 10)
    return data[start:]


def get_rms(data: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(data))))


def remove_dc_offset(data: np.ndarray) -> np.ndarray:
    return data - np.mean(data)


def preprocess_acceleration(
    data: np.ndarray,
    sampling_rate: int,
) -> np.ndarray:
    length = len(data)
    low_cutoff = 10
    x = to_g(data)
    x = highpass_filter(x, low_cutoff, sampling_rate)
    x = signal.detrend(x, type="linear")
    x = remove_settling_time(x, low_cutoff, length, sampling_rate)
    x = remove_dc_offset(x)
    return x


def preprocess_velocity(
    data: np.ndarray,
    sampling_rate: int,
) -> np.ndarray:
    length = len(data)
    low_cutoff = 10
    high_cutoff = 1000
    x = convert_ms2_to_mms2(data)
    x = integrate_data(x, sampling_rate)
    x = bandpass_filter(x, low_cutoff, high_cutoff, sampling_rate)
    x = signal.detrend(x, type="linear")
    x = remove_settling_time(x, low_cutoff, length, sampling_rate)
    x = remove_dc_offset(x)
    return x


def get_fft(
    data: np.ndarray,
    sampling_rate: int,
):
    N = len(data)
    line_of_resolution = N / 2.56
    maximum_frequency = sampling_rate / 2.56
    resolution = maximum_frequency / line_of_resolution
    nperseg = int(sampling_rate / resolution)
    nperseg = min(nperseg, len(data))
    noverlap = int(0.5 * nperseg)
    frequencies, psd = signal.welch(
        data,
        fs=sampling_rate,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        scaling="density",
        detrend=False,
    )
    mask = frequencies <= maximum_frequency
    frequencies = frequencies[mask]
    psd = psd[mask]
    df = frequencies[1] - frequencies[0]
    rms = np.sqrt(psd * df)
    return frequencies, rms


def get_features(
    data: np.ndarray,
    sampling_rate: int,
) -> dict:
    features = {}
    preprocessed_acceleration = preprocess_acceleration(data, sampling_rate)
    features["acceleration_rms"] = get_rms(preprocessed_acceleration)
    features["acceleration_fft"] = get_fft(preprocessed_acceleration, sampling_rate)
    preprocessed_velocity = preprocess_velocity(data, sampling_rate)
    features["velocity_rms"] = get_rms(preprocessed_velocity)
    features["velocity_fft"] = get_fft(preprocessed_velocity, sampling_rate)
    return features


def plot_rms_violin(dfs, column, labels=None):
    if labels is None:
        labels = [f"DataFrame {i + 1}" for i in range(len(dfs))]

    if len(labels) != len(dfs):
        raise ValueError("Provide one label per DataFrame.")

    fig, ax = plt.subplots(figsize=(10, 5))
    rng = np.random.default_rng(42)
    tick_labels = []

    for position, (df, label) in enumerate(zip(dfs, labels), start=1):
        values = pd.to_numeric(df[column], errors="coerce").to_numpy(
            dtype=float, na_value=np.nan
        )
        values = values[np.isfinite(values)]
        tick_labels.append(f"{label}\n(n={values.size:,})")

        if values.size == 0:
            ax.text(
                position, 0.5, "No data",
                transform=ax.get_xaxis_transform(), ha="center",
            )
            continue

        if np.unique(values).size > 1:
            violin = ax.violinplot(
                values,
                positions=[position],
                widths=0.7,
                showmedians=True,
                showextrema=False,
            )
            violin["cmedians"].set_color("black")
        else:
            ax.hlines(
                values[0], position - 0.15, position + 0.15,
                color="black",
            )

        jitter = rng.uniform(-0.08, 0.08, values.size)
        ax.scatter(position + jitter, values, s=8, alpha=0.15)

    ax.set_xticks(range(1, len(dfs) + 1), tick_labels)
    ax.set_ylabel("RMS (units unconfirmed)")
    ax.set_title(column)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()



def find_active_periods(df, cutoff, rms_col, max_gap="2h"):
    gap_limit = pd.Timedelta(max_gap)

    if gap_limit <= pd.Timedelta(0):
        raise ValueError("max_gap must be positive.")
    if not np.isfinite(cutoff):
        raise ValueError("cutoff must be finite.")

    data = df[["date", rms_col]].copy()
    data["date"] = pd.to_datetime(data["date"], utc=True, errors="coerce")
    data["rms"] = pd.to_numeric(data[rms_col], errors="coerce")

    valid = (
        data["date"].notna()
        & data["rms"].notna()
        & np.isfinite(data["rms"])
    )
    data = data.loc[valid].sort_values("date").reset_index(drop=True)

    if data["date"].duplicated().any():
        raise ValueError("Resolve duplicate timestamps first.")

    data["active"] = data["rms"] >= cutoff

    new_period = data["active"] & (
        ~data["active"].shift(fill_value=False)
        | (data["date"].diff() > gap_limit)
    )

    data["run_id"] = (
        new_period.cumsum().where(data["active"]).astype("Int64")
    )

    periods = (
        data.loc[data["active"]]
        .groupby("run_id")
        .agg(
            first_active=("date", "min"),
            last_active=("date", "max"),
            readings=("date", "size"),
            median_rms=("rms", "median"),
        )
    )

    periods["observed_span_hours"] = (
        (periods["last_active"] - periods["first_active"])
        .dt.total_seconds() / 3600
    )

    fig, ax = plt.subplots(figsize=(15, 4))

    for active, label, colour in [
        (False, "Low vibration", "grey"),
        (True, "Likely active", "tab:orange"),
    ]:
        subset = data.loc[data["active"] == active]
        ax.scatter(
            subset["date"], subset["rms"],
            s=10, alpha=0.6, color=colour, label=label,
        )

    ax.axhline(
        cutoff, color="red", linestyle="--",
        label=f"Cutoff = {cutoff:.3f}",
    )

    ax.set_title(f"Likely active periods\n{rms_col}")
    ax.set_xlabel("Date (UTC)")
    ax.set_ylabel("RMS (units unconfirmed)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    with pd.option_context(
        "display.max_rows", None,
        "display.max_columns", None,
    ):
        display(periods.round(3))

    return data, periods


def compare_waveforms(
    df1, index1, df2, index2,
    labels=("Dataset 1", "Dataset 2"),
):
    for df, index in [(df1, index1), (df2, index2)]:
        if index not in df.index:
            raise KeyError(f"Index label {index} is not in the DataFrame.")
        if not df.index.is_unique:
            raise ValueError("DataFrame indexes must be unique.")

    row1 = df1.loc[index1]
    row2 = df2.loc[index2]

    wave_cols = list(dict.fromkeys(
        col for df in (df1, df2) for col in df.columns
        if ("Wave" in col and "RMS" not in col)
        or col == "Motor DE_Audio Signal"
    ))

    for col in wave_cols:
        fig, axes = plt.subplots(
            1, 2, figsize=(16, 3), sharex=True, sharey=True
        )

        for ax, row, index, label in zip(
            axes, [row1, row2], [index1, index2], labels
        ):
            value = row.get(col)

            if value is None or np.isscalar(value):
                ax.text(
                    0.5, 0.5, "No waveform",
                    transform=ax.transAxes, ha="center",
                )
            else:
                waveform = np.asarray(value, dtype=float).ravel().copy()
                waveform[~np.isfinite(waveform)] = np.nan

                if waveform.size:
                    ax.plot(
                        np.arange(waveform.size),
                        waveform,
                        linewidth=0.6,
                    )
                else:
                    ax.text(
                        0.5, 0.5, "Empty waveform",
                        transform=ax.transAxes, ha="center",
                    )

            ax.set_title(
                f"{label} | Index: {index}\n{row.get('date', '')}"
            )
            ax.set_xlabel("Sample index")
            ax.grid(True, alpha=0.3)

        axes[0].set_ylabel("Raw value (units unconfirmed)")
        fig.suptitle(col)
        plt.tight_layout()
        plt.show()

def calculate_raw_waveform_rms(value):
    if value is None or np.isscalar(value):
        return np.nan

    samples = np.asarray(value, dtype=float).ravel()
    if samples.size == 0 or not np.isfinite(samples).all():
        return np.nan

    return float(np.sqrt(np.mean(np.square(samples))))