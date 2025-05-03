import sys
import numpy as np
import librosa
from typing import Tuple, Union, Optional

def compute_energy(signal: np.ndarray) -> float:
    """Compute the energy of the audio signal."""
    return np.sum(signal ** 2) / np.float64(len(signal))

def compute_entropy_of_energy(signal: np.ndarray, num_of_short_blocks: int = 10) -> float:
    """Compute entropy of energy for the audio signal."""
    epsilon = sys.float_info.epsilon
    frame_energy = np.sum(signal ** 2)
    frame_length = len(signal)

    sub_win_len = int(np.floor(frame_length / num_of_short_blocks))
    if frame_length != sub_win_len * num_of_short_blocks:
        signal = signal[0:sub_win_len * num_of_short_blocks]

    sub_win = signal.reshape(sub_win_len, num_of_short_blocks, order="F").copy()
    norm_sub_frame_energy = np.sum(sub_win ** 2, axis=0) / (frame_energy + epsilon)
    return -np.sum(epsilon * np.log2(norm_sub_frame_energy + epsilon))

def compute_tempo(signal: np.ndarray, sample_rate: int) -> float:
    """Estimate the tempo (beats per minute) of the audio."""
    if hasattr(librosa.feature, 'rhythm'):  # librosa 0.10.0+
        tempo = librosa.feature.rhythm.tempo(y=signal, sr=sample_rate)
    else:  # legacy support
        tempo = librosa.beat.tempo(y=signal, sr=sample_rate)
    return float(tempo[0])

def compute_rms(signal: np.ndarray, frame_length: int = 2048, hop_length: int = 512) -> np.ndarray:
    """Compute root-mean-square (RMS) energy for each frame."""
    return librosa.feature.rms(y=signal, frame_length=frame_length, hop_length=hop_length)

def compute_zcr(signal: np.ndarray, frame_length: int = 2048, hop_length: int = 512) -> np.ndarray:
    """Compute zero-crossing rate of the audio signal."""
    return librosa.feature.zero_crossing_rate(y=signal, frame_length=frame_length, hop_length=hop_length)

def compute_mfcc(signal: np.ndarray, sample_rate: int, 
                 num_mfcc: int = 20, n_fft: int = 2048, 
                 hop_length: int = 512) -> np.ndarray:
    """Compute Mel-frequency cepstral coefficients (MFCCs)."""
    mfccs = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=num_mfcc,
                                 n_fft=n_fft, hop_length=hop_length)
    return np.mean(mfccs, axis=1)  # Return mean of MFCCs across frames

def compute_chroma_stft(signal: np.ndarray, sample_rate: int, 
                        n_fft: int = 2048, hop_length: int = 512) -> np.ndarray:
    """Compute chroma features from STFT."""
    return librosa.feature.chroma_stft(y=signal, sr=sample_rate, 
                                      n_fft=n_fft, hop_length=hop_length)

def compute_spectral_centroid(signal: np.ndarray, sample_rate: int, 
                             n_fft: int = 2048, hop_length: int = 512) -> np.ndarray:
    """Compute spectral centroid for each frame."""
    return librosa.feature.spectral_centroid(y=signal, sr=sample_rate, 
                                            n_fft=n_fft, hop_length=hop_length)

def compute_spectral_bandwidth(signal: np.ndarray, sample_rate: int, 
                              n_fft: int = 2048, hop_length: int = 512) -> np.ndarray:
    """Compute spectral bandwidth for each frame."""
    return librosa.feature.spectral_bandwidth(y=signal, sr=sample_rate, 
                                             n_fft=n_fft, hop_length=hop_length)

def compute_spectral_rolloff(signal: np.ndarray, sample_rate: int, 
                            n_fft: int = 2048, hop_length: int = 512, 
                            roll_percent: float = 0.85) -> np.ndarray:
    """Compute spectral rolloff for each frame."""
    return librosa.feature.spectral_rolloff(y=signal, sr=sample_rate, 
                                          n_fft=n_fft, hop_length=hop_length,
                                          roll_percent=roll_percent)

def compute_pitch(signal: np.ndarray, sample_rate: int, 
                 frame_length: int = 2048, hop_length: int = 512,
                 method: str = 'yin', fmin: float = 100.0, 
                 fmax: float = 1000.0) -> np.ndarray:
    """Compute pitch (fundamental frequency) of the audio signal."""
    if method.lower() == 'yin':
        pitch = librosa.yin(y=signal, fmin=fmin, fmax=fmax, 
                           sr=sample_rate, frame_length=frame_length,
                           hop_length=hop_length)
    elif method.lower() == 'pyin':
        pitch, _, _ = librosa.pyin(y=signal, fmin=fmin, fmax=fmax,
                                 sr=sample_rate, frame_length=frame_length,
                                 hop_length=hop_length)
    else:
        raise ValueError("Method must be either 'yin' or 'pyin'")
    return pitch

import numpy as np

def extract_all_features(signal: np.ndarray, sample_rate: int,
                        frame_length: int = 2048, hop_length: int = 512) -> dict:
    """Extract all audio features and return as a dictionary with only statistics."""
    features = {}
    
    # Time-domain features
    features['energy'] = compute_energy(signal)
    features['entropy_of_energy'] = compute_entropy_of_energy(signal)
    features['tempo'] = compute_tempo(signal, sample_rate)
    
    # Frame-based features
    features['rms'] = compute_rms(signal, frame_length, hop_length)
    features['zcr'] = compute_zcr(signal, frame_length, hop_length)
    
    # Spectral features
    features['mfcc'] = compute_mfcc(signal, sample_rate, n_fft=frame_length, hop_length=hop_length)
    features['chroma_stft'] = compute_chroma_stft(signal, sample_rate, n_fft=frame_length, hop_length=hop_length)
    features['spectral_centroid'] = compute_spectral_centroid(signal, sample_rate, n_fft=frame_length, hop_length=hop_length)
    features['spectral_bandwidth'] = compute_spectral_bandwidth(signal, sample_rate, n_fft=frame_length, hop_length=hop_length)
    features['spectral_rolloff'] = compute_spectral_rolloff(signal, sample_rate, n_fft=frame_length, hop_length=hop_length)
    
    # Pitch features
    features['pitch'] = compute_pitch(signal, sample_rate, frame_length, hop_length)
    
    # Compute statistics for frame-based features
    stat_features = {}
    for feature_name in ['rms', 'zcr', 'spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff', 'pitch']:
        if feature_name in features:
            values = features[feature_name]
            valid_values = values[~np.isnan(values)]  # Remove NaNs
            stat_features[f'{feature_name}_mean'] = np.mean(valid_values) if valid_values.size > 0 else np.nan
            stat_features[f'{feature_name}_std'] = np.std(valid_values) if valid_values.size > 0 else np.nan
            stat_features[f'{feature_name}_min'] = np.min(valid_values) if valid_values.size > 0 else np.nan
            stat_features[f'{feature_name}_max'] = np.max(valid_values) if valid_values.size > 0 else np.nan

    final_features = {**{k: v for k, v in features.items() if not isinstance(v, np.ndarray)}, **stat_features}
    
    return final_features
