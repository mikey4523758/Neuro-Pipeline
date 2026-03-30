# ============================================================
# Name:    Michael Lopez
# Date:    3/29/26
# Purpose: Extract time-domain (RMS) and frequency-domain
#          (band power via Welch PSD) features from EEG
#          epochs for use in brain-computer interface (BCI)
#          or neural signal classification pipelines.
# ============================================================

import numpy as np
from scipy.signal import welch

def extract_features(epochs, config):
    """
    Extracts PSD and Band Power features from epochs.
    Returns: X (features), y (labels)
    """
    print("--- Extracting Features ---")

    # Retrieve raw epoch data as a 3D array: (n_epochs, n_channels, n_times)
    data = epochs.get_data()

    # Get the sampling frequency from the epochs metadata
    sfreq = epochs.info['sfreq']
    
    features = []
    
    # Loop over each epoch to compute its feature vector
    for i in range(len(data)):
        epoch_data = data[i]  # Shape: (n_channels, n_times)
        epoch_feat = []
        
        # 1. Time-domain: RMS per channel
        # Root Mean Square captures the overall signal power for each channel
        rms = np.sqrt(np.mean(epoch_data**2, axis=1))
        epoch_feat.extend(rms)
        
        # 2. Frequency-domain: Band Power
        # We calculate PSD for each channel using Welch's method,
        # which reduces noise by averaging overlapping FFT segments
        freqs, psd = welch(epoch_data, sfreq, nperseg=int(sfreq))
        
        # For each frequency band defined in config (e.g., alpha, beta, theta),
        # find the relevant frequency bins and average the PSD across them
        for band_name, (fmin, fmax) in config['bands'].items():
            # Create a boolean mask to isolate frequencies within this band
            idx_band = np.logical_and(freqs >= fmin, freqs <= fmax)

            # Average PSD across the band frequencies for each channel
            band_psd = psd[:, idx_band].mean(axis=1)
            epoch_feat.extend(band_psd)
            
        # Append this epoch's full feature vector to the feature list
        features.append(epoch_feat)

    # Return features as a 2D NumPy array and the corresponding event labels
    return np.array(features), epochs.events[:, -1]