# ============================================================
# Name:    Michael Lopez
# Date:    3/29/26
# Purpose: Apply signal preprocessing steps to raw EEG data,
#          including bandpass filtering to isolate motor imagery
#          frequency bands, notch filtering to remove power line
#          noise, and re-referencing to a common average to
#          reduce spatial bias across electrodes.
# ============================================================

import mne

def apply_preprocessing(raw, config):
    """
    Applies filtering and re-referencing to raw EEG signal.
    """
    print("--- Preprocessing Signal ---")

    # Bandpass filter to isolate motor imagery frequencies (Mu/Beta)
    # l_freq and h_freq define the lower and upper cutoff boundaries,
    # and 'firwin' uses a windowed FIR design for a clean frequency response
    raw.filter(l_freq=config['l_freq'], h_freq=config['h_freq'], fir_design='firwin')
    
    # Notch filter to suppress power line interference (typically 60 Hz in the US,
    # 50 Hz in Europe) which can otherwise dominate the frequency spectrum
    raw.notch_filter(freqs=config['notch_freq'])
    
    # Re-reference to common average so each channel reflects activity
    # relative to the mean of all electrodes, reducing global drift and bias.
    # projection=False applies the reference directly rather than storing it as an SSP projector.
    if config['reference'] == 'average':
        raw.set_eeg_reference('average', projection=False)
        
    return raw