# ============================================================
# Name:    Michael Lopez
# Date:    3/29/26
# Purpose: Load and preprocess EEG data from the PhysioNet
#          Motor Movement/Imagery dataset. Handles data
#          fetching, channel standardization, montage setup,
#          and epoch extraction for left/right hand motor
#          imagery classification tasks.
# ============================================================

import mne
from mne.datasets import eegbci

def fetch_physionet_data(subject, runs):
    """
    Downloads and loads the PhysioNet EEG Motor Movement dataset.
    """
    print(f"--- Fetching Data for Subject {subject}, Runs {runs} ---")

    # Download the EDF files for the given subject and run numbers
    # (MNE caches files locally after the first download)
    files = eegbci.load_data(subject, runs)

    # Read each EDF file into a Raw object with data preloaded into memory
    raws = [mne.io.read_raw_edf(f, preload=True) for f in files]

    # Concatenate all run segments into a single continuous Raw object
    raw = mne.concatenate_raws(raws)
    
    # Standardize channel names to follow the 10-05 naming convention
    # so they match the standard montage used below
    eegbci.standardize(raw)

    # Load the standard 10-05 electrode position montage and apply it,
    # giving each channel a known physical location on the scalp
    montage = mne.channels.make_standard_montage('standard_1005')
    raw.set_montage(montage)
    
    return raw

def get_epochs(raw, tmin=-1.0, tmax=4.0):
    """
    Extracts epochs based on events. 
    T1 = left hand, T2 = right hand.
    """
    # Parse annotations embedded in the Raw object into a standard events array
    events, event_id = mne.events_from_annotations(raw)

    # Select only EEG channels, excluding stimulus/trigger channels
    picks = mne.pick_types(raw.info, eeg=True, stim=False)
    
    # Map events to meaningful labels
    # 2 = Left Hand, 3 = Right Hand
    event_dict = {'left_hand': 2, 'right_hand': 3}
    
    # Slice the continuous signal into epochs time-locked to each event.
    # tmin/tmax define the window around each event marker (in seconds).
    # baseline=None skips baseline correction since it's handled downstream.
    # proj=True applies any stored SSP projectors for artifact suppression.
    epochs = mne.Epochs(raw, events, event_dict, tmin, tmax, proj=True, 
                        picks=picks, baseline=None, preload=True)
    return epochs