# ============================================================
# Name:    Michael Lopez
# Date:    3/29/26
# Purpose: Generate and save summary visualizations for the
#          EEG pipeline, including a before/after PSD comparison
#          to verify filter effectiveness, and a bar chart
#          comparing SVM vs Random Forest classification accuracy
#          on the motor imagery task.
# ============================================================

import matplotlib.pyplot as plt
import os

def plot_pipeline_results(raw_pre, raw_post, metrics, output_dir):
    """
    Generates summary visualizations.
    """
    # 1. PSD Comparison
    # Side-by-side power spectral density plots let us visually confirm
    # that the bandpass and notch filters removed unwanted frequency content
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    raw_pre.plot_psd(ax=ax[0], show=False)   # PSD of the original unfiltered signal
    ax[0].set_title("Raw PSD")
    raw_post.plot_psd(ax=ax[1], show=False)  # PSD after bandpass and notch filtering
    ax[1].set_title("Filtered PSD")

    # tight_layout prevents subplot labels and titles from overlapping
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "psd_comparison.png"))
    
    # 2. Model Accuracy Comparison
    # Bar chart comparing accuracy scores across classifiers so performance
    # differences between SVM and Random Forest are immediately readable
    plt.figure(figsize=(8, 6))
    plt.bar(metrics.keys(), metrics.values(), color=['skyblue', 'salmon'])
    plt.ylab