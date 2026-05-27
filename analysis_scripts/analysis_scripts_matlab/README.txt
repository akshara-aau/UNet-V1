========================================================================
1) EVALUATION_OF_INDIVIDUAL_FILES.M
========================================================================

DESCRIPTION:
This script was used to analyze individual noisy, enhanced, and clean files.

FUNCTIONS:

  * loadFiles
    - Loads the noisy, enhanced, and (optionally) clean files.
  * setLength
    - Ensures that all loaded files have the exact same length.
  * resampleIfNeeded
    - Handles resampling of the audio files if sample rates differ.

  * plot_spectrogram
    - Generates a spectrogram plot for time-frequency analysis.
  * plot_spectrum
    - Generates an FFT (Fast Fourier Transform) spectrum plot.
  * show_waveforms
    - Generates a time-domain plot; used for time-shift analysis.
  * phase_stft
    - Generates a time-frequency plot of phase differences between files; 
      used for phase analysis.
  * estimate_time_varying_delay
    - Plots cross-correlation; used for investigating the time-variance 
      of a delay between two signals.

  * analysisWithReference
    - Performs the complete analysis pipeline when the clean reference file 
      is AVAILABLE.
  * analysisWithoutReference
    - Performs a partial analysis pipeline when the clean reference file 
      is NOT available.

========================================================================
2) AVERAGE_SPECTRUM_PLOT.M
========================================================================
DESCRIPTION: PLOTS AVERAGE FFT MAGNITUDE SPECTRUM OF .WAV FILES SAVED IN AUDIO_FOLDER

========================================================================
3) AVERAGE_ATTENUATION.M
========================================================================
DESCRIPTION: COMPUTES RATIO ENHANCED/CLEAN (IN FFT DOMAIN) FOR EVERY .WAV FILE IN THE CLEAN AND ENHANCED FOLDER

========================================================================
4) AVERAGE_PHASE_ERROR.M
========================================================================
DESCRIPTION: COMPUTES AVERAGE PHASE DIFFERENCE BETWEEN NOISY VS. CLEAN AND ENHANCED VS. CLEAN FILES

========================================================================
5) SNR_HISTOGRAM.M
========================================================================
DESCRIPTION: USED FOR SNR ESTIMATION OF FILES IN TESTSET B

========================================================================
6) DURATION_HISTOGRAM.M
========================================================================
USED FOR ANALYSIS OF TESTSET B

========================================================================
7) TESTSETA_MIXER_WITH_THRESHOLD.M
========================================================================
USED FOR ARTIFICIAL MIXING OF TESTSET A