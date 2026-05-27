%%%%%%%%%%%%%%%%ř AVERAGE_SPECTRUM_PLOT.M %%%%%%%%%%%%%%%%řř
%% PLOTS AVERAGE FFT MAGNITUDE SPECTRUM OF .WAV FILES SAVED IN AUDIO_FOLDER


% Folder containing audio files
audioFolder = "";

% Get all .wav files
files = dir(fullfile(audioFolder, '*.wav'));

% Parameters
NFFT = 2048;

% Initialize
avgSpectrum = zeros(NFFT/2+1,1);

% Process files
for k = 1:length(files)
    % Read audio
    filePath = fullfile(audioFolder, files(k).name);
    [x, fs] = audioread(filePath);

    % peak normalization
    x = x ./ (max(abs(x)) + eps);

    % Spectrogram
    [S, F, T] = spectrogram(x, hann(512), 256, NFFT, fs);

    % Mean magnitude spectrum
    mag = mean(abs(S), 2);

    % Accumulate
    avgSpectrum = avgSpectrum + mag;

end

% Average over dataset
avgSpectrum = avgSpectrum / length(files);
avgSpectrum_dB = 20 * log10(avgSpectrum + eps);

% Plot
figure;
semilogx(F, avgSpectrum_dB, 'LineWidth', 1);
xlabel('Frequency (Hz)', 'FontSize', 14);
ylabel('Magnitude (dB)', 'FontSize', 14);
title('Average Log-Frequency Spectrum', 'FontSize', 14);
grid on;
xlim([20 8000]);
ylim([-40,40]);

xticks([20 50 100 200 500 1000 2000 4000 8000]);
set(gca, 'FontSize', 14);
set(gcf, 'Color', 'w');

% Save
saveas(gcf, audioFolder+"\average_spectrum.png");
