%%%%%%%% SNR_HISTOGRAM.m %%%%%%%%%%%%%
%% ESTIMATES SNR LEVEL BASED ON A SEGMENT WITHOUT WIND GUST (REFERENCE SPEECH/NOISE) 
% USED FOR SNR ESTIMATION OF FILES IN TESTSET B 

cleanFolder = "C:/Users/zikan/Uni/erasmus2026/PBLproject/RECORDINGS/roof-rec-4th-testset/50cm/dynamic/segmented/clean/";
noisyFolder = "C:/Users/zikan/Uni/erasmus2026/PBLproject/RECORDINGS/roof-rec-4th-testset/50cm/dynamic/segmented/noisy/";

cleanFolder2 = "C:/Users/zikan/Uni/erasmus2026/PBLproject/RECORDINGS/roof-rec-4th-testset/20cm/dynamic/segmented/clean/";
noisyFolder2 = "C:/Users/zikan/Uni/erasmus2026/PBLproject/RECORDINGS/roof-rec-4th-testset/20cm/dynamic/segmented/noisy/"

output_path = cleanFolder;

referenceSpeechNoisyPath = "C:/Users/zikan/Uni/erasmus2026/PBLproject/RECORDINGS/roof-rec-4th-testset/50cm/dynamic/segmented/noisy/00124_noisy.wav";
referenceSpeechCleanPath = "C:/Users/zikan/Uni/erasmus2026/PBLproject/RECORDINGS/roof-rec-4th-testset/50cm/dynamic/segmented/clean/00124_clean.wav";

referenceSpeechNoisyPath2 = "C:/Users/zikan/Uni/erasmus2026/PBLproject/RECORDINGS/roof-rec-4th-testset/20cm/dynamic/segmented/noisy/00155_noisy.wav";
referenceSpeechCleanPath2 = "C:/Users/zikan/Uni/erasmus2026/PBLproject/RECORDINGS/roof-rec-4th-testset/20cm/dynamic/segmented/clean/00155_clean.wav";

function gain_factor = get_gain_factor(pathNoisy, pathClean)
    [referenceSpeechNoisy, fs] = audioread(pathNoisy);
    [referenceSpeechClean, fs] = audioread(pathClean);
    
    noisyRMS = rms(referenceSpeechNoisy);
    cleanRMS = rms(referenceSpeechClean);
    gain_factor = noisyRMS / cleanRMS;
end

gain_factor_1 = get_gain_factor(referenceSpeechNoisyPath, referenceSpeechCleanPath);
gain_factor_2 = get_gain_factor(referenceSpeechNoisyPath2, referenceSpeechCleanPath2);

[cleanFiles_1, noisyFiles_1] = getFileLists(cleanFolder, noisyFolder);
[cleanFiles_2, noisyFiles_2] = getFileLists(cleanFolder2, noisyFolder2);


function [cleanFiles, noisyFiles] = getFileLists(cleanFolder, noisyFolder)

    cleanFiles = dir(fullfile(cleanFolder, '*.wav'));
    noisyFiles = dir(fullfile(noisyFolder, '*.wav'));

    if length(cleanFiles) ~= length(noisyFiles)
        error('Number of clean and noisy files must match.');
    end
end


snrValues_1 = computeSNR(cleanFiles_1, cleanFolder, noisyFiles_1, noisyFolder, gain_factor_1);
snrValues_2 = computeSNR(cleanFiles_2, cleanFolder2, noisyFiles_2, noisyFolder2, gain_factor_2);

snrValues = [snrValues_1; snrValues_2];

function snrValues=computeSNR(cleanFiles, cleanFolder, noisyFiles, noisyFolder, gain_factor)
% Compute SNR for each file
    snrValues = zeros(length(cleanFiles),1);
    for k = 1:length(cleanFiles)

        cleanPath = fullfile(cleanFolder, cleanFiles(k).name);
        noisyPath = fullfile(noisyFolder, noisyFiles(k).name);
    
        [clean, fs1] = audioread(cleanPath);
        clean = clean * gain_factor;
        [noisy, fs2] = audioread(noisyPath);
   
        if fs1 ~= fs2
            error('Sampling rates do not match.');
        end

        minLen = min(length(clean), length(noisy));
        clean = clean(1:minLen);
        noisy = noisy(1:minLen);
    
        % Estimate noise
        noise = noisy - clean;
    
        % Power computation
        Ps = mean(clean.^2);
        Pn = mean(noise.^2);
    
        % SNR in dB
        snrValues(k) = 10 * log10(Ps / Pn);    
    end
end

% Plot histogram
figure;
histogram(snrValues, 'BinWidth', 2, 'FaceColor', [0.2 0.4 0.8], 'EdgeColor', 'black');
ylim([0, 600]);
xlabel('SNR (dB)', 'FontSize', 12);
ylabel('Number of Segments', 'FontSize', 12);
title('Distribution of SNR Levels', 'FontSize', 14);
grid on;
set(gca, 'FontSize', 11);
hold on;
xline(mean(snrValues), '--r', ...
    sprintf('Mean = %.2f dB', mean(snrValues)), ...
    'LineWidth', 2);

% Print statistics
fprintf('\n===== Dataset SNR Statistics =====\n');
fprintf('Number of files : %d\n', length(snrValues));
fprintf('Mean SNR        : %.2f dB\n', mean(snrValues));
fprintf('Median SNR      : %.2f dB\n', median(snrValues));
fprintf('Minimum SNR     : %.2f dB\n', min(snrValues));
fprintf('Maximum SNR     : %.2f dB\n', max(snrValues));
fprintf('Std deviation   : %.2f dB\n', std(snrValues));

% Save figure
saveas(gcf, output_path+'snr_histogram.png');
print(gcf, 'snr_histogram.pdf', '-dpdf', '-bestfit');