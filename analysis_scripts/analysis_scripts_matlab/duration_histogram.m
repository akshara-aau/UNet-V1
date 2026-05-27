%%%%%%%%%% DURATION_HISTOGRAM.M %%%%%%%%%%%
%% PLOTS HISTOGRAM OF DURATIONS OF .WAV FILES IN AUDIO FOLDER
% USED FOR ANALYSIS OF TESTSET A

audioFolder = 'C:\Users\zikan\Uni\erasmus2026\PBLproject\RECORDINGS\test_sets\wind_plus_valentini_sequential\speech_norm\SNR_-5';
files = dir(fullfile(audioFolder, '*.wav'));
durations = zeros(length(files), 1);

for k = 1:length(files)
    filePath = fullfile(audioFolder, files(k).name);
    info = audioinfo(filePath);
    durations(k) = info.Duration;
end

% Plot histogram
figure;

histogram(durations,'BinWidth', 1, 'FaceColor', [0.2 0.4 0.8],'EdgeColor', 'black');

xlabel('Duration (seconds)', 'FontSize', 12);
ylabel('Number of Segments', 'FontSize', 12);
title('Distribution of Segment Durations', 'FontSize', 14);
grid on;
set(gca, 'FontSize', 11);

% Print summary statistics
fprintf('Number of files: %d\n', length(durations));
fprintf('Total duration: %.2f minutes\n', sum(durations)/60);
fprintf('Mean duration: %.2f seconds\n', mean(durations));
fprintf('Min duration: %.2f seconds\n', min(durations));
fprintf('Max duration: %.2f seconds\n', max(durations));