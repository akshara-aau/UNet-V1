%%%%%%%%%%%%%%%%%%%%%% AVERAGE_ATTENUATION.M %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% COMPUTES RATIO ENHANCED/CLEAN FOR EVERY .WAV FILE IN THE CLEAN AND ENHANCED FOLDER
% IT COMPUTES AVERAGE OF THESE RATIOS AND TRANSFORMS TO dB SCALE 
% THIS IS THEN PLOTTED IN FREQUENCY DOMAIN
% THIS SCIPT IS USED TO SEE HOW IS ENHANCED VERSION SIMILAR TO CLEAN
% VERSION (TO INVESTIGATE SPEECH PRESERVATION AFTER ENHANCEMENT)

function [mean_ratio, F, fs] = compute_average_attenuation(clean_dir, enh_dir)

    clean_files = dir(fullfile(clean_dir, '*.wav'));
    enh_files = dir(fullfile(enh_dir, '*.wav'));

    if isempty(clean_files)
        error('No WAV files found in clean directory.');
    end

    if length(clean_files) ~= length(enh_files)
        warning('Different number of files in directories.');
    end

    % STFT parameters
    win = hann(512);
    hop = 256;
    nfft = 2048;

    accumulated_ratio = [];
    num_processed = 0;

    for k = 1:length(clean_files)

        clean_name = clean_files(k).name;
        enh_name = enh_files(k).name;

        clean_path = fullfile(clean_dir, clean_name);
        enh_path = fullfile(enh_dir, enh_name);

        % Check if enhanced file exists
        if ~exist(enh_path, 'file')
            fprintf('Missing enhanced file: %s\n', enh_name);
            continue;
        end

        % Read files
        [clean, fs1] = audioread(clean_path);
        [enh, fs2] = audioread(enh_path);

        max_cl = max(abs(clean));
        max_enh = max(abs(enh));

        clean = clean/max_cl;
        enh = enh/max_enh;

        % Check sampling rate
        if fs1 ~= fs2
            fprintf('Skipping %s (sampling rate mismatch)\n', clean_name);
            continue;
        end

        fs = fs1;

        % Match lengths
        N = min(length(clean), length(enh));
        clean = clean(1:N);
        enh = enh(1:N);

        [S_clean, F, ~] = stft(clean, fs, 'Window', win, 'OverlapLength', length(win)-hop, 'FFTLength', nfft);

        [S_enh, ~, ~] = stft(enh, fs, 'Window', win, 'OverlapLength', length(win)-hop, 'FFTLength', nfft);

        % Magnitude spectra
        mag_clean = abs(S_clean);
        mag_enh = abs(S_enh);

        % Average over time
        avg_clean = mean(mag_clean, 2);
        avg_enh = mean(mag_enh, 2);

        % Ratio
        ratio = avg_enh ./ (avg_clean + eps);

        % Accumulate
        if isempty(accumulated_ratio)
            accumulated_ratio = ratio;
        else
            accumulated_ratio = accumulated_ratio + ratio;
        end

        num_processed = num_processed + 1;
        fprintf('Processed: %s\n', clean_name);
    end

    if num_processed == 0
        error('No valid file pairs processed.');
    end
    % Dataset average
    mean_ratio = accumulated_ratio / (num_processed);
end


model = "no-overlap";
base = "C:\Users\zikan\Uni\erasmus2026\PBLproject\RECORDINGS\roof-rec-4th-testset\";
subdirectories = {base + "50cm\shotgun\segmented\"};%, base + "20cm\shotgun\segmented\", base + "50cm\dynamic\segmented\", base + "50cm\shotgun\segmented\"};
comb_number = length(subdirectories);

accumulated_ratio = 0;


for index = 1:(comb_number)
    clean_dir = subdirectories(index) + "clean\";
    enh_dir = subdirectories(index) + "enhanced\"+model+"\";

    [mean_ratio, F, fs] = compute_average_attenuation(clean_dir, enh_dir);
    accumulated_ratio = accumulated_ratio + mean_ratio;
end

mean_ratio_all = accumulated_ratio / comb_number;
attenuation_db_all = 20 * log10(mean_ratio_all);

% Plot
figure;
semilogx(F, attenuation_db_all, 'LineWidth', 1);
xlabel('Frequency [Hz]', 'FontSize', 14);
ylabel('Average Attenuation [dB]', 'FontSize', 14);
title("Dataset Average Spectral Attenuation", 'FontSize', 16);
grid on;
grid minor;
xlim([20 fs/2]);
ylim([-40 40]);
xticks([20 50 100 200 500 1000 2000 4000 8000 16000]);
set(gca, 'FontSize', 14);
set(gca, 'LineWidth', 1);

saveas(gcf, clean_dir+"average_attenuation_"+model+".png");

    
