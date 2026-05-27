%%%%%%%%%%%%%%%%%% TESTSET A MIXER %%%%%%%%%%%%%%%%%%%%%%
% MIXES AUDIO WITH WIND BASED ON THE ENERGY OF THE WIND (LOOKS FOR SEGMENTS
% THAT MATCH SPECIFIC THRESHOLD- THAT CONTAIN ENOUGH WIND ENERGY)

speech_dir = "C:\Users\zikan\Uni\erasmus2026\PBLproject\RECORDINGS\test_sets\wind_plus_valentini_smart\speech\";
noise_dir  = "C:\Users\zikan\Uni\erasmus2026\PBLproject\RECORDINGS\RESAMPLED\wind\NOISY_original_fs\";
output_dir_mix = "C:\Users\zikan\Uni\erasmus2026\PBLproject\RECORDINGS\test_sets\wind_plus_valentini_sequential\";
output_dir_ref = "C:\Users\zikan\Uni\erasmus2026\PBLproject\RECORDINGS\test_sets\wind_plus_valentini_sequential\speech_norm\";

if ~exist(output_dir_mix, 'dir'), mkdir(output_dir_mix); end
if ~exist(output_dir_ref, 'dir'), mkdir(output_dir_ref); end

target_fs = 16000;
desired_SNR_dB = [-5 5 10];
stride_len = round(target_fs * 2.0);

for snr = desired_SNR_dB
    folderName = sprintf('SNR_%d', snr);
    
    if ~exist(output_dir_mix+folderName, 'dir')
        mkdir(output_dir_mix+folderName);
    end
    if ~exist(output_dir_ref+folderName, 'dir')
        mkdir(output_dir_ref+folderName);
    end
end

wind_threshold = 0.1; 
wind_freq_range = [0, 200];

taper_len = round(0.02 * target_fs); 
win = hann(2 * taper_len);
fade_in = win(1:taper_len);
fade_out = win(taper_len+1:end);

speech_files = dir(fullfile(speech_dir, "*.wav"));
noise_files  = dir(fullfile(noise_dir, "*.wav"));
num_speech = length(speech_files);
num_noise  = length(noise_files);

accepted_powers = zeros(1, num_speech);

max_est_candidates = num_speech * 20;
all_candidate_powers = zeros(1, max_est_candidates);
cand_idx = 0;

all_powers = []; 
powers_index = 1;

last_loaded_noise_idx = -1;

noise_idx = 1;
start_idx = 1;

for i = 1:num_speech
    % Load Speech
    [speech, fs_s] = audioread(fullfile(speech_dir, speech_files(i).name));
    if fs_s ~= target_fs, speech = resample(speech, target_fs, fs_s); end
    if size(speech, 2) > 1, speech = mean(speech, 2); end
    L = length(speech);

    if L > 2 * taper_len
        speech(1:taper_len) = speech(1:taper_len) .* fade_in;
        speech(end-taper_len+1:end) = speech(end-taper_len+1:end) .* fade_out;
    end

    if noise_idx ~= last_loaded_noise_idx
        [noise_full, fs_n] = audioread(fullfile(noise_dir, noise_files(noise_idx).name));
      
        if fs_n ~= target_fs, noise_full = resample(noise_full, target_fs, fs_n); end
        if size(noise_full, 2) > 1, noise_full = mean(noise_full, 2); end

        last_loaded_noise_idx = noise_idx;
    end

    % Search for a segment matching the threshold
    found_valid_segment = false;
    
    while (~found_valid_segment)
        if length(noise_full) > L 
            if start_idx + L > length(noise_full)
              
                if noise_idx + 1 > num_noise
                    error("not enough noise files");
                end
                noise_idx = noise_idx + 1;
                start_idx = 1;
                [noise_full, fs_n] = audioread(fullfile(noise_dir, noise_files(noise_idx).name));
                if fs_n ~= target_fs, noise_full = resample(noise_full, target_fs, fs_n); end
                if size(noise_full, 2) > 1, noise_full = mean(noise_full, 2); end
            end
            noise_segment = noise_full(start_idx : start_idx + L - 1);
            noise_segment = noise_segment / (rms(noise_full) + eps);
    
            current_wind_energy = bandpower(noise_segment, target_fs, wind_freq_range);
            cand_idx = cand_idx + 1;
            all_powers(powers_index) = current_wind_energy;
            powers_index = powers_index + 1;
     
            if cand_idx > length(all_candidate_powers)
                all_candidate_powers = [all_candidate_powers, zeros(1, num_speech * 10)];
            end
            all_candidate_powers(cand_idx) = current_wind_energy;

                
            if current_wind_energy >= wind_threshold
                found_valid_segment = true;
                disp(found_valid_segment);
                accepted_powers(i) = current_wind_energy;       
                start_idx = start_idx + L;
            else
                start_idx = start_idx + L;
                disp(found_valid_segment);
            end
        end

    end
                   
   
    if length(noise_segment) > 2 * taper_len
        noise_segment(1:taper_len) = noise_segment(1:taper_len) .* fade_in;
        noise_segment(end-taper_len+1:end) = noise_segment(end-taper_len+1:end) .* fade_out;
    end

    sig_rms = rms(speech);
    noi_rms = rms(noise_segment);
    
    for index = 1:length(desired_SNR_dB)
        if noi_rms > 0 && sig_rms > 0
            target_noi_rms = sig_rms / (10^(desired_SNR_dB(index) / 20));
            scaling_factor = target_noi_rms / noi_rms;
            noise_scaled = noise_segment * scaling_factor;
        else
            noise_scaled = noise_segment;
        end

        % Mix, Normalize, and Save
        mixed = speech + noise_scaled;
        
        if max(abs(mixed)) > 1.0
            maximum = max(abs(mixed));
            mixed = mixed / maximum;
            clean_norm = speech / maximum;
        else
            clean_norm = speech;
        end
        
        folderName = sprintf('SNR_%d', desired_SNR_dB(index));
        out_filename = speech_files(i).name;
        
        audiowrite(fullfile(output_dir_mix+folderName, out_filename), mixed, target_fs);
        audiowrite(fullfile(output_dir_ref+folderName, out_filename), clean_norm, target_fs);

    end
    start_idx = start_idx + L + 1;
    fprintf('Processed (%d/%d): %s\n', i, num_speech, out_filename);
end

all_candidate_powers = all_candidate_powers(1:cand_idx);

figure('Color', 'w', 'Position', [100, 100, 800, 500]);

h1 = histogram(all_candidate_powers, 'BinWidth', wind_threshold/5,'FaceColor', [0.7 0.7 0.7], 'EdgeColor', 'none', 'DisplayName', 'Rejected Candidates');
hold on;
h2 = histogram(accepted_powers, 'BinWidth', wind_threshold/5,'FaceColor', [0.2 0.6 0.2], 'EdgeColor', 'none', 'DisplayName', 'Accepted for Test Set');

xline(wind_threshold, '--r', 'LineWidth', 2, 'DisplayName', 'Selection Threshold');
grid on;
xlabel('Wind Bandpower (0-200 Hz)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Count', 'FontSize', 12, 'FontWeight', 'bold');
title('Test Set Validation: Wind Intensity Distribution', 'FontSize', 14);
legend('Location', 'northeast');

set(gca, 'XScale', 'log'); 
xlim([min(all_candidate_powers(all_candidate_powers>0)) max(all_candidate_powers)*1.2]);


figure('Color', 'w', 'Position', [100, 100, 900, 600]);
% Create Histogram with Log Scale
h = histogram(all_powers, 50, 'FaceColor', [0.2 0.4 0.8], 'EdgeAlpha', 0.5);
set(gca, 'XScale', 'log'); 
grid on;
xlabel('RMS-Normalized Bandpower (0-200 Hz)', 'FontSize', 12);
ylabel('Number of Segments', 'FontSize', 12);
title('Wind Energy Distribution (Low Frequencies)', 'FontSize', 14);
text_y = max(h.Values) * 0.9;
xline(mean(all_powers), '--r', 'LineWidth', 2, 'Label', 'Average Energy');

fprintf('Min Power:  %.6f\n', min(all_powers));
fprintf('Max Power:  %.6f\n', max(all_powers));
fprintf('Mean Power: %.6f\n', mean(all_powers));
fprintf('Median:     %.6f\n', median(all_powers));