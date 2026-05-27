%%%%%%%%%%%%%%%%% AVERAGE_PHASE_ERROR.m %%%%%%%%%%%%ř
%% COMPUTES AVERAGE PHASE DIFFERENCE BETWEEN NOISY VS. CLEAN AND ENHANCED VS. CLEAN FILES
% USED for PHASE ANALYSIS 

function[] = phase_error_avg(cleanDir, noisyDir, enhDir, fs, log)

    cleanFiles = dir(fullfile(cleanDir, '*.wav'));
    noisyFiles = dir(fullfile(noisyDir, '*.wav'));
    enhFiles   = dir(fullfile(enhDir, '*.wav'));
    
    [~, idxClean] = sort({cleanFiles.name}); cleanFiles = cleanFiles(idxClean);
    [~, idxNoisy] = sort({noisyFiles.name}); noisyFiles = noisyFiles(idxNoisy);
    [~, idxEnh]   = sort({enhFiles.name});   enhFiles   = enhFiles(idxEnh);
    
    numFiles = length(cleanFiles);
    if numFiles == 0
        error('No .wav files found in the clean directory.');
    elseif length(noisyFiles) ~= numFiles || length(enhFiles) ~= numFiles
        error('The number of files in the directories do not match.');
    end
    
    % STFT Parameters
    win = hann(512, 'periodic');
    noverlap = 512 / 2;
    nfft = 2048;
    
    running_error_noisy = [];
    running_error_enh   = [];
  
    for i = 1:numFiles
        cleanPath = fullfile(cleanDir, cleanFiles(i).name);
        noisyPath = fullfile(noisyDir, noisyFiles(i).name);
        enhPath   = fullfile(enhDir, enhFiles(i).name);
        
        clean = audioread(cleanPath);
        noisy = audioread(noisyPath);
        enh   = audioread(enhPath);

        [S_clean, f, ~] = stft(clean, fs, 'Window', win, 'OverlapLength', noverlap, 'FFTLength', nfft, 'FrequencyRange', 'onesided');
        [S_noisy, ~, ~] = stft(noisy, fs, 'Window', win, 'OverlapLength', noverlap, 'FFTLength', nfft, 'FrequencyRange', 'onesided');
        [S_enh, ~, ~]   = stft(enh,   fs, 'Window', win, 'OverlapLength', noverlap, 'FFTLength', nfft, 'FrequencyRange', 'onesided');

        phase_err_noisy = abs(angle(exp(1i * (angle(S_clean) - angle(S_noisy)))));
        phase_err_enh   = abs(angle(exp(1i * (angle(S_clean) - angle(S_enh)))));
        

        if i == 1
            running_error_noisy = zeros(size(phase_err_noisy));
            running_error_enh   = zeros(size(phase_err_enh));
        end
        
        running_error_noisy = running_error_noisy + phase_err_noisy;
        running_error_enh   = running_error_enh   + phase_err_enh;


    end

    avg_error_map_noisy = running_error_noisy / numFiles;
    avg_error_map_enh   = running_error_enh / numFiles;

   
    % Compute the mean across the time dimension
    avg_error_per_freq_noi = mean(avg_error_map_noisy, 2);
    avg_error_per_freq_enh = mean(avg_error_map_enh, 2);
    
    figure;
    plot(f, avg_error_per_freq_noi, 'r', 'LineWidth', 1); hold on;
    plot(f, avg_error_per_freq_enh, 'b', 'LineWidth', 1);

    grid on;
    
    set(gca, 'FontSize', 12);
    ylim([0, pi]);
    if log == true
      set(gca, 'XScale', 'log');  
    end
    title('Average phase error');
    xlabel('Frequency [Hz]');
    ylabel('Average Phase Error [rad]');
    legend('Noisy vs Clean', 'Enhanced vs Clean');
    if log == true
        saveas(gcf, enhDir+"\avg_phase_error_log.png");
    else
        saveas(gcf, "avg_phase_error_lin.png");
    end
end


cleanDir = "";
noisyDir = "";
enhDir = "";
phase_error_avg(cleanDir, noisyDir, enhDir, 16000, true);