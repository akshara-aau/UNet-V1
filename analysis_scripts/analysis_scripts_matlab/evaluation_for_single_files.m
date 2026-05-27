%%%%%%%%%%%%%%%%%%%%%% EVALUATION_OF_INDIVIDUAL_FILES.M %%%%%%%%%%%%%%%
%% THIS SCRIPT WAS USED TO ANALYSE INDIVIDUAL NOISY, ENHANCED AND CLEAN FILES

% FUNCTIONS:
%   loadFiles - loads the noisy, enhanced and optionaly clean file
%   setLength - ensures that all files have a same length
%   resampleIfNeeded - resampling
%
%   plot_spectrogram
%   plot_spectrum - FFT
%   show_waveforms - time-domain plot - used for time-shift analysis
%   phase_stft - time-frequnecy plot of phase differences between files - used for phase analysis
%   estimate_time_varying_delay - plots cross correlation - used for investigating time-variance of a delay between two signals
%
%   analysisWithReference - performs whole analysis if the clean file is available
%   analysisWithoutReference - performs part of the analysis if the clean file is NOT available


%%

function [noisySpeech, cleanSpeech, denoisedSpeech] = loadFiles(target_fs, audio_directory, load_clean)
% LOADFILES loads audio files
%       audio (.wav) files must be in the same folder and named: noisy.wav, clean.wav, enhanced.wav
%       target_fs = target sampling rate (audio files will be resampled to
%                   this sampling rate
%       audio_directory = directory where the wav files are saved
%       load_clean: boolean whether to load also the file named clean.wav

    if nargin < 3
        load_clean = false;
    end
    [noisySpeech,fs] = audioread(audio_directory+"noisy.wav");
    noisySpeech = resampleIfNeeded(noisySpeech, target_fs, fs);
    
    if load_clean == true
        
       [cleanSpeech,fs] = audioread(audio_directory+"clean.wav");
       cleanSpeech = resampleIfNeeded(cleanSpeech, target_fs, fs);
    else
        cleanSpeech = 0;
    end
    

    [denoisedSpeech,fs] = audioread(audio_directory+"enhanced.wav");
    denoisedSpeech = resampleIfNeeded(denoisedSpeech, target_fs, fs);
    
end


function [outputA, outputB, outputC] = setLength(startIDX, length, inputA, inputB, inputC)
% SETLENGTH Trims the signals to a range from startIDX to length 

    outputA = inputA(startIDX:startIDX + length);
    outputB = inputB(startIDX:startIDX + length);
    if nargin > 4
        outputC = inputC(startIDX:startIDX + length);
    else
        outputC = 0;
    end
end


function [output] = resampleIfNeeded(input, target_fs, fs)
% RESAMPLEIFNEEDED resamples the input signal if it does not match the
% target sampling rate 
%   input - signal to be resampled
%   target_fs - target sampling rate
%   fs - current sampling rate
    if fs ~= 16000
        output = resample(input, target_fs, fs);
    else
        output = input;
    end
end


function[] = plot_spectrogram(input, fs, name, log, freq_range)
% PLOTSPECTROGRAM plots spectrogram of the input
%   log: boolean - whether to make the frequency axis logarhytmic
    if nargin < 4
        log = true;
    end

    % Parameters
    winLen = 512;
    win = hann(winLen,'periodic');
    ovrlp = winLen/2;
    Blocksize = winLen;
    used_colormap='parula';

    [freq_content,freq_vector,time_vector] = spectrogram(input, win, ovrlp, Blocksize, fs,'yaxis','MinThreshold',-70); 

    figure;
        
    freq_content_dB=20*log10(abs(freq_content));
    plot=pcolor(time_vector,freq_vector,freq_content_dB); 
    axis xy;                   
    colormap(used_colormap);   
    shading interp              
    clim([-60 40]);             

    xlabel('Time [s]');        
    ylabel('Frequency [Hz]');   
    title(name);   
    
    if log == true
       set(gca,'YScale','log');    %Make y-axis logarithmic     
       yticks([50 100 200 500 1000 1500 2000 4000 6000 8000])
       yticklabels({'100','200','500','1000','1500','2000','4000', '6000', '8000'})
    else
       yticks([20, 100, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000,8000])
       yticklabels({'20','100','500','1000','2000','3000','4000','5000', '6000', '7000', '8000'})
    end
    ylim(freq_range);   % Specify data range for the frequency axis    
    bar_handle=colorbar;        %Plot a colorbar next to the spectrogram
    bar_handle.Label.String='Magnitude [dB]';   %Make a label for the colorbar
    set(gca, 'FontSize', 12, 'FontName', 'Helvetica');
    saveas(gcf, getAudioDir+"spectrogram_"+name+".png");
end


function[] = plot_spectrum(input, fs, name)
% PLOTSPECTRUM - plots FFT spectrum of the input 

    % Compute FFT
    L = length(input);
    win = hann(L,"periodic");
    input = input .* win;
    Y = fft(input);

    P2 = abs(Y/L); % Two-sided spectrum
    P1 = P2(1:floor(L/2)+1); % Single-sided spectrum
    P1(2:end-1) = 2*P1(2:end-1); % Correct amplitudes
    f = fs*(0:floor((L/2)))/L; % Frequency vector
    assert(length(f) == length(P1), 'Size mismatch')

    figure;
    semilogx(f, 20*log10(P1 + eps))
    xlim([20 8000]);
    ylim([-110,-30]);
    title("Magnitude Spectrum (FFT) of "+name)
    xlabel('Frequency (Hz)')
    ylabel('|P(f)|')
    grid on
end


function [] = show_waveforms(noisy, clean, denoised, target_fs, name, plot_clean)
% SHOW_WAVEFORMS plots noisy, clean and enhanced in time domain
%       plot_clean: boolean - whether to plot also the clean signal
    
    %%%%%%%%%%%% time window setup: %%%%%%%%
    start_idx = floor(target_fs * 0.6);
    end_idx = start_idx + floor(target_fs * 0.01);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    if plot_clean == true
        time = (0:length(clean)-1) / target_fs;
    else
        time = (0:length(noisy)-1) / target_fs;
    end
    
    figure('Color', [1, 1, 1], 'Position', [100, 100, 1000, 600]); 

    % Plot Noisy
    plot(time(start_idx:end_idx)*1000, noisy(start_idx:end_idx), ...
        'Color', [1, 0, 0], 'LineWidth', 0.7);
    hold on;   

    % Plot Denoised (enhanced)
    plot(time(start_idx:end_idx)*1000, denoised(start_idx:end_idx), ...
        'Color', [0, 0.4470, 0.7410], 'LineWidth', 1.0);

    if plot_clean == true
        % Plot clean
        plot(time(start_idx:end_idx)*1000, clean(start_idx:end_idx), ...
        'k--', 'LineWidth', 1.0);
    end
       
    set(gca, 'Color', [1, 1, 1]);          % White axes background
    set(gca, 'XColor', 'k', 'YColor', 'k'); % Black tick marks and labels
    set(gca, 'GridColor', [0.15, 0.15, 0.15], 'GridAlpha', 0.15); % Subtle dark grid
    set(gca, 'FontSize', 12, 'FontName', 'Helvetica'); % Professional typography
    grid on;   
    xlabel('Time (ms)');
    ylabel('Amplitude');
    if plot_clean == true
        lgd = legend('Noisy', 'Enhanced', 'Clean', 'Location', 'best');
    else
        lgd = legend('Noisy', 'Enhanced', 'Location', 'best');
    end
    set(lgd, 'Color', 'w', 'EdgeColor', [0.5, 0.5, 0.5], 'TextColor', 'k','FontSize', 13); % White background, soft gray border
    title("Time-Domain Phase Alignment: " + name, 'Color', 'k');
    
    saveas(gcf, getAudioDir()+"time_waveform.png");   
end


function [] = phase_stft(clean, noisy, denoised, fs)
% PHASE_STFT plots phase differences between denoised vs. clean signals and
%            noisy vs. clean signals in time-frequency domain

    win = hann(512,'periodic');
    noverlap = 512 / 2;
    nfft = 512;
    
    % Get Spectrograms
    [S_clean, f, t] = stft(clean, fs, 'Window', win, 'OverlapLength', noverlap, 'FFTLength', nfft, 'FrequencyRange', 'onesided');
    [S_noisy, ~, ~] = stft(noisy, fs, 'Window', win, 'OverlapLength', noverlap, 'FFTLength', nfft, 'FrequencyRange', 'onesided');
    [S_enh, ~, ~] = stft(denoised, fs, 'Window', win, 'OverlapLength', noverlap, 'FFTLength', nfft, 'FrequencyRange', 'onesided');
    
    % Calculate phase difference
    phase_err_enh = angle(exp(1i * (angle(S_clean) - angle(S_enh))));
    phase_error_plot_enh = abs(phase_err_enh);

    phase_err_noisy = angle(exp(1i * (angle(S_clean) - angle(S_noisy))));
    phase_error_plot_noisy = abs(phase_err_noisy); 
    
    f_log = logspace(log10(20), log10(fs/2), length(f));
    phase_error_plot_noisy_log = interp1(f, phase_error_plot_noisy, f_log, 'linear', 0);
    phase_error_plot_enh_log   = interp1(f, phase_error_plot_enh,   f_log, 'linear', 0);
        
    % Plot
    freq_ticks = [20 50 100 200 500 1000 2000 5000 10000 20000];
    figure;
    imagesc(t, f_log, phase_error_plot_noisy_log); colorbar;  
    axis xy; 
    set(gca, 'YScale', 'log');
    ylim([20, fs/2]);
    clim([0, pi]);
    set(gca, 'FontSize', 12, 'FontName', 'Helvetica');
    title('Phase Error: Noisy vs Clean');
    yticks(freq_ticks(freq_ticks <= fs/2));
    ylabel('Frequency [Hz]');
    xlabel('Time [s]');
    colormap('hot');
    saveas(gcf, getAudioDir+"phase_error_noi_log.png");
    
    figure;
    imagesc(t, f_log, phase_error_plot_enh_log); colorbar;
    axis xy;
    set(gca, 'YScale', 'log');
    ylim([20, fs/2]);
    clim([0, pi]);
    set(gca, 'FontSize', 12, 'FontName', 'Helvetica');
    title('Phase Error: Enhanced vs Clean');
    yticks(freq_ticks(freq_ticks <= fs/2));
    ylabel('Frequency [Hz]');
    xlabel('Time [s]');
    colormap('hot'); 
    saveas(gcf, getAudioDir+"phase_error_enh_log.png");
end


function [delays_ms, time_axis] = estimate_time_varying_delay(clean, enhanced, fs, window_ms, hop_ms)
% for plotting the cross correlations - for investigating time-variance of
% the delay

    % Convert window/hop to samples
    window_len = round(window_ms * fs / 1000);
    hop_len = round(hop_ms * fs / 1000);

    % Number of windows
    num_windows = floor((N - window_len) / hop_len) + 1;
    delays_ms = zeros(num_windows, 1);
    time_axis = zeros(num_windows, 1);

    for k = 1:num_windows
        start_idx = (k-1)*hop_len + 1;
        end_idx = start_idx + window_len - 1;

        x = clean(start_idx:end_idx);
        y = enhanced(start_idx:end_idx);

        % Remove DC offset
        x = x - mean(x);
        y = y - mean(y);

        % skip low-energy windows
        if rms(x) < 1e-4
            delays_ms(k) = NaN;
            continue;
        end

        % Cross-correlation
        [corr_vals, lags] = xcorr(y, x);

        % Find best alignment
        [~, idx] = max(abs(corr_vals));
        best_lag = lags(idx);

        % Convert to milliseconds
        delays_ms(k) = best_lag / fs * 1000;

        % Time axis (window center)
        center_sample = start_idx + window_len/2;
        time_axis(k) = center_sample / fs;
    end

    % Plot results
    figure;
    plot(time_axis, delays_ms, 'LineWidth', 1.5);
    ylim([-20 20]);
    xlabel('Time [s]');
    ylabel('Estimated Delay [ms]');
    title('Time-Varying Delay Estimation');
    grid on;
end


%% WHOLE ANALYSIS:

function [] = analysisWithReference(audio_directory, target_fs)
% performs analysis with clean reference file, noisy file and enhanced file
% audio_directory is path where the .wav files are saved. 
% Files in the direcotry must be named: noisy.wav, enhanced.wav, clean.wav
    
    % load audio
    [noisy, clean, denoised] = loadFiles(16000, audio_directory, true);
    
    % set length
    min_value = min(min(length(denoised), length(noisy)),length(clean))-1;
    [noisy, denoised, clean] = setLength(1, min_value, noisy, denoised, clean);
    
    % plot phase errors
    phase_stft(clean, noisy, denoised, 16000);
    
    plot_spectrogram(noisy, 16000, "Noisy", true,[0 8000]);
    plot_spectrogram(clean, 16000, "Clean", true,[0 8000]);
    plot_spectrogram(denoised, 16000, "Enhanced", true,[0 8000]);
    
    show_waveforms(noisy, clean, denoised,target_fs, "",true);  
end

function [] = analysisWithoutReference(audio_directory, target_fs)
% performs analysis without clean reference file, only noisy file and
% enhanced file are analysed
% audio_directory is path where the .wav files are saved. 
% Files in the direcotry must be named: noisy.wav, enhanced.wav, clean.wav
    [noisy, ~, denoised] = loadFiles(16000, audio_directory, false);
    
    % set length
    min_value = min(length(denoised), length(noisy))-1;
    [noisy, denoised, clean] = setLength(1, min_value, noisy, denoised);
    
    plot_spectrogram(noisy, 16000, "Noisy", true,[0 8000]);
    plot_spectrogram(denoised, 16000, "Enhanced", true,[0 8000]);
    
    show_waveforms(noisy, clean, denoised,target_fs,"", false);
end


function audio_directory = getAudioDir()
    audio_directory = "";  
end
audio_directory = getAudioDir();

% analysisWithoutReference(audio_directory, 16000);
% analysisWithReference(audio_directory, 16000);
