% Spectrogram generation for training CNN
function output = spectogram_extraction(audio, display)

    %set display to false if not provided
    if nargin < 2
        display = false;
    end

    % Parameters
    sampling_rate = 16000;                % 16 kHz
    window_length = 0.025 * sampling_rate; % 25 ms window in samples
    hop_length = window_length / 4;       % 25% overlap
    alpha = 2.5;                           % Gaussian shape parameter
    frequency_bins = 512;

    % Create Gaussian window
    window = gausswin(round(window_length), alpha);

    %ensure audio is one window length
    if length(audio) < window_length
        padding_size = window_length - length(audio);
        audio = [audio; zeros(padding_size,1)];
    end

    % Create a Spectrogram
    % [S, F, T] = spectrogram(audio, window, overlap, NFFT, Fs)
    % audio   : input audio signal
    % window  : window vector applied to each segment (Gaussian)
    % overlap : number of overlapping samples between windows (window_length - hop_length)
    % NFFT    : number of FFT points (frequency resolution)
    % Fs      : sampling frequency of the audio (Hz)
    % S       : complex spectrogram matrix(time x frequency x powerr)
    % F       : frequency vector
    % T       : time vector
    [S, F, T] = spectrogram(audio, window, round(window_length - hop_length), frequency_bins, sampling_rate);

    %convert to magnitude spectrogram and log scale(decibel scale)
    spec = 20*log10(abs(S) + 1e-6);
   
    % Z-score normalization for CNN training
    spec = zscore(spec(:));             % normalize all values to mean 0, std 1
    spec = reshape(spec, size(S));      % reshape back to original spectrogram size

    % Resize as CNN requires fixed input size
    output = imresize(spec, [1024 1024]); % Resize to 1024x1024


    % Display the spectrogram if required
    if display
        % Original axes (before resizing)
        figure;
        imagesc(T, F, spec); 
        axis xy;
        colormap jet;
        colorbar;
        title('Original Spectrogram (Log-Magnitude)');
        xlabel('Time (s)');
        ylabel('Frequency (Hz)');

        % Resized 1024x1024 spectrogram
        figure;
        imagesc(linspace(0, max(T), 1024), linspace(0, max(F), 1024), output);
        axis xy;
        colormap jet;
        colorbar;
        title('Resized Spectrogram (Log-Magnitude)');
        xlabel('Time (s)');
        ylabel('Frequency (Hz)');
    end

end
