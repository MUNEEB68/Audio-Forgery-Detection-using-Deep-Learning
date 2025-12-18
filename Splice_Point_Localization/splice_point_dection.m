clc; clear; close all;

% file reading and fixed parameter
audioFile = 'C:\Users\Hasnat Ahmad\OneDrive\Desktop\Audio-Forgery-Detection-using-Deep-Learning-main\HAD_test_00000001.wav';
frameSize = 0.03;  % size of each frame
frameShift = 0.01; % jump from frame to frame
windowDuration = 0.5; %window of comparision
K_threshold = 0.8; %sens

%% Load audio
[audioData, Fs] = audioread(audioFile);
if size(audioData,2) > 1, audioData = mean(audioData,2); end
audioData = audioData / max(abs(audioData)); % detcting if audio is stereo or mono and converting to moino

% Frame parameters
winLen = round(frameSize * Fs); %o of samples per frame
overlap = winLen - round(frameShift * Fs);%no of samples overlapping 

% Extract features
[s, f, ~] = spectrogram(audioData, hamming(winLen), overlap, [], Fs); %taking short time fourier transform
%s --> complex-valued spectrogram (frequency × time). Each column is the FFT of a frame.
%f --> freq
%~ --> time unused
magSpec = abs(s);% converts value of s into magnitudes
centroid = (f' * magSpec) ./ sum(magSpec,1); centroid = centroid(:);% calculates the balance poi nt of energies of each frame

if exist('pitch','file')
    f0 = pitch(audioData, Fs, 'WindowLength', winLen, 'OverlapLength', overlap, 'Range',[50 400]);%pitch extraction
    f0 = resample(f0, length(centroid), length(f0));%interpolation yto match centroid vector length 
end

if exist('mfcc','file')% copmputing mfcc to get texture of audio
    mfccs = mfcc(audioData, Fs, 'Window', hamming(winLen,'periodic'), 'OverlapLength', overlap, 'NumCoeffs', 13);
else
    mfccs = log(sum(magSpec,1))';
end

minLen = min([length(centroid), length(f0), size(mfccs,1)]);
centroid = centroid(1:minLen); f0 = f0(1:minLen); mfccs = mfccs(1:minLen,:);

% Combine & normalize
features = [f0, centroid, mfccs];
features = zscore(features); % normalizing and balancing of weights/coeff
features = features .* [2,2,ones(1,size(mfccs,2))]; %choose how senstivie which is

% Sliding window divergence
numFrames = size(features,1);%total no of fram
framesInWindow = round(windowDuration/frameShift);%frames per window
divCurve = zeros(numFrames,1);%zero array fr score

for i = framesInWindow+1:numFrames-framesInWindow
    meanPast = mean(features(i-framesInWindow:i-1,:),1);%cals past feature value
    meanFuture = mean(features(i:i+framesInWindow-1,:),1);%FUTURER VALUE
    divCurve(i) = norm(meanFuture - meanPast);% making splicing detction less noisy?smoothensdivergence curve
end

divCurve = smoothdata(divCurve,'movmean',10);
threshold = mean(divCurve) + K_threshold*std(divCurve);% bvaseline divergence+ self adjustment applied
[pks, locs] = findpeaks(divCurve,'MinPeakHeight',threshold,'MinPeakDistance',framesInWindow);%only consider peaks for splice(0realized from general overview),locs-->indices of frame 
detectedTimes = locs * frameShift; % actual timestamps  

% Plot
t_axis = (0:length(audioData)-1)/Fs;
t_feat = (0:numFrames-1)*frameShift;

figure('Color','w');
subplot(3,1,1); plot(t_axis,audioData); title('Audio Signal'); axis tight;
subplot(3,1,2); plot(t_feat,f0,'g'); title('Pitch (F0)'); axis tight; grid on;
subplot(3,1,3); plot(t_feat,divCurve,'LineWidth',2); hold on;
yline(threshold,'r--'); plot(detectedTimes,pks,'ro','MarkerFaceColor','r');
title('Frequency & Timbre Difference'); xlabel('Time (s)'); ylabel('Difference'); grid on; axis tight;

fprintf('Detected %d potential splices.\n', length(detectedTimes));