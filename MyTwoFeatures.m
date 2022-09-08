function myFeatures = MyTwoFeatures(x)
% Computes two features of an input time series, x
%-------------------------------------------------------------------------------

% ---
% Recall that the integral of the PSD function is equal to
% the variance of the signal.
% ---
% Start by z-scoring to remove arbitrary variance-dependence
y = zscore(x);
% Length:
N = length(y);
% (now work off of y)

%-------------------------------------------------------------------------------
% Load in the Kepler sampling rate, fs:
fs = KeplerSamplingRate();

% Compute power spectral density:
t = (1:1/fs:N/fs)';
xTable = timetable(seconds(t),y);
[pxx,f] = pspectrum(xTable);

%-------------------------------------------------------------------------------
% Compute two features of the power spectrum
myFeatures = zeros(2,1);

%-------------------------------------------------------------------------------
% Feature 1: peaks of the power spectrum
% (e.g., max of power spectral density, pxx)

myFeatures(1) = ...;

%-------------------------------------------------------------------------------
% Feature 2
% (e.g., use bandpower on the power spectrum of y):
% You need to fill in a minimum and maximum frequency for your band!
% (You can check the syntax by running 'doc bandpower')

myFeatures(2) = bandpower(pxx,f,[...,...],'psd');

end
