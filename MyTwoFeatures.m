function myFeatures = MyTwoFeatures(x,fs)
% Computes two features of an input time series, x
%-------------------------------------------------------------------------------

% ---
% Recall that the integral of the PSD function is equal to
% the variance of the signal.
% ---
% Start by z-scoring to remove arbitrary variance-dependence
y = zscore(x);
% (now work off of y)

%-------------------------------------------------------------------------------
% Compute power spectral density:
t = (1:1/fs:N/fs)';
xTable = timetable(seconds(t),y);
[pxx,f] = pspectrum(xTable);

% Compute two features of the power spectrum
myFeatures = zeros(2,1);

%-------------------------------------------------------------------------------
% Feature 1
% (e.g., max of power spectral density)

myFeatures(1) = ...;

%-------------------------------------------------------------------------------
% Feature 2
% (e.g., use bandpower)

myFeatures(2) = bandpower(...);

end
