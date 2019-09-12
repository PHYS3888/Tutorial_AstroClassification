function ToFrequency(x,fs,doPlot,oldMatlab)

% Set default sampling rate:
if nargin < 2
    fs = 1;
end
% Set default plotting behavior:
if nargin < 3
    doPlot = true;
end
if nargin < 4
    oldMatlab = false;
end
%-------------------------------------------------------------------------------

N = length(x);
t = (1:1/fs:N/fs)';

if oldMatlab
    Y = fft(x);
    P2 = abs(Y/N);
    P1 = P2(1:N/2+1);
    P1(2:end-1) = 2*P1(2:end-1);
    pxx = P1.^2;
    f = fs*(0:(N/2))/N;
else
    xTable = timetable(seconds(t),zscore(x));
    [pxx,f] = pspectrum(xTable);
end

if doPlot
    plot(f,pow2db(pxx),'k')
    grid('on')
    xlabel('Frequency (Hz)')
    ylabel('Power Spectrum (dB)')
end

end
