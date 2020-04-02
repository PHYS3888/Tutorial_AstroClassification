function plotTimeSeries(TimeSeries,plotIndex,maxL)
% Plot a selected time series, indexed as plotIndex,
% from a table of time series, TimeSeries.
%-------------------------------------------------------------------------------
% Default: plot the first time series in the table
if nargin < 2
    plotIndex = 1;
end
% Default: plot the first 1000 samples
if nargin < 3
    maxL = 1000;
end
%-------------------------------------------------------------------------------
% FILL IN THE BLANKS ... BELOW:

% Sampling rate (Hz) of Kepler data:
fs = ...;

% Retrieve data:
x = TimeSeries.Data{plotIndex};

% Produce time axis
N = length(x);
tSec = (1:1/fs:N/fs)';
tDay = ...; % convert tSec (seconds) to tDay (days)

% Plot:
plot(tDay(1:maxL),x(1:maxL),'k')
xlabel('Time (day)')

title(sprintf('%s (%s)',TimeSeries.Name{plotIndex},...
                        TimeSeries.Keywords{plotIndex}))

end
