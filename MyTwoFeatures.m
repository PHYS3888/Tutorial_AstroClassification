function f = MyTwoFeatures(x)
% Computes two features of an input time series, x
%-------------------------------------------------------------------------------

% ---
% Recall that the integral of the PSD function is equal to
% the variance of the signal.
% ---
% Start by z-scoring to remove arbitrary variance-dependence
y = zscore(x);
% (now work off of y)


end
