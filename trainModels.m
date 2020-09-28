function [Mdl_SVMlinear,Mdl_SVMrbf] = trainModels(dataMatrix,dataLabels)

if length(categories(dataLabels)) == 2
    % Train a linear SVM:
    Mdl_linearSVM = fitcsvm(dataMatrix,dataLabels,...
                        'Standardize',true,...
                        'KernelFunction','linear');
    % Train an rbf SVM:
    Mdl_rbfSVM = fitcsvm(dataMatrix,dataLabels,...
                    'Standardize',true,...
                    'KernelFunction','rbf',...
                    'KernelScale','auto');
else
    % Train a linear SVM:
    tLinear = templateSVM('Standardize',true,'KernelFunction','linear');
    Mdl_SVMlinear = fitcecoc(dataMatrix,dataLabels,'Learners',tLinear);

    % Train an rbf SVM:
    tRBF = templateSVM('Standardize',true,'KernelFunction','rbf',...
                'KernelScale','auto');
    Mdl_SVMrbf = fitcecoc(dataMatrix,dataLabels,'Learners',tRBF);
end

end
