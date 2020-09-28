function X = LoadTestData()

theFiles = {'9664607.txt','10933414.txt'};
numFiles = length(theFiles);
X = cell(numFiles,1);
for i = 1:numFiles
    fileLocation = fullfile('toTest',theFiles{i});
    X{i} = dlmread(fileLocation,'',1,1);
end

end
