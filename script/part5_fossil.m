%{
class-1, cluster-1, acc:0.6594 (total#:323)
class-1, cluster-2, acc:0.8418 (total#:158)
class-2, cluster-1, acc:0.6852 (total#:305)
class-2, cluster-2, acc:0.5607 (total#:107)
    0.7193
    0.6529
    0.6887

linear reg acc: 0.6898, (alpha=10.0000)
linear SVM acc 0.6898
%}
clear
close all
clc

addpath(genpath('../ompbox10'));
addpath(genpath('../spams-matlab'));

%% load modern pollen dataset
flagSelective = 1;

databaseDIR = '..\LOO_combo\database_CNNfeature_CanonicalShape';
outputFileName = '.\result_txt.txt';

setbyLayer = dir(databaseDIR);
a = zeros(numel(setbyLayer)-2,1);
for i = 3:numel(setbyLayer)
    aa = strfind(setbyLayer(i).name, '_');
    a(i-2) = str2double(  setbyLayer(i).name(aa(1)+1:aa(2)-1)  );
    setbyLayer(i).layerID = a(i-2);
end
layerIDall = unique(a);

%% fetch the training set of modern pollens
Tlist = 50:15:50;
lambdaList = 0.00005;%[0.00005, 0.0001, 0.0005];

accTensor = zeros(length(Tlist), length(lambdaList),3);
errorList = [];


% for tt = 1: length(Tlist)
%     for llambda = 1:length(lambdaList)
tt = 1;
llambda = 1;        
T = Tlist(tt);
lambda = lambdaList(llambda);
LayerID = 5;
LAYERID = 24;

dataSetName = strcat('ModernPollenMatchByCNN_CanonicalShape_layer_', num2str(LAYERID), '_Dataset_LOOV.mat');
fprintf('CNNfeature at layer-%d...\n', LAYERID);

strContent = 'CNNfeature at layer-%d...\r\n';
fileID = fopen(outputFileName, 'a+');
fprintf(fileID, strContent, LAYERID);
fclose(fileID);

if exist(strcat('./',dataSetName), 'file')
    load(strcat('./',dataSetName));
else
    categoryList = [];
    for i = 1:numel(setbyLayer)
        if setbyLayer(i).layerID == LAYERID
            categoryList = [categoryList, setbyLayer(i)];
        end
    end
    dataSet = cell(2,1);
    fprintf('fetch data...\n');
    
    strContent = 'fetch data...\r\n';
    fileID = fopen(outputFileName,'a+'); fprintf(fileID, strContent); fclose(fileID);
    
    for i = 1:numel(categoryList)
        fprintf('\ncategory-%s...\n', categoryList(i).name);
        
        strContent = '\r\ncategory-%s...\r\n';
        fileID = fopen(outputFileName,'a+'); fprintf(fileID, strContent, categoryList(i).name);     fclose(fileID);
        
        imList = dir( fullfile(databaseDIR, categoryList(i).name,'*.mat') );
        dataSet{i}.name = categoryList(i).name;
        dataSet{i}.data = cell(numel(imList),1);
        dataSet{i}.clusterID = zeros(numel(imList),1);
        for imIDX = 1:numel(imList)
            fprintf('.');
            fileName =  fullfile(databaseDIR, categoryList(i).name, imList(imIDX).name );
            mat = load(fileName);
            %mat = mat ./ repmat( sqrt(sum(mat.^2,1)), size(mat,1),1 );
            dataSet{i}.data{imIDX} = mat;
            a = strfind( imList(imIDX).name, '.mat');
            dataSet{i}.clusterID(imIDX) = str2double(imList(imIDX).name(a-1));
        end
    end
    save(dataSetName, 'dataSet', 'dataSetName', 'categoryList');
    fprintf('\nDone!\n');
    
    strContent = '\r\nDone!\r\n';
    fileID = fopen(outputFileName,'a+'); fprintf(fileID, strContent);     fclose(fileID);
end

%% fetch the training set of modern pollens
clusterIDX = unique(dataSet{1}.clusterID);
cluserK = length(clusterIDX);

dictSet = cell(numel(dataSet), cluserK); % training set is represented by cells, of size  #category-by-#ShapeClusters
dictSetLoc = cell(numel(dataSet), cluserK); % location of each patch
dictSetImgLabel = cell(numel(dataSet), cluserK); % indicate which image the patch belongs to
count = 1;
for i_class = 1:size(dictSet,1)
    for i_img = 1:numel(dataSet{i_class}.data)
        i_cluster = dataSet{i_class}.clusterID(i_img);
        imFeat = dataSet{i_class}.data{i_img}.patchFeat(1:end-2,:);
        tmp = sqrt(sum(imFeat.^2,1));
        tmpIdx = find(tmp>1);
        imFeat(:,tmpIdx) = imFeat(:,tmpIdx) ./ repmat( tmp(tmpIdx), size(imFeat,1), 1 );
        dictSet{i_class,i_cluster} = [dictSet{i_class,i_cluster}, imFeat];%dataSet{1}.data{i}.patchFeat];
        
        patchLoc = dataSet{i_class}.data{i_img}.patchFeat(end-1:end,:);
        patchLoc = patchLoc-1;
        imSize = dataSet{i_class}.data{i_img}.imSize;
        feaSize = dataSet{i_class}.data{i_img}.feaSize(1:2);
        patchLoc = bsxfun(@rdivide, patchLoc, feaSize(:));
        patchLoc = bsxfun(@times, patchLoc, imSize(:));
        patchLoc = patchLoc + 1;
        patchLoc = bsxfun(@minus, patchLoc, imSize(:)/2);
        dictSetLoc{i_class,i_cluster} = [dictSetLoc{i_class,i_cluster}, patchLoc]; % dataSet{1}.data{i}.patchFeat];
        dictSetImgLabel{i_class,i_cluster} = [dictSetImgLabel{i_class,i_cluster}, i_img*ones(1,size(patchLoc,2))];
        count = count + 1;
    end
end

%% generate the dictionary from training set
% if i_cluster == 1
%     numA = 1860; % 1860
%     numB = 1136; % 1136
% else
%     numA = 3135; % 3135
%     numB = 1699; % 1699
% end
numAllocation = [ [2100;2100], [2100;2100]]; % [ [1860;1136], [3135;1699]];

for i_cluster = 1:size(dictSet,2)
    for i_class = 1:size(dictSet,1)
        a = numAllocation(i_class,i_cluster);
        
        A = dictSet{i_class,i_cluster};
        dictSet{i_class,i_cluster} = A(:,1:a);
        
        A = dictSetImgLabel{i_class,i_cluster};
        dictSetImgLabel{i_class,i_cluster} = A(1:a);
        A = dictSetLoc{i_class,i_cluster};
        dictSetLoc{i_class,i_cluster} = A(:,1:a);
    end
end

%% fetch fossil pollen set
databaseDIR = './database_fossil_CNNfeature_CanonicalShape';

setbyLayer = dir(databaseDIR);
a = zeros(numel(setbyLayer)-2,1);
for i = 3:numel(setbyLayer)
    aa = strfind(setbyLayer(i).name, '_');
    a(i-2) = str2double(  setbyLayer(i).name(aa(1)+1:aa(2)-1)  );
    setbyLayer(i).layerID = a(i-2);
end

categoryList = [];
for i = 1:numel(setbyLayer)
    if setbyLayer(i).layerID == LAYERID
        categoryList = [categoryList, setbyLayer(i)];
    end
end
dataSet = cell(2,1);
fprintf('fetch fossil data...\n\n');

strContent = 'fetch data...\r\n';
fileID = fopen(outputFileName,'a+'); fprintf(fileID, strContent); fclose(fileID);


imNameList = cell(1,2);
for i = 1:numel(categoryList)
    fprintf('category-%s...\n', categoryList(i).name);
    
    strContent = '\r\ncategory-%s...\r\n';
    fileID = fopen(outputFileName,'a+'); fprintf(fileID, strContent, categoryList(i).name);     fclose(fileID);
    
    imList = dir( fullfile(databaseDIR, categoryList(i).name,'*.mat') );
    imNameList{i} = imList;
    dataSet{i}.name = categoryList(i).name;
    dataSet{i}.data = cell(numel(imList),1);
    dataSet{i}.clusterID = zeros(numel(imList),1);
    for imIDX = 1:numel(imList)
        fprintf('.');
        fileName =  fullfile(databaseDIR, categoryList(i).name, imList(imIDX).name );
        mat = load(fileName);
        %mat = mat ./ repmat( sqrt(sum(mat.^2,1)), size(mat,1),1 );
        dataSet{i}.data{imIDX} = mat;
        a = strfind( imList(imIDX).name, '.mat');
        dataSet{i}.clusterID(imIDX) = str2double(imList(imIDX).name(a-1));
    end
    fprintf('\n');
end

%% matching for classification
labelpred = [];
labelTrue = [];
clusterLabel = [];
errorList = [];
count = 1;
for i_class = numel(dataSet):-1:1
    fprintf('\n\tcategory-%d testing...\n', i_class);
    
    strContent = '\r\n\tcategory-%d testing...\r\n';
    fileID = fopen(outputFileName,'a+'); fprintf(fileID, strContent, i_class); fclose(fileID);
    
    for i_img = 1:numel(dataSet{i_class}.data)
        fprintf('\t image-%d ---', i_img);
        
        strContent = '\t image-%d ---';
        fileID = fopen(outputFileName,'a+'); 
        fprintf(fileID, strContent, i_img); 
        fclose(fileID);
        
        %% testing image and its cluster label 
        i_cluster = dataSet{i_class}.clusterID(i_img);        
        A = dataSet{i_class}.data{i_img};
        imFeat = A.patchFeat(1:end-2,:);
        tmp = sqrt(sum(imFeat.^2,1));
        tmpIdx = find(tmp>1);
        imFeat(:,tmpIdx) = imFeat(:,tmpIdx) ./ repmat( tmp(tmpIdx), size(imFeat,1), 1 );
        
        %% the patches locations for this testing image 
        patchLoc = A.patchFeat(end-1:end,:);
        patchLoc = patchLoc-1;        
        imSize = A.imSize;
        feaSize = A.feaSize(1:2);        
        patchLoc = bsxfun(@rdivide, patchLoc, feaSize(:));
        patchLoc = bsxfun(@times, patchLoc, imSize(:));
        patchLoc = patchLoc + 1;
        patchLoc = bsxfun(@minus, patchLoc, imSize(:)/2);
        
        %% dictionary
%         dict = dictSet(:,i_cluster);
%         Dloc = dictSetLoc(:,i_cluster);
        Dict = [dictSet{1,i_cluster}, dictSet{2,i_cluster}];
        Dloc = [dictSetLoc{1,i_cluster}, dictSetLoc{2,i_cluster}];
        DictLabel = [ones(1,size(dictSet{1,i_cluster},2)), 2*ones(1,size(dictSet{2,i_cluster},2))];
        
        %% locality penalty 
        W = zeros(size(Dict,2), size(patchLoc,2));
        for i = 1 : size(patchLoc,2)
            a = bsxfun(@minus, Dloc, patchLoc(:,i) );
            a = sum(a.^2);
            a = sqrt(a(:));
            W(:, i) = a;
        end
        
        param.L = T; % not more than param.L non-zeros coefficients (default: min(size(D,1),size(D,2)))
        param.lambda = lambda;
        param.numThreads = -1; % number of processors/cores to use; the default choice is -1 and uses all the cores of the machine
        param.mode = 2; % penalized formulation
        
        A = mexLassoWeighted(imFeat, Dict, W, param);
        
        errList = ones(numel(dataSet),1)*Inf;
        for i = 1:numel(dataSet)
            a = find(DictLabel==i);
            err = imFeat-Dict(:,a)*A(a,:);
            err = sum(err(:).^2);
            errList(i) = err;
        end
        
        errorList = [errorList, errList(:)];
        [valMIN, idxMIN] = min(errList);
        
        if idxMIN == i_class
            fprintf('correct -- class-%d cluster-%d\n', i_class,  i_cluster);
            strContent = 'correct\r\n';
            fileID = fopen(outputFileName,'a+'); fprintf(fileID, strContent); fclose(fileID);        
        else
            fprintf('wrong -- class-%d cluster-%d\n', i_class,  i_cluster);
            strContent = 'wrong\r\n';
            fileID = fopen(outputFileName,'a+'); fprintf(fileID, strContent); fclose(fileID);   
        end
        
        labelTrue = [labelTrue, i_class];
        labelpred = [labelpred, idxMIN];
        clusterLabel = [clusterLabel, i_cluster];
    end
end


acc = mean(labelTrue == labelpred);
fprintf('\naccuracy=%.4f (lambda=%.4f, T=%d)\n\n', acc, lambda, T);

strContent = '\r\naccuracy=%.4f (lambda=%.4f, T=%d)\r\n\r\n';
fileID = fopen(outputFileName,'a+'); fprintf(fileID, strContent, acc, lambda, T); fclose(fileID);

accTensor(tt, llambda, 3) = acc;
for i = 1:numel(dataSet)
    a = find(labelTrue==i);
    accTensor(tt, llambda, i) = mean(labelTrue(a) == labelpred(a));
end
%     end
% end

save( strcat('resultFossil_', num2str(acc), '.mat'), ...
    'accTensor', 'Tlist', 'lambdaList', 'errorList', 'clusterLabel', 'labelTrue', 'labelpred');

%% learning weights to register the errors
addpath(genpath('../libsvm-3.20'));

for i_class = 1:2
    for i_cluster = 1:2
        a = find(labelTrue==i_class & clusterLabel ==i_cluster);
        lt = labelTrue(a);
        lp = labelpred(a);
        fprintf('class-%d, cluster-%d, acc:%.4f (total#:%d)\n', i_class, i_cluster, mean(lt==lp), length(lt));
    end
end

disp(squeeze(accTensor))

Y = round(labelTrue-1.5);
errorListTMP = errorList;
X = [errorListTMP; ones(1,size(errorList,2))];


alphaList = [0, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 5, 10:3:120];
accList = [];
for i_alpha = 1:length(alphaList)
    alpha = alphaList(i_alpha);
    W = (X*X'+alpha*eye(size(X,1)))\X*Y';
    
    Yhat = W'*X;
    thr = 0.0;
    Yhat(Yhat<thr) = -1;
    Yhat(Yhat>=thr) = 1;
    %fprintf('alpha=%.4f, acc=%.4f\n', alpha, mean(Y==Yhat));
    accList(end+1) = mean(Y==Yhat);
end

[val, idx] = max(accList);
fprintf('linear reg acc: %.4f, (alpha=%.4f)\n', val, alphaList(idx));

%% svm
cl = fitcsvm(X', Y, ...
    'KernelFunction', 'linear',...
    'ClassNames', [-1,1]);

[Yhat, b] = predict(cl, X');
fprintf('linear SVM acc %.4f\n\n', mean(Yhat(:)==Y(:)));

%% copy images that are incorrectly classified
%{
databaseSourceDir = './database_fossil_canonicalShape';
tmpDIR = './incorrectClassifiedImage';
if isdir( tmpDIR )
    rmdir(tmpDIR,'s');
end
mkdir(tmpDIR);

categoryList = dir( strcat(databaseSourceDir, '/* fossil'));
for i_class = 1:numel(dataSet)
    categoryName = categoryList(i_class).name;
    imList = dir( fullfile(databaseSourceDir, categoryName,'*.jpg') );
    a = find( labelTrue==i_class & labelTrue~=labelpred);
    for i = 1:length(a)
        imFileName = imList(i).name;
        im = imread( fullfile(databaseSourceDir, categoryList(i_class).name, imFileName) );
        copyfile( fullfile(databaseSourceDir, categoryList(i_class).name, imFileName),...
            fullfile(tmpDIR, strcat(categoryName, imFileName))  );
    end
end
%}











