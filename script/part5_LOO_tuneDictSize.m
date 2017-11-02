clear
close all
clc

addpath(genpath('../ompbox10'));
addpath(genpath('../spams-matlab'));

portionList = .3;%[0.1,0.2,0.3,0.4,0.5,0.6];
for i_portion = 1:length(portionList)
    portion = portionList(i_portion);

%% load dataset
flagSelective = 1;

databaseDIR = '.\database_CNNfeature_CanonicalShape';
outputFileName = '.\result_txt.txt';


setbyLayer = dir(databaseDIR);
a = zeros(numel(setbyLayer)-2,1);
for i = 3:numel(setbyLayer)
    aa = strfind(setbyLayer(i).name, '_');
    a(i-2) = str2double(  setbyLayer(i).name(aa(1)+1:aa(2)-1)  );
    setbyLayer(i).layerID = a(i-2);
end
layerIDall = unique(a);
% accList = zeros(2, length(layerIDall));

%% iteratively study

Tlist = 125:15:125;
lambdaList = 0.0001;%[0.00005, 0.0001, 0.0005];

accTensor = zeros(length(Tlist), length(lambdaList),3);
errorList = [];


for tt = 1: length(Tlist)
    for llambda = 1:length(lambdaList)
T = Tlist(tt);
lambda = lambdaList(llambda);
LayerID = 5;
LAYERID = 24;

dataSetName = strcat('PollenMatchByCNN_CanonicalShape_layer_', num2str(LAYERID), '_Dataset_LOOV.mat');
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
%     save(dataSetName, 'dataSet', 'dataSetName', 'categoryList');
    fprintf('\nDone!\n');
    
    strContent = '\r\nDone!\r\n';
    fileID = fopen(outputFileName,'a+'); fprintf(fileID, strContent);     fclose(fileID);
end

%% get the training set of modern pollens
clusterIDX = unique(dataSet{1}.clusterID);
cluserK = length(clusterIDX);

dataSetSplit = cell(numel(dataSet), cluserK); % training set is represented by cells, of size  #category-by-#ShapeClusters
dataSetLoc = cell(numel(dataSet), cluserK); % location of each patch
dataSetImgLabel = cell(numel(dataSet), cluserK); % indicate which image the patch belongs to
count = 1;
for i_class = 1:size(dataSetSplit,1)
    for i_img = 1:numel(dataSet{i_class}.data)
        i_cluster = dataSet{i_class}.clusterID(i_img);
        imFeat = dataSet{i_class}.data{i_img}.patchFeat(1:end-2,:);
        tmp = sqrt(sum(imFeat.^2,1));
        tmpIdx = find(tmp>1);
        imFeat(:,tmpIdx) = imFeat(:,tmpIdx) ./ repmat( tmp(tmpIdx), size(imFeat,1), 1 );
        dataSetSplit{i_class,i_cluster} = [dataSetSplit{i_class,i_cluster}, imFeat];%dataSet{1}.data{i}.patchFeat];
        
        patchLoc = dataSet{i_class}.data{i_img}.patchFeat(end-1:end,:);
        patchLoc = patchLoc-1;
        imSize = dataSet{i_class}.data{i_img}.imSize;
        feaSize = dataSet{i_class}.data{i_img}.feaSize(1:2);
        patchLoc = bsxfun(@rdivide, patchLoc, feaSize(:));
        patchLoc = bsxfun(@times, patchLoc, imSize(:));
        patchLoc = patchLoc + 1;
        patchLoc = bsxfun(@minus, patchLoc, imSize(:)/2);
        dataSetLoc{i_class,i_cluster} = [dataSetLoc{i_class,i_cluster}, patchLoc]; % dataSet{1}.data{i}.patchFeat];
        dataSetImgLabel{i_class,i_cluster} = [dataSetImgLabel{i_class,i_cluster}, i_img*ones(1,size(patchLoc,2))];
        count = count + 1;
    end
end


%% matching for classification
labelpred = [];
labelTrue = [];
clusterLabel = [];

count = 1;
for i_class = 1:numel(dataSet)
    fprintf('\n\tcategory-%d testing...\n', i_class);
    
    strContent = '\r\n\tcategory-%d testing...\r\n';
    fileID = fopen(outputFileName,'a+'); fprintf(fileID, strContent, i_class); fclose(fileID);
    
    for i_img = 1:numel(dataSet{i_class}.data)
        fprintf('\t image-%d ---', i_img);
        
        strContent = '\t image-%d ---';
        fileID = fopen(outputFileName,'a+'); 
        fprintf(fileID, strContent, i_img); 
        fclose(fileID);
        
        %% the cluster label of this testing image
        i_cluster = dataSet{i_class}.clusterID(i_img);        
        
        %% testing image 
        a = find( dataSetImgLabel{i_class,i_cluster}==i_img );
        testImg = dataSetSplit{i_class,i_cluster};
        testImg = testImg(:,a);
        
        %% the patches locations for this testing image        
        patchLoc = dataSetLoc{i_class,i_cluster};
        patchLoc = patchLoc(:,a);
        
        %% dictionary
        
        a = find( dataSetImgLabel{i_class,i_cluster} ~= i_img );     
        numAtoms = 2000;
        
        if i_cluster == 1
            numA = 1860; % 1860
            numB = 1136; % 1136
        else
            numA = 3135; % 3135
            numB = 1699; % 1699
        end
        
        %{
        if i_cluster == 1
            numA = 1900;
            numB = 1100;
        else
            numA = 1500;
            numB = 1700;
        end
        %}
        
        if i_class == 1
            A = dataSetSplit{1,i_cluster};
            A = A(:,a);
            B = dataSetSplit{2,i_cluster};
            
%             numA = round(portion*size(A,2));
%             numB = round(portion*size(B,2));
%             numA2 = 2000;
%             numB2 = 2000;
            
            A = A(:,1:numA);
            B = B(:,1:numB);
            Dict = [ A, B ];
            
            A = dataSetLoc{1,i_cluster};
            A = A(:,a);
            B = dataSetLoc{2,i_cluster};
            
            A = A(:,1:numA);
            B = B(:,1:numB);
            
            Dloc = [ A, B ];
            DictLabel = [ones(1,size(A,2)), 2*ones(1,size(B,2))];
        else
            B = dataSetSplit{1,i_cluster};            
            A = dataSetSplit{2,i_cluster};
            A = A(:,a);            
            
%             numA = round(portion*size(A,2));
%             numB = round(portion*size(B,2));
                        
            A = A(:,1:numA);
            B = B(:,1:numB);
            
            Dict = [ B, A ];
            
            B = dataSetLoc{1,i_cluster};
            A = dataSetLoc{2,i_cluster};
            A = A(:,a);            
            
            A = A(:,1:numA);
            B = B(:,1:numB);
            
            Dloc = [ B, A ];
            DictLabel = [ones(1,size(B,2)), 2*ones(1,size(A,2))];
        end
        
        %%
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
        
        A = mexLassoWeighted(testImg, Dict, W, param);
        
        errList = ones(numel(dataSet),1)*Inf;
        for i = 1:numel(dataSet)
            a = find(DictLabel==i);
            err = testImg-Dict(:,a)*A(a,:);
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
    end
end

save( strcat('resultWeightedError2_portion', num2str(portion), '.mat'), ...
    'accTensor', 'Tlist', 'lambdaList', 'errorList', 'clusterLabel', 'labelTrue', 'labelpred');


% Y = round(labelTrue-1.5);
% errorListTMP = errorList;
% X = [errorListTMP; ones(1,size(errorList,2))];
% alpha = 1;
% W = (X*X'+alpha*eye(size(X,1)))\X*Y';
% 
% Yhat = W'*X;
% thr = 0.0;
% Yhat(Yhat<thr) = -1;
% Yhat(Yhat>=thr) = 1;
% mean(Y==Yhat)

end



%% learning weights to register the errors
addpath(genpath('../libsvm-3.20'));

labelTrue = [ones(1,193), ones(1,200)*2];

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
