clear
close all
clc

addpath(genpath('../ompbox10'));
addpath(genpath('../spams-matlab'));

%% load dataset
flagSelective = 1;

databaseDIR = '.\database_CNNfeature_CanonicalShape'; 

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
if exist('result_CNNfeature_canonicalShape.mat', 'file')
    load('result_CNNfeature_canonicalShape.mat');
else
Tlist = 125:5:125;
lambdaList = 0.0001;%[0.0001, 0.0005, 0.001];
accTensor = zeros(length(Tlist), length(lambdaList), length(layerIDall), 3);
errorList = [];

for i_T = 1:length(Tlist)
    T = Tlist(i_T);
for i_lambda = 1:length(lambdaList)
    lambda = lambdaList(i_lambda);
for LayerID = 7:7%1:length(layerIDall) %
    LAYERID = layerIDall(LayerID);
    
    dataSetName = strcat('PollenMatchByCNN_CanonicalShape_layer_', num2str(LAYERID), '_Dataset.mat');
    fprintf('CNNfeature at layer-%d...\n', LAYERID);
    
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
        for i = 1:numel(categoryList)            
            fprintf('\ncategory-%s...\n', categoryList(i).name);
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
%         save(dataSetName, 'dataSet', 'dataSetName', 'categoryList');
        fprintf('\nDone!\n');
    end
    
    %% get the training set of modern pollens
    clusterIDX = unique(dataSet{1}.clusterID);
    cluserK = length(clusterIDX);
    
    trainSet = cell(numel(dataSet), cluserK); % training set is represented by cells, of size  #category-by-#ShapeClusters
    trainSetLoc = cell(numel(dataSet), cluserK); % location of each patch
    
    numTrain = 100;    
    trID = cell(1,2);
    teID = cell(1,2);
    
    for i_class = 1:size(trainSet,1)
        a = randperm(numel(dataSet{i_class}.data));
        trID{i_class} = a(1:numTrain);
        teID{i_class} = a(numTrain+1:end);
        for ii_img = 1:numTrain % numTrain+1:numel(dataSet{i_class}.data)
            i_img = trID{i_class}(ii_img);
%             i_img = ii_img;
            
            i_cluster = dataSet{i_class}.clusterID(i_img);
            imFeat = dataSet{i_class}.data{i_img}.patchFeat(1:end-2,:);
            tmp = sqrt(sum(imFeat.^2,1));
            tmpIdx = find(tmp>1);
            imFeat(:,tmpIdx) = imFeat(:,tmpIdx) ./ repmat( tmp(tmpIdx), size(imFeat,1), 1 );
            trainSet{i_class,i_cluster} = [trainSet{i_class,i_cluster}, imFeat];%dataSet{1}.data{i}.patchFeat];
            
            patchLoc = dataSet{i_class}.data{i_img}.patchFeat(end-1:end,:);
            patchLoc = patchLoc-1;
            imSize = dataSet{i_class}.data{i_img}.imSize;
            feaSize = dataSet{i_class}.data{i_img}.feaSize(1:2);
            patchLoc = bsxfun(@rdivide, patchLoc, feaSize(:));
            patchLoc = bsxfun(@times, patchLoc, imSize(:));
            patchLoc = patchLoc + 1;
            patchLoc = bsxfun(@minus, patchLoc, imSize(:)/2);
            trainSetLoc{i_class,i_cluster} = [trainSetLoc{i_class,i_cluster}, patchLoc]; % dataSet{1}.data{i}.patchFeat];
        end
    end    
    
    %{
    testSet = cell(numel(dataSet), cluserK); % testing set is represented by cells, of size  #category-by-#ShapeClusters
    testSetLoc = cell(numel(dataSet), cluserK); % location of each patch of testing images
    for i_class = 1:size(testSet,1)
        for i_img = numTrain+1:numel(dataSet{i_class}.data)
            i_cluster = dataSet{i_class}.clusterID(i_img);
            imFeat = dataSet{i_class}.data{i_img}.patchFeat(1:end-2,:);
            tmp = sqrt(sum(imFeat.^2,1));
            tmpIdx = find(tmp>1);
            imFeat(:,tmpIdx) = imFeat ./ repmat( tmp(tmpIdx), size(imFeat,1), 1 );
            testSet{i_class,i_cluster} = [testSet{i_class,i_cluster}, imFeat];%dataSet{1}.data{i}.patchFeat];
            
            patchLoc = dataSet{i_class}.data{i_img}.patchFeat(end-1:end,:);
            patchLoc = patchLoc-1;
            imSize = dataSet{i_class}.data{i_img}.imSize;
            feaSize = dataSet{i_class}.data{i_img}.feaSize(1:2);
            patchLoc = bsxfun(@rdivide, patchLoc, feaSize(:));
            patchLoc = bsxfun(@times, patchLoc, imSize(:));
            patchLoc = patchLoc + 1;
            patchLoc = bsxfun(@minus, patchLoc, imSize(:)/2);
            testSetLoc{i_class,i_cluster} = [testSetLoc{i_class,i_cluster}, patchLoc]; % dataSet{1}.data{i}.patchFeat];
        end
    end 
    %}
    
    %% matching for classification
    labelpred = [];
    labelTrue = [];
    
    for i_class = 1:numel(dataSet)
        fprintf('\n\tcategory-%d testing... \n', i_class);
        for ii_img = numTrain+1:numel(dataSet{i_class}.data) %  1:numTrain %             
            i_img = teID{i_class}(ii_img-numTrain);            
%             i_img = ii_img;
            
            fprintf('\timage-%d --- ', i_img);
            %the cluster label of this testing image
            i_cluster = dataSet{i_class}.clusterID(i_img);
            
            %the patches represented by CNN features for this testing image
            imFeat = dataSet{i_class}.data{i_img}.patchFeat(1:end-2,:);
            tmp = sqrt(sum(imFeat.^2,1));
            tmpIdx = find(tmp>1);
            imFeat(:,tmpIdx) = imFeat(:,tmpIdx) ./ repmat( tmp(tmpIdx), size(imFeat,1), 1 );
            
            %the patches locations for this testing image
            patchLoc = dataSet{i_class}.data{i_img}.patchFeat(end-1:end,:);
            patchLoc = patchLoc-1;
            imSize = dataSet{i_class}.data{i_img}.imSize;
            feaSize = dataSet{i_class}.data{i_img}.feaSize(1:2);
            patchLoc = bsxfun(@rdivide, patchLoc, feaSize(:));
            patchLoc = bsxfun(@times, patchLoc, imSize(:));
            patchLoc = patchLoc + 1;
            patchLoc = bsxfun(@minus, patchLoc, imSize(:)/2);
            
            %% weighted lasso for locality preserving
            Dict = trainSet(:, i_cluster);
            DictLabel = [];
            for i = 1:numel(Dict)
                DictLabel = [DictLabel, i*ones(1,size(Dict{i},2))];
            end
            Dict = cell2mat(Dict');
            
            Dloc = trainSetLoc(:, i_cluster);
            Dloc = cell2mat(Dloc');
            
            W = zeros(size(Dict,2), size(patchLoc,2));
            for i = 1 : size(patchLoc,2)
                a = bsxfun(@minus, Dloc, patchLoc(:,i) );
                a = sum(a.^2); a = sqrt(a(:));
                %a = sum(abs(a));
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
            
            labelTrue = [labelTrue, i_class];
            labelpred = [labelpred, idxMIN];
            
            if labelpred(end) == labelTrue(end)
                fprintf('correct\n');
            else
                fprintf('wrong\n');
            end
        end
    end 
    acc = mean(labelTrue == labelpred);
    fprintf('\naccuracy=%.4f (lambda=%.4f, T=%d)\n\n', acc, lambda, T);     
    accTensor(i_T, i_lambda, LayerID, end) = acc;
    for i = 1:numel(dataSet)
        a = find(labelTrue==i);        
        accTensor(i_T, i_lambda, LayerID, i) = mean(labelTrue(a) == labelpred(a));
    end    
end

%}
end
end
%save('result_CNNfeature_canonicalShape2.mat', 'accTensor', 'layerIDall', 'Tlist', 'lambdaList');
end

%% visualization of performance w.r.t paramter settings
close all;
%accTensor = zeros(length(Tlist), length(lambdaList), length(layerIDall), 3);
% accTensor2 = accTensor(:,:,layerIDall,3);
accTensor2 = accTensor;%(:,:,layerIDall,3);


figure('name', 'acc vs. T and lambda');
for i_layer = 1:length(layerIDall)
    subplot(2,4,i_layer);
    layerID = layerIDall(i_layer);
    A = accTensor2(:, :, i_layer, end);
    surf(A);
    % legend('glauca', 'mariana', 'mean');
    xlabel('\lambda');
    ylabel('#nonzero');
    zlabel('acc');
    title(strcat('acc vs. \lambda and #nonzero SC at layer-', num2str(layerID)));
    
    ax = gca;
    ax.XTick = 1:numel(lambdaList);
    ax.YTick = 1:numel(Tlist);
    ax.XTickLabel = num2cell(lambdaList);
    ax.YTickLabel = num2cell(Tlist);    
end


figure('name', 'view acc vs. lambda'); % view acc vs. lambda
for i_layer = 1:length(layerIDall)
    subplot(2,4,i_layer);
    layerID = layerIDall(i_layer);
    A = accTensor2(:, :, i_layer, end);
    surf(A);
    % legend('glauca', 'mariana', 'mean');
    xlabel('\lambda');
    ylabel('#nonzero');
    zlabel('acc');
    title(strcat('acc vs. \lambda and #nonzero SC at layer-', num2str(layerID)));
    view(0, 0);
    
    ax = gca;
    ax.XTick = 1:numel(lambdaList);
    ax.YTick = 1:numel(Tlist);
    ax.XTickLabel = num2cell(lambdaList);
    ax.YTickLabel = num2cell(Tlist);    
end



figure('name', 'view acc vs. T');
for i_layer = 1:length(layerIDall)
    subplot(2,4,i_layer);
    layerID = layerIDall(i_layer);
    A = accTensor2(:, :, i_layer, end);
    surf(A);
    % legend('glauca', 'mariana', 'mean');
    xlabel('\lambda');
    ylabel('#nonzero');
    zlabel('acc');
    title(strcat('acc vs. \lambda and #nonzero SC at layer-', num2str(layerID)));
    view(90, 0);
    
    ax = gca;
    ax.XTick = 1:numel(lambdaList);
    ax.YTick = 1:numel(Tlist);
    ax.XTickLabel = num2cell(lambdaList);
    ax.YTickLabel = num2cell(Tlist);    
end


% figureName = strcat('.\result\weightedLASSO_CanonicalShape', ...
%                     '_layerID', num2str(layerID), '_lambda', num2str(lambda), '_T', num2str(T), '.pdf');
% print(1, figureName, '-dpdf') ;


