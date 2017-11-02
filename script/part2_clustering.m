clear
close all
clc

%% fetch data of modern pollen grains
dirDataset = './database_2Dshape';
dirDataset_orgImg = '../database';
categoryList = dir( strcat(dirDataset,'/* modern'));

% dataMat = [];
label = [];
imList = {};
fprintf('fetch data...\n');
for categID = 1:numel(categoryList)
    fprintf('\t%s...\n', categoryList(categID).name);
    imList{end+1} = dir( fullfile(dirDataset, categoryList(categID).name, '*.bmp') );
    for imID = 1:numel(imList{end})        
        im = imread( fullfile(dirDataset, categoryList(categID).name, imList{end}(imID).name) );        
        im = double(im)/255;
        
        label(end+1) = categID;
%         dataMat = [dataMat, im(:)];
    end
end
fprintf('\tdone!\n');
% dataMat = reshape(dataMat, [size(im,1), size(im,2), size(dataMat,2)] );
imageNameList = [imList{1};imList{2}];

%% calculate the affinity matrix between every pair of images
fprintf('calculate/load the affinity matrix between every pair of images...\n');
if ~exist('affinityMatrix.mat', 'file')    
    affMat = zeros(size(dataMat,3));
    bestThyI2J = zeros(size(dataMat,3));
    for i = 1:size(dataMat,3)
        fprintf('\timage-%d is being matched to all others...\n', i);
        for j = i+1:size(dataMat,3)
            [mindist, thy, im3new]= minDistRot2D(dataMat(:,:,i), dataMat(:,:,j));
            bestThyI2J(j,i) = thy; % rotate image-j by thy to best match image-i
            affMat(i,j) = mindist;
        end
    end
    fprintf('\ndone!\n');    
    affMat = affMat + affMat';    
    save('affinityMatrix.mat', 'affMat', 'dataMat', 'bestThyI2J');
else
    load('affinityMatrix.mat');
end

%% clustering
K = 2;
if exist('mediods.mat', 'file')
    load('mediods.mat');
    
    C = reshape(C3D, [size(C3D,1)*size(C3D,2), size(C3D,3)]);
    Cdisp = showdict(C, [size(im,1), size(im,2)], ceil(sqrt(K)), ceil(sqrt(K)));    
    figure; imshow(Cdisp); title('2D shape centroids');
    
    figure;
    rNUM = ceil(sqrt(K));
    cNUM = ceil(K/rNUM);
    for i = 1:K
        subplot(rNUM, cNUM, i);
        imshow(ImgMedoid{i});
        title(num2str(i));
    end
else
    
    %% k-medoid on whole set
    [medoidIdx, cidx] = kmedioids(affMat, K);
    
    C3D = dataMat(:,:,cidx);
    C = reshape(C3D, [size(C3D,1)*size(C3D,2), size(C3D,3)]);
    Cdisp = showdict(C, [size(im,1), size(im,2)], ceil(sqrt(K)), ceil(sqrt(K)));
    
    figure; imshow(Cdisp); title('2D shape centroids');
    countK = zeros(K, 1);
    fprintf('\nstatistics for training set\n');
    for i = 1:K
        countK(i) = length(find(medoidIdx==i));
    end
    disp(countK);
    
    %% first pass to get the medoids of original images
    fprintf('\nfirst pass to get the medoids of original images...');
    
    ImgMedoid = cell(K,1);
    for i = 1:K
        idx = cidx(i);
        [junk, NAME, EXT] = fileparts(imageNameList(idx).name);
        im = imread(  fullfile(dirDataset_orgImg , categoryList( label(idx) ).name, strcat(NAME,'.jpg')) );
        ImgMedoid{i} = im;
    end
    
    % visualize the medoids of pollen images
    figure;
    rNUM = ceil(sqrt(K));
    cNUM = ceil(K/rNUM);
    for i = 1:K
        subplot(rNUM, cNUM, i);
        imshow(ImgMedoid{i});
        title(num2str(i));
    end
    save('mediods.mat', 'ImgMedoid', 'cidx', 'C3D', 'C');
end

%% fetch data of fossil pollen grains
dirDataset = './database_2Dshape';
dirDataset_orgImg = '../database';
categoryList = dir( strcat(dirDataset,'/* fossil'));

dataMat = [];
label = [];
imList = {};
fprintf('fetch fossil pollen grain images...\n');
for categID = 1:numel(categoryList)-2
    fprintf('\t%s...\n', categoryList(categID).name);
    imList{end+1} = dir( fullfile(dirDataset, categoryList(categID).name, '*.bmp') );
    for imID = 1:numel(imList{end})        
        im = imread( fullfile(dirDataset, categoryList(categID).name, imList{end}(imID).name) );        
        im = double(im)/255;
        
        label(end+1) = categID;
        dataMat = [dataMat, im(:)];
    end
end
fprintf('\tdone!\n');
dataMat = reshape(dataMat, [size(im,1), size(im,2), size(dataMat,2)] );
%imageNameList = [imList{1};imList{2}];
imageNameList = [imList{1}];

%% save fossil pollen grain clustering results
fprintf('\nsaving for visualization...\n');
dirDataSave = './database_fossil_thumbnail_visualize';
dirDataSaveCanonicalShape = './database_fossil_canonicalShape';
testMedoidIdx = zeros(1,numel(imageNameList));
for i = 1:numel(imageNameList)
    fprintf('image-%d/%d...\n', i,numel(imageNameList));
    [junk, NAME, EXT] = fileparts(imageNameList(i).name);
    im = imread(  fullfile(dirDataset_orgImg, categoryList( label(i) ).name, strcat(NAME,'.jpg')) );
    
    disList = -1*ones(1,K);
    bestThyI2J = zeros(1,K);
    for k = 1:K
        [mindist, thy, im3new]= minDistRot2D(C3D(:,:,k), dataMat(:,:,i));
        bestThyI2J(k) = thy; % rotate image-j by thy to best match image-i
        disList(k) = mindist;
    end
    [valjunk, idjunk] = min(disList);
    testMedoidIdx(i) = idjunk;
    im = makeImSquareByPadding(double(im)/255);
    im = imrotate(im, thy*180/pi);
    
    if ~isdir( fullfile(dirDataSave, categoryList(label(i)).name) )
        mkdir(fullfile(dirDataSave, categoryList(label(i)).name));
    end
    if ~isdir( fullfile(dirDataSave, categoryList(label(i)).name, strcat('cluster_', num2str(testMedoidIdx(i)))) )
        mkdir(fullfile(dirDataSave, categoryList(label(i)).name, strcat('cluster_', num2str(testMedoidIdx(i)))));
    end
    
    imwrite( im, fullfile(dirDataSave, categoryList(label(i)).name, strcat('cluster_', num2str(testMedoidIdx(i))), imageNameList(i).name) );
    
    
    if ~isdir( fullfile(dirDataSaveCanonicalShape, categoryList(label(i)).name) )
        mkdir(fullfile(dirDataSaveCanonicalShape, categoryList(label(i)).name) );
    end
    
    [PATHSTR,NAME,EXT] = fileparts(imageNameList(i).name);
    imwrite( im, fullfile(dirDataSaveCanonicalShape, categoryList(label(i)).name, ...
        strcat(NAME, '_K', num2str(testMedoidIdx(i)), '.jpg')   ) );
end

% 
% 
% fprintf('\nsaving for visualization...\n');
% dirDataSave = '.\database_2Dshape_visualize_LOOV';
% dirDataSaveCanonicalShape = '.\database_canonicalShape_LOOV';
% 
% if isdir( dirDataSave )
%     rmdir( dirDataSave, 's')
% end
% mkdir( dirDataSave );
% 
% if isdir( dirDataSaveCanonicalShape )
%     rmdir( dirDataSaveCanonicalShape, 's')
% end
% mkdir( dirDataSaveCanonicalShape );
% 
% 
% % 
% for i = 1:numel(imageNameList)    
%     [junk, NAME, EXT] = fileparts(imageNameList(i).name);
%     im = imread(  fullfile(dirDataset_orgImg , categoryList( label(i) ).name, strcat(NAME,'.jpg')) );
%     
%     im2 = dataMat(:, :, i);
%     im1 = C3D(:,:,medoidIdx(i));%ImgMedoid{inds(count)};
%     [mindist, thy, im3new]= minDistRot2D(im1, im2);
%     
%     im = makeImSquareByPadding(double(im)/255);
%     im = imrotate(im, thy*180/pi);
%     
%     if ~isdir( fullfile(dirDataSave, categoryList(label(i)).name) )
%         mkdir(fullfile(dirDataSave, categoryList(label(i)).name));
%     end
%     if ~isdir( fullfile(dirDataSave, categoryList(label(i)).name, strcat('cluster_', num2str(medoidIdx(i)))) )
%         mkdir(fullfile(dirDataSave, categoryList(label(i)).name, strcat('cluster_', num2str(medoidIdx(i)))));
%     end
%         
%     imwrite( im, fullfile(dirDataSave, categoryList(label(i)).name, strcat('cluster_', num2str(medoidIdx(i))), imageNameList(i).name) );
%     
%     
%     if ~isdir( fullfile(dirDataSaveCanonicalShape, categoryList(label(i)).name) )
%         mkdir(fullfile(dirDataSaveCanonicalShape, categoryList(label(i)).name) );
%     end
%     
%     [PATHSTR,NAME,EXT] = fileparts(imageNameList(i).name);
%     imwrite( im, fullfile(dirDataSaveCanonicalShape, categoryList(label(i)).name, ...
%         strcat(NAME, '_K', num2str(medoidIdx(i)), '.jpg')   ) );
% end



