%
% Shu Kong
% 05/03/2015

clear
close all
clc;

%% obtain and preprocess an image
dirDataset_feature = '.\database_CNNfeature_CanonicalShape';
dirDataset_img = '.\database_canonicalShape';

layerIdx = 24;

im2NameList = {'P glauca B1500 Pos 5.tif_Files', 'P glauca B1503 Pos 104', 'P glauca B1500 Pos 202.tif_Files', 'P glauca B1503 Pos 120'};
im2ID = 93; % 11, 51, 71

imList = dir( fullfile(dirDataset_img, 'mariana - modern', '*.jpg') ); % glauca mariana
imname = imList(im2ID).name;
im = imread( fullfile(dirDataset_img, 'mariana - modern', imname )  );
[i, NAME, EXT] = fileparts( imname );
imFea = load( fullfile(dirDataset_feature, strcat('layer_', num2str(layerIdx), '_mariana - modern'), strcat(NAME,'.mat') )  );

%% shape
patchLoc = imFea.patchFeat(end-1:end,:);

patchLoc = patchLoc-1;
imSize = imFea.imSize;
feaSize = imFea.feaSize(1:2);
patchLoc = bsxfun(@rdivide, patchLoc, feaSize(:));
patchLoc = bsxfun(@times, patchLoc, imSize(:));
patchLoc = patchLoc+1;
patchSize = 52;

imshow(im); 
for i = 1:size(patchLoc,2)
    line([patchLoc(2,i)-round(patchSize/2), patchLoc(2,i)+round(patchSize/2)-1, patchLoc(2,i)+round(patchSize/2)-1, patchLoc(2,i)-round(patchSize/2), patchLoc(2,i)-round(patchSize/2)],...
        [patchLoc(1,i)-round(patchSize/2), patchLoc(1,i)-round(patchSize/2), patchLoc(1,i)+round(patchSize/2)-1, patchLoc(1,i)+round(patchSize/2)-1, patchLoc(1,i)-round(patchSize/2)], ...
        'linewidth', 1, 'color', 'm' )
end



