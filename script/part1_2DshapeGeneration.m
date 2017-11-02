clear
close all
clc

%%
dirDataset = '../database';
dirDest = './database_2Dshape';
% categoryList = dir( strcat(dirDataset,'\* fossil'));
categoryList = dir( strcat(dirDataset));
categoryList = categoryList(3:end);

reSZ = [40, 40];
thresList = [0.2, 0.2, 0.35];
hsize = 7;
sigma = 2;
GauF = fspecial('gaussian', hsize, sigma);

for categID = 1:1%numel(categoryList)
    thres = thresList(categID);
    
    if ~isdir( fullfile(dirDest, categoryList(categID).name) )
        mkdir( fullfile(dirDest, categoryList(categID).name) );
    end
    
    imList = dir( fullfile(dirDataset, categoryList(categID).name , '*.jpg') );
    for imID = 1:numel(imList)
        im = imread( fullfile(dirDataset, categoryList(categID).name, imList(imID).name) );
        
        im = double(im)/255;
        %subplot(3,3,1); imshow(im); title('original image');
        
        %         imBinary = im;
        %         imBinary(imBinary > thres) = 1;
        %         imBinary(imBinary <= thres) = 0;
        %         %subplot(3,3,2); imshow(imBinary); title('image-binary');
        %
        %
        %         imBinaryBlur = imfilter(imBinary, GauF, 'replicate', 'same', 'conv');
        %         %subplot(3,3,3); imshow(imBinaryBlur); title('image-binary-blur');
        
        imBlur = imfilter(im, GauF, 'replicate', 'same', 'conv');
        %subplot(3,3,4); imshow(imBlur); title('image-blur');
        
        imBlurBinary = imBlur;
        imBlurBinary(imBlurBinary > thres) = 1;
        imBlurBinary(imBlurBinary <= thres) = 0;
        %subplot(3,3,5); imshow(imBlurBinary); title('image-blur-binary');
        
        
        imBlurBinary = makeImSquareByPadding(imBlurBinary);
        imBlurBinarySmall = imresize(imBlurBinary, reSZ, 'nearest');
        %subplot(3,3,6); imshow(imBlurBinarySmall);         title('image-blur-binary-small');
        
        imBlurBinarySmallMask = imBlurBinarySmall;
        imBlurBinarySmallMaskcomplement = 1-imBlurBinarySmallMask;
        
        if categID == 2
            SE = strel('rectangle',[3 3]);
            BW1 = imdilate(imBlurBinarySmallMask, SE);
            BW2 = imerode(BW1,SE);
            imBlurBinarySmallMask = BW2;
            
            SE = strel('rectangle',[3 3]);
            BW1 = imdilate(imBlurBinarySmallMask, SE);
            BW2 = imerode(BW1,SE);
            imBlurBinarySmallMask = BW2;
            
            SE = strel('rectangle',[3 3]);
            BW1 = imdilate(imBlurBinarySmallMask, SE);
            BW2 = imerode(BW1,SE);
            imBlurBinarySmallMask = BW2;
        end
        
        cntCMPO = bwconncomp(imBlurBinarySmallMaskcomplement, 4);
        CMPOholeList = zeros(numel(cntCMPO.PixelIdxList),1);
        for i = 1:length(cntCMPO.PixelIdxList)
            if isempty(find(cntCMPO.PixelIdxList{i}==1, 1)) &&...
                    isempty(find(cntCMPO.PixelIdxList{i}==size(imBlurBinarySmallMask,1), 1)) && ...
                    isempty(find(cntCMPO.PixelIdxList{i}==numel(imBlurBinarySmallMask(:,1:end-1))+1, 1)) && ...
                    isempty(find(cntCMPO.PixelIdxList{i}==numel(imBlurBinarySmallMask), 1))
                
                CMPOholeList(i) = 1;
                imBlurBinarySmallMask(cntCMPO.PixelIdxList{i}) = 1;
            end
        end
        %subplot(3,3,9); imshow(imBlurBinarySmallMask); title('pollen shape mask');
        
        if categID == 2
            SE = strel('rectangle',[3 3]);
            BW1 = imdilate(imBlurBinarySmallMask, SE);
            BW2 = imerode(BW1,SE);
            imBlurBinarySmallMask = BW2;
            
            SE = strel('rectangle',[3 3]);
            BW1 = imdilate(imBlurBinarySmallMask, SE);
            BW2 = imerode(BW1,SE);
            imBlurBinarySmallMask = BW2;
            
            SE = strel('rectangle',[3 3]);
            BW1 = imdilate(imBlurBinarySmallMask, SE);
            BW2 = imerode(BW1,SE);
            imBlurBinarySmallMask = BW2;
            
            SE = strel('rectangle',[3 3]);
            BW1 = imdilate(imBlurBinarySmallMask, SE);
            BW2 = imerode(BW1,SE);
            imBlurBinarySmallMask = BW2;
        end
        
        
        [junk, fileName, junkExt] = fileparts(imList(imID).name);
        imDestName = fullfile(dirDest, categoryList(categID).name, strcat(fileName, '.bmp') );
        imwrite( imBlurBinarySmallMask, imDestName );
        
        fprintf('%s\n', imList(imID).name);
    end
end



















