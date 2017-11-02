clear
close all
clc

%% fetch data
dirDataset = '../database';
dirDest = './database_2Dshape';
% categoryList = dir( strcat(dirDataset,'\* fossil'));
categoryList = dir( strcat(dirDataset));
categoryList = categoryList(3:end);


reSZ = [40, 40];
thresList = [0.35, 0.2, 0.35, 0.2, 0.35];
hsize = 7;
sigma = 2;
GauF = fspecial('gaussian', hsize, sigma);

for categID = 1:1 % numel(categoryList)
    thres = thresList(categID);
    
    if ~isdir( fullfile(dirDest, categoryList(categID).name) )
        mkdir( fullfile(dirDest, categoryList(categID).name) );
    end
    
    imList = dir( fullfile(dirDataset, categoryList(categID).name , '*.jpg') );
    for imID = 1:numel(imList)
        im = imread( fullfile(dirDataset, categoryList(categID).name, imList(imID).name) );
        
        im = double(im)/255;
        imBlur = imfilter(im, GauF, 'replicate', 'same', 'conv');
        
        imBlurBinary = imBlur;
        %imBlurBinary(imBlurBinary > thres) = 1;
        %imBlurBinary(imBlurBinary <= thres) = 0;
        %subplot(3,3,5); imshow(imBlurBinary); title('image-blur-binary');
        
        
        imBlurBinary = makeImSquareByPadding(imBlurBinary);
        imBlurBinarySmall = imresize(imBlurBinary, reSZ, 'nearest');
        %subplot(3,3,6); imshow(imBlurBinarySmall);         title('image-blur-binary-small');
        
        imBlurBinarySmallMask = imBlurBinarySmall;
        
        
        
        [junk, fileName, junkExt] = fileparts(imList(imID).name);
        imDestName = fullfile(dirDest, categoryList(categID).name, strcat(fileName, '.bmp') );
        imwrite( imBlurBinarySmallMask, imDestName );
        
        fprintf('%s\n', imList(imID).name);        
    end
end



















