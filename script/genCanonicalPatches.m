function patchLoc = genCanonicalPatches(im, patchSize, patchNum, overlapBorder)


imSize = size(im);

thres = 0.1;
hsize = 7;
sigma = 2;
GauF = fspecial('gaussian', hsize, sigma);

im = double(im)/255;
imMask = imfilter(im, GauF, 'replicate', 'same', 'conv');
imMask(imMask > thres) = 1;
imMask(imMask <= thres) = 0;

SE = strel('rectangle',[7 7]);
imMask = imdilate(imMask, SE);

[gridY, gridX] = meshgrid( 1:patchSize-overlapBorder:imSize(1)-patchSize, 1:patchSize-overlapBorder:imSize(2)-patchSize );

%%
patchLoc = [];
for i = 1:size(gridX,1)
    for j = 1:size(gridX,2)        
        if mean(mean(imMask(gridY(i,j):gridY(i,j)+patchSize-1, gridX(i,j):gridX(i,j)+patchSize-1))) > 0.8
            patchLoc = [patchLoc, [gridY(i,j)+round(patchSize/2);gridX(i,j)+round(patchSize/2)]];
        end
    end
end

idx = randperm( size(patchLoc,2) );
if size(patchLoc,2) < patchNum
    idx = 1:size(patchLoc,2);
else
    idx = idx(1:patchNum);
end
patchLoc = patchLoc(:,idx );




%% 
% imshow(im); title( strcat(num2str(patchNum), ' patches') );
% for i = 1:size(patchLoc,2)
%     line([patchLoc(2,i)-round(patchSize/2), patchLoc(2,i)+round(patchSize/2)-1, patchLoc(2,i)+round(patchSize/2)-1, patchLoc(2,i)-round(patchSize/2), patchLoc(2,i)-round(patchSize/2)],...
%         [patchLoc(1,i)-round(patchSize/2), patchLoc(1,i)-round(patchSize/2), patchLoc(1,i)+round(patchSize/2)-1, patchLoc(1,i)+round(patchSize/2)-1, patchLoc(1,i)-round(patchSize/2)], ...
%         'linewidth', 1, 'color', 'm' )
% end


