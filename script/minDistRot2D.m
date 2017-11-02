function [mindist, thy, im3new]= minDistRot2D(im1, im2, rotList)
% Find the distance between two object centering at images. The objects are
% supposed to be arbitrarily rotated at the center. So how to calculate
% their distance becomes a problem. This function tries rotation bins and
% flip one image to match the other one. Then their distance is obtained as
% their minimum distance among all possible rotations and flipping.
%
% input
%       im1             -- image 1
%       im2             -- image 2
%       rotList         -- rotation bins
%
% output
%       mindist         -- minimum distance
%       thy             -- the rotation bin/angle for image 2 that best match image 1
%       im3new          -- the rotated image 2 to best match image 1
%
%
%   see also: fillObjectHoles
%
% Shu Kong
% 04/26/2015

if ~exist('rotList', 'var')
    rotList = [0:10:359] *pi/180 ; % pre-defined rotation bins
end

%% fix image 1
diagLength = 4+ceil(sqrt(sum(size(im1).^2)));

im2backup = im2;

% when rotating an image, it becomes larger due to its boundary, therefore
% padding with zero values
im1new = zeros(diagLength);
cIm1New = round(size(im1new)/2);
im1new( 1+cIm1New(1)-floor(size(im1,1)/2):cIm1New(1)+ceil(size(im1,1)/2), 1+cIm1New(2)-floor(size(im1,2)/2):cIm1New(2)+ceil(size(im1,2)/2)) = im1;

[idx1, idx2] = ind2sub( size(im2), 1:numel(im2));
X = [idx1; idx2];

%% try fixed possible rotations
distList = zeros(2*length(rotList),1);
count = 1;
for rotId = 1:length(rotList)
    thy = rotList(rotId);
    %{
    RotMat = [  cos(thy)   -sin(thy) ; sin(thy)   cos(thy) ]; % rotation matrix
    
    Y = RotMat*X; % rotate the image
    
    Z = round(bsxfun(@minus, Y, min(Y,[],2))+1);
    im3 = zeros( max(Z(1,:)), max(Z(2,:)) );
    z = sub2ind(size(im3), Z(1,:), Z(2,:));
    im3(z) = im2(:);
    
    im3 = fillObjectHoles(im3); % fill the holes in the object
    %}
    im3 = imrotate(im2, thy*180/pi, 'nearest');
    im3new = zeros(diagLength);
    cIm3New = round(size(im3new)/2);
    im3new( 1+cIm3New(1)-floor(size(im3,1)/2):cIm3New(1)+ceil(size(im3,1)/2), 1+cIm3New(2)-floor(size(im3,2)/2):cIm3New(2)+ceil(size(im3,2)/2)) = im3;
    distList(count) = norm(im1new-im3new, 'fro');
    count = count + 1;
end

%% flip left-right and match again
im2flip = fliplr(im2);

for rotId = 1:length(rotList)
    thy = rotList(rotId);
    %{
    RotMat = [  cos(thy)   -sin(thy) ; sin(thy)   cos(thy) ];
    
    Y = RotMat*X;
    
    Z = round(bsxfun(@minus, Y, min(Y,[],2))+1);
    im3 = zeros( max(Z(1,:)), max(Z(2,:)) );
    z = sub2ind(size(im3), Z(1,:), Z(2,:));
    im3(z) = im2flip(:);
    
    im3 = fillObjectHoles(im3);
    %}
    im3 = imrotate(im2, thy*180/pi, 'nearest');
    im3new = zeros(diagLength);
    cIm3New = round(size(im3new)/2);
    im3new( 1+cIm3New(1)-floor(size(im3,1)/2):cIm3New(1)+ceil(size(im3,1)/2), 1+cIm3New(2)-floor(size(im3,2)/2):cIm3New(2)+ceil(size(im3,2)/2)) = im3;
    distList(count) = norm(im1new-im3new, 'fro');
    count = count + 1;
end

%% find the minimum distance by considering rotation and flipping
[mindist, minIdx] = min(distList);
flipFlag = 0;
if minIdx > length(distList)/2
    minIdx = minIdx - length(distList)/2;
    flipFlag = 1;
end

thy = rotList(minIdx);
RotMat = [  cos(thy)   -sin(thy) ; sin(thy)   cos(thy) ];
%{
Y = RotMat*X;

Z = round(bsxfun(@minus, Y, min(Y,[],2))+1);
im3 = zeros( max(Z(1,:)), max(Z(2,:)) );
z = sub2ind(size(im3), Z(1,:), Z(2,:));
if flipFlag
    im3(z) = im2flip(:);
else
    im3(z) = im2(:);
end

im3 = fillObjectHoles(im3);
%}
im3 = imrotate(im2, rotList(minIdx)*180/pi, 'nearest');

im3new = zeros(diagLength);
cIm3New = round(size(im3new)/2);
im3new( 1+cIm3New(1)-floor(size(im3,1)/2):cIm3New(1)+ceil(size(im3,1)/2), 1+cIm3New(2)-floor(size(im3,2)/2):cIm3New(2)+ceil(size(im3,2)/2)) = im3;

