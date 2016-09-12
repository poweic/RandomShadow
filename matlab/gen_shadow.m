function [ shadow ] = gen_shadow( tree_mask, H, W )
%GEN_SHADOW Summary of this function goes here
%   Detailed explanation goes here

%% Random rotate
deg_range = 30;
deg = (90 + (rand - 0.5) * deg_range) * sign(rand - 0.5);
rotated_tree_mask = imrotate(tree_mask, deg);
rotated_tree_mask = uint8(repmat(rotated_tree_mask, [1,1,3])) * 255;

%% Rand crop with random size
[h, w, ~] = size(rotated_tree_mask);
crop_w = uint32(w * min(max(rand, 0.2), 0.8));
crop_h = uint32(h * min(max(rand, 0.2), 0.8));

left = uint32(rand * (w - crop_w) * 0.95);
top = uint32(rand * (h - crop_h) * 0.95);

x = [left, left + crop_w, left + crop_w, left];
y = [top, top, top + crop_h, top + crop_h];

cropped_tree = rotated_tree_mask(top:top+crop_h, left:left+crop_w);

%% Random light source
illumination = rand * 0.3 + 0.7;
light_distance = rand * 30 + 5;

shadow_mask = imresize(cropped_tree, [uint32(H * 0.6), W]);
shadow_mask = imgaussfilt(shadow_mask, light_distance) * illumination;
shadow = 1.0 - im2double(shadow_mask);
shadow = [ones(H - size(shadow_mask, 1), W); shadow];

% imshow(shadow_mask);