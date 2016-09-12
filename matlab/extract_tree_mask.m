function [ tree_mask ] = extract_tree_mask( raw_path, label_path )
%EXTRACT_TREE_MASK Summary of this function goes here
%   Detailed explanation goes here
scene = imread(raw_path);
[h, w, ~] = size(scene);

treeId = 21;
mask = (imread(label_path) == treeId);
masked_scene = scene .* repmat(uint8(mask), [1, 1, 3]);

%%

R = scene(:, :, 1);
G = scene(:, :, 2);
B = scene(:, :, 3);

rgbs = double([R(mask), G(mask), B(mask)]);
rgb_mean = mean(rgbs, 1);
rgb_var = [std(rgbs(:, 1)), std(rgbs(:, 2)), std(rgbs(:, 3))];

%%
rgbs = double(reshape(masked_scene, [h*w, 3]));
tree_mask = ...
    rgbs(:, 1) - rgb_mean(1) < rgb_var(1) * 1 & ...
    rgbs(:, 2) - rgb_mean(2) < rgb_var(2) * 1 & ...
    rgbs(:, 3) - rgb_mean(3) < rgb_var(3) * 1 & ...
    reshape(mask, [h*w, 1]);

tree_mask = reshape(tree_mask, [h, w]);

end