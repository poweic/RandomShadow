%%
% scene = im2double(imread('test.png'));

scene = im2double(imread('../img_raw00058.jpg'));

tree_mask = extract_tree_mask('../raw.png', '../labels.png');

[H, W, ~] = size(scene);
%%
for i = 1:1
    shadow = gen_shadow(tree_mask, H, W);
    new_scene = scene .* repmat(shadow, [1,1,3]);
    imshow(new_scene);

%     imwrite(new_scene, sprintf('ex_%02d.jpg', i));
end