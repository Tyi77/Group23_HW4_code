data = load('output\Statue.mat');
% data = load('output\Mesona.mat');
P = data.points_3D;
% p_img2 = double(data.pts1);
[p_img2, unique_indices] = unique(double(data.pts1), 'rows', 'stable');
[P, unique_indices] = unique(P, 'rows', 'stable');
disp('Size of P:'); disp(size(P));
M = data.P2;
tex_name = 'data/Statue1.bmp'; % 紋理圖片名稱
% tex_name = 'data/Mesona1.JPG'; % 紋理圖片名稱
im_index = 2; % 圖片索引
output_dir = './output'; % 輸出目錄
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end
if size(p_img2, 1) ~= size(P, 1)
    error('p_img2 和 P 的行數不一致：p_img2 有 %d 行，而 P 有 %d 行。', size(p_img2, 1), size(P, 1));
end
obj_main(P, p_img2, M, tex_name, im_index, output_dir);
