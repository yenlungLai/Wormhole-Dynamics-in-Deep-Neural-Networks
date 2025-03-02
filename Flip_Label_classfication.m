clearvars; close all; clc;

% Load MNIST dataset
for digit = 0:9
    load(sprintf("data/%d.mat", digit));
end

img = zero(:,:,1) ./ 255;
vec1 = m2vec(img);
u = vec1;

output_folder_flip = 'output_images_flip';
output_folder = 'output_images';

% Create output folders if they do not exist
if ~exist(output_folder_flip, 'dir')
    mkdir(output_folder_flip);
end
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

% Define L values for testing
L_values = 1:1:4;
n = 8000;

% Create figure for subplots
figure;
tiledlayout(1,4); % 2 rows, 3 columns

% Iterate over L values
for idx = 1:length(L_values)
    L = L_values(idx);
    gen = [];
    gen2 = [];
    
    % Train model for given L
    [mat, g1] = main_alg(u, n, L);
    
    for i = 2:3000
        % Unflipped and flipped test images
        img2 = zero(:,:,i) ./ 255;
        vec2 = m2vec(img2);
        
        img2inv = -zero(:,:,i) ./ 255;
        vec2inv = m2vec(img2inv);
        
        % Compute distances
        g2 = mat * vec2;
        g2inv = mat * vec2inv;
        dis1 = acos(dot(g1, g2) / (norm(g1) * norm(g2))) / pi;
        dis2 = acos(dot(g1, g2inv) / (norm(g1) * norm(g2inv))) / pi;
        
        gen = [gen; dis1];
        gen2 = [gen2; dis2];
    end

    % Plot score distributions in subplot
    plot_score_distributions2(gen, gen2, idx);
end

% Overall title
% sgtitle('Score Distributions for Different L Values');





























function [vec] =m2vec(m)
vec=reshape(m,size(m,1)*size(m,2),1);
end




function [mat]=vec2m(m)
mat=reshape(m,28,28);
end


function [yfil,frmat]=Encoding_mat(x,n,t)
k=size(x,1);

rmat=randn(n,k);
% rmat=orth(rmat);
y=rmat*x;
absy=(y);
[sorted_data, sortedindex ]= sort(absy, 'ascend');
topindex=(sortedindex(1:t));
frmat=rmat(topindex,:);
yfil=y(topindex);

end




function [RandomfieldM,yfil] =main_alg(Ib,n,L)

xpos=[];  % bob first generate RandomfieldM= WL WL-1 ... LW1
ii=1; inI=Ib(:);
RandomfieldM=eye(length(inI)); 
while ii<=L
    [yfil,frmat]=Encoding_mat(inI,n,length(inI));

    %     yfil=yfil/norm(yfil);
    inI=yfil;
    RandomfieldM=frmat*RandomfieldM;
 
    ii=ii+1;
end
end