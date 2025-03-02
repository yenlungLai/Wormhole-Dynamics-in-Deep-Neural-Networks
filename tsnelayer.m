clear all;

% This code is inheritly same concept applied to tsne_round2d and
% tsne_round3d, just for clean up purpurse, i have included Lfocus
% parameter to set our focus on particular Layer feature, for example
% H=G3G2G1, setting Lfocus=2 means we are observing the change of feagure at
% 1,2,3 layer after recursively apply H=G2G1 up to layer 2. This is because
% for too large Lfocus, the feature might collapse into singularity, tsne
% may not display properly the output distribution.


% Initialize parameters
n = 2^20; L = 3; k = 25; N = 300;
gens = [];
u = randn(1, k);
u = u / norm(u);  % Generate and normalize initial vector u
ii = 1;
inI = u(:);
w_ell = [];

% Generate output trained feature w for each layer
while ii <= L
    [y1, frmat] = Encoding_mat(inI, n, length(inI));  % Encoding step
    inI = y1;
    y1 = y1 / norm(y1);  % Normalize the result
    frmatii{ii} = frmat;  % Store weight matrix, G_ell
    w_ell = [w_ell; y1'];  % Store vectors w_ell
    ii = ii + 1;
end





vvect=randn(k,N);


for Ncount=1:N                % generate layer feature for each point N
    v=vvect(:,Ncount);
    wp_ell = [];
    for ell=1:L
        G_ell= frmatii{ell};
        wp_i=G_ell*v;
        wp_i=wp_i/norm(wp_i)';
        v=wp_i;
        wp_ell=[wp_ell;v'];
    end
    featureset{Ncount}=wp_ell;      % N point cell, each has L feature (Dim-K)
end







vv2=[];
for ll = 1:L
    vv = [];
    for jj = 1: N  % Adjust for all layers and N samples
        vvecjj = featureset{jj};
        vvecs = vvecjj(ll, :);
        vv = [vv; vvecs];
    end
    % vv2=[vv2;vv];
    vvL{ll} = vv ;  % Store vectors for t-SNE
end


for i = 1:L
    y{i} = vvL{i};  % Prepare data for each subplot for particular i-th layer feature
end




% Ensure the UMAP toolbox is added to the MATLAB path
addpath('C:\Users\xiaomi\AppData\Roaming\MathWorks\MATLAB Add-Ons\Collections\Uniform Manifold Approximation and Projection (UMAP)'); % Update with your path.

% Initialize parameters
cols = ceil(sqrt(L));
rows = ceil(L / cols);
markerSize = 20;
lineWidth = 0.1;
fontSize = 12;

% Create a figure for all subplots
figure('Position', [100, 100, 1000, 800]);

% Generate a custom colormap with sufficient distinct colors
num_colors = 2*i; % Define the number of unique colors
cmap1 =    [
    0.00, 0.45, 0.74;  % Blue
    0.85, 0.33, 0.10;  % Red
    0.93, 0.69, 0.13;  % Yellow
    0.49, 0.18, 0.56;  % Purple
    0.47, 0.67, 0.19;  % Green
    0.30, 0.75, 0.93;  % Cyan
    0.64, 0.08, 0.18;  % Dark Red
    0.00, 0.50, 0.00;  % Dark Green
    0.25, 0.25, 0.25;  % Gray
    0.10, 0.80, 0.40;  % Light Green
    0.80, 0.20, 0.80;  % Violet
    1.00, 0.50, 0.00;  % Orange
    0.50, 0.00, 0.50;  % Dark Purple
    0.00, 0.75, 0.75;  % Turquoise
    0.80, 0.80, 0.00;  % Olive
    0.80, 0.00, 0.80;  % Magenta
    0.00, 0.80, 0.20;  % Emerald
    ];


% Loop through each layer
for i = 1:L
    figure;
    for j = 1:i
        extra_dim = j * 10;
        combined_data = y{j};

        rng('shuffle');  % Ensure different randomness in every execution
        Y = tsne(combined_data, 'NumDimensions', 2);


        cluster_labels = kmeans(Y, 2);


        hold on;

        dex = 1;
        for c = unique(cluster_labels)'
            cluster_points = Y(cluster_labels == c, :);
            cluster_points_3D = [cluster_points repmat(extra_dim * dex, size(cluster_points, 1), 1)];


            % Randomly select a different color for each cluster
            random_color = cmap1(randperm(size(cmap1, 1), 1), :);
            scatter3(cluster_points_3D(:, 1), cluster_points_3D(:, 2), cluster_points_3D(:, 3), markerSize, ...
                'filled', 'MarkerFaceColor', random_color,'MarkerEdgeColor', 'k', 'LineWidth',0.5);  % Set black edge

            hold on;
        end

        % Highlight the Z-plane for this layer
        xlim_vals = xlim;
        ylim_vals = ylim;
        fill3([xlim_vals(1) xlim_vals(2) xlim_vals(2) xlim_vals(1)], ...
            [ylim_vals(1) ylim_vals(1) ylim_vals(2) ylim_vals(2)], ...
            repmat(extra_dim * dex, 1, 4), ...
            [0.8 0.8 0.8], 'FaceAlpha', 0.3, 'EdgeColor', 'none'); % Transparent plane

        hold off;
        xlabel('t-SNE D-1', 'FontSize', fontSize); xlim=[-30,30];
        ylabel('t-SNE D-2', 'FontSize', fontSize); ylim=[-30,30];
        zlabel('$\ell$', 'Interpreter', 'latex', 'FontSize', fontSize);
        grid on;
        set(gca, 'FontSize', fontSize, 'LineWidth', 1.2);
        axis equal;
        title(['$\ell$ = ' num2str(i)], 'Interpreter', 'latex');
        view(3);
    end
end



















%%%%%%%  Main function (MaxLikelihood Alg) %%%%%%




% Function to calculate Davies-Bouldin Index
function dbi_value = davies_bouldin_index(X, idx)
k = max(idx);
centroids = zeros(k, size(X, 2));
for i = 1:k
    centroids(i, :) = mean(X(idx == i, :), 1);
end

% Intra-cluster distances
intra_dist = zeros(k, 1);
for i = 1:k
    intra_dist(i) = mean(pdist(X(idx == i, :)));
end

% Inter-cluster distances
inter_dist = zeros(k, k);
for i = 1:k
    for j = i+1:k
        inter_dist(i, j) = norm(centroids(i, :) - centroids(j, :));
        inter_dist(j, i) = inter_dist(i, j);  % Symmetric
    end
end

% Davies-Bouldin index
dbi_value = 0;
for i = 1:k
    max_ratio = 0;
    for j = 1:k
        if i ~= j
            ratio = (intra_dist(i) + intra_dist(j)) / inter_dist(i, j);
            max_ratio = max(max_ratio, ratio);
        end
    end
    dbi_value = dbi_value + max_ratio;
end
dbi_value = dbi_value / k;
end

% Function to calculate inter-cluster distance
function inter_dist = inter_cluster_distance(X, idx)
k = max(idx);
centroids = zeros(k, size(X, 2));
for i = 1:k
    centroids(i, :) = mean(X(idx == i, :), 1);
end

% Calculate inter-cluster distances
inter_dist = 0;
for i = 1:k
    for j = i+1:k
        inter_dist = inter_dist + norm(centroids(i, :) - centroids(j, :));
    end
end
inter_dist = inter_dist / (k * (k - 1) / 2);  % Average inter-cluster distance
end

% Function to calculate intra-cluster distance
function intra_dist = intra_cluster_distance(X, idx)
k = max(idx);
intra_dist = 0;
for i = 1:k
    cluster_points = X(idx == i, :);
    centroid = mean(cluster_points, 1);
    intra_dist = intra_dist + mean(pdist(cluster_points));
end
intra_dist = intra_dist / k;  % Average intra-cluster distance
end


% sgtitle('3D t-SNE Visualization of x and Multiple y Sets'); % Overall title for the subplots

% Function definitions below remain unchanged
function read_and_save_csv_first_column(filename)
% Read the file into a table
T = readtable(filename, 'ReadVariableNames', false);

% Extract the first column
T_without_first_column = T(:, 1);

% Convert the table to a numeric matrix
index = table2array(T_without_first_column);

% Save data in .mat format
[~, index, ~] = fileparts(index);
save([index, 'index.mat'], 'index');
end

function [mat]=generate_m(n)
mat=randn(n,512);
end

function [ww] = LSH(w, mat)
v = mat * w';
ww = sign(v);
ww(ww == -1) = 0;
ww = ww';
end

function [frr, far] = getfarfrr(gen, imp, mine, maxe, dec)
gar = [];
for t = mine:dec:maxe
    gencal = gen(gen(:) <= t);
    if isempty(gencal)
        genrate = 0;
    else
        genrate = length(gencal) / length(gen);
    end
    gar = [gar; genrate];
end
frr = 1 - gar;

far = [];
for t = mine:dec:maxe
    impcal = imp(imp(:) <= t);
    if isempty(impcal)
        imprate = 0;
    else
        imprate = length(impcal) / length(imp);
    end
    far = [far; imprate];
end
end

function [yfil, frmat] = Encoding_mat(x, n, t)
k = size(x, 1);
rmat = randn(n, k);
y = rmat * x;
absy = abs(y);
[sorted_data, sortedindex] = sort(absy, 'descend');
topindex = (sortedindex(1:t));
frmat = rmat(topindex, :);
yfil = y(topindex);
end

function [RandomfieldM, yfil] = main_alg(Ib, n, L)
ii = 1;
inI = Ib(:);
RandomfieldM = eye(length(inI));
while ii <= L
    [yfil, frmat] = Encoding_mat(inI, n, length(inI));
    inI = yfil;
    RandomfieldM = frmat * RandomfieldM;
    ii = ii + 1;
end
end




function [data, labels] = generateClusters(N, dim)
% Generates synthetic data with 4 clusters in high-dimensional space.
% N: Total number of points to generate
% dim: Number of dimensions for the data

% Parameters for synthetic data
numClusters = 4; % Number of clusters
pointsPerCluster = N / numClusters; % Divide points equally across clusters

% Set random seed for reproducibility
rng(42);

% Define cluster centers in high-dimensional space (closer together)
clusterCenters = [rand(1, dim)*5;
    rand(1, dim)*5 + 5;
    rand(1, dim)*5 + 10;
    rand(1, dim)*5 + 15];

% Define smaller covariance matrices for smaller spread (smaller variance)
covariances = repmat(0.5 * eye(dim), [1, 1, numClusters]); % Reduce variance to make clusters tighter

% Generate synthetic data
data = [];
labels = [];

for i = 1:numClusters
    % Generate random data for each cluster using multivariate normal distribution
    clusterData = mvnrnd(clusterCenters(i,:), covariances(:,:,i), pointsPerCluster);
    data = [data; clusterData];
    labels = [labels; repmat(i, pointsPerCluster, 1)]; % Assign a label to each cluster
end

% Optionally visualize the data using PCA if it's high-dimensional
if dim <= 10 % Only reduce dimensions if the data is high-dimensional
    [coeff, score, ~] = pca(data);  % PCA for dimensionality reduction
    reducedData = score(:,1:2);  % Keep the first two principal components

    % Visualize the reduced data
    figure;
    gscatter(reducedData(:,1), reducedData(:,2), labels, 'rgbc', 'osd^', 8);
    title('PCA Visualization of 100D Data with 4 Clusters');
    xlabel('PC1');
    ylabel('PC2');
    legend('Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4');
    axis equal;
end
data=data';
end
