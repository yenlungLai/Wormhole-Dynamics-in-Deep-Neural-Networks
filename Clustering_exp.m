clear all;

% This code is inheritly same concept applied to tsne_round2d and
% tsne_round3d, just for clean up purpurse, i have included Lfocus
% parameter to set our focus on particular Layer feature, for example
% H=G3G2G1, setting Lfocus=2 means we are observing the change of feagure at
% 1,2,3 layer after recursively apply H=G2G1 up to layer 2. This is because
% for too large Lfocus, the feature might collapse into singularity, tsne
% may not display properly the output distribution.


% Initialize parameters
n = 2^8; L = 12; k = 50; N = 500; R =6; % Set R to any desired number of rounds
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



%Section for feature selection Lfocus
%H=GLGl0e...G1   control the parameter Lfocus to 1, only focus on layer 1
Lfocus=1;
G_L = eye(length(u));
for ii=1:Lfocus
    G_L=frmatii{ii}*G_L;      %H=GLGl0e...G1
end



for Ncount=1:N                % generate layer feature for each point N
    v=randn(k,1);
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



for i=1:R        % generate feature for R iteration using previous set features
    for Ncount=1:N
        wp_ell= featureset{Ncount};
        wp_ell_t=G_L*wp_ell';
        wp_ell_t=wp_ell_t/norm(wp_ell_t);
        featureset{Ncount}=wp_ell_t';
        featuresetR{i}=featureset;
    end
end


% Prepare data for t-SNE
combined_vveccell = [featuresetR{:}];


for ll = 1:L
    vv = [];
    for jj = 1:R * N  % Adjust for all rounds and N samples
        vvecjj = combined_vveccell{jj};
        vvecs = vvecjj(ll, :);
        vv = [vv; vvecs];
    end
    vvL{ll} = vv;  % Store vectors for t-SNE
end


for i = 1:L
    y{i} = vvL{i};  % Prepare data for each subplot for particular i-th layer feature
end






% Number of rows and columns for subplots
cols = ceil(sqrt(L));
rows = ceil(L / cols);
% Marker size and edge thickness for plotting
markerSize1 = 1;
markerSize2 = 20;
lineWidth = 1.5;
fontSize = 12;

figure('Position', [100, 100, 800, 800]);  % Square aspect ratio for the whole figure
sil_scores = [];  % Array to store silhouette scores
dbi_scores = [];  % Array to store Davies-Bouldin index
inter_cluster_distances = [];  % Store inter-cluster distances
intra_cluster_distances = [];  % Store intra-cluster distances

% Iterate over L to generate t-SNE plots for each dataset
for i = 1:L
    combined_data =y{i};  % Combine data (x and y{i}) for t-SNE
    rng('default');  % Set random seed for reproducibility
    Y = tsne(combined_data, 'NumDimensions', 2);  % Perform t-SNE in 2D

    % Create subplot
    subplot(rows, cols, i);
    hold on;

    % Plot the 2D t-SNE result
    scatter(Y(:,1), Y(:,2), markerSize1, 'filled', 'MarkerEdgeColor', '#0072BD', 'MarkerFaceColor', '#1E90FF', 'LineWidth', lineWidth);

    % Highlight the first point
    scatter(Y(1, 1), Y(1, 2), markerSize2, 'MarkerEdgeColor', '#000000', 'MarkerFaceColor', '#D95319', 'LineWidth', lineWidth);

    hold off;

    % Set axes labels and other properties
    xlabel('t-SNE D-1', 'FontSize', fontSize);
    ylabel('t-SNE D-2', 'FontSize', fontSize);
    grid on;
    set(gca, 'FontSize', fontSize, 'LineWidth', 1.2);
    axis equal;
    title(['$\ell$=' num2str(i)], 'Interpreter', 'latex');
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
