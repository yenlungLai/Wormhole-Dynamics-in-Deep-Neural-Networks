clear all

n = 5;L=1; c=1;

L_cluster = [];


 
ii=1; 

while ii<=L
    for i = 1:10000
        inp_vec1 = generate_random_vector(2, 'normal', 0,1)';
        [RandomfieldM,y1] =main_alg(inp_vec1,n,ii);

%                y1=y1/norm(y1);
        L_cluster = [L_cluster; [ii*c y1']];
    end

    ii=ii+1;
end





% figure;
hold on;
unique_values = unique(L_cluster(:,1));
num_unique = numel(unique_values);
colors = parula(num_unique); % Change colormap for better visibility
for i = 1:num_unique
    indices = L_cluster(:,1) == unique_values(i);
    color_index = mod(i-1, size(colors,1)) + 1; % Use modulo arithmetic to cycle through colors
    scatter3(L_cluster(indices,2), L_cluster(indices,3), L_cluster(indices,1), '.', 'MarkerFaceColor', colors(color_index,:), 'DisplayName', ['$L$ = ', num2str(unique_values(i))]);
end
hold off;
grid on;
      xlabel('$\hat{i}_{L}$', 'Interpreter', 'latex');
        ylabel('$\hat{j}_{L}$', 'Interpreter', 'latex');
     zlabel('$L$', 'Interpreter', 'latex');

% title('Cluster Plot for L Cluster');
% legend('show', 'Location', 'Best','Interpreter', 'latex'); % Show legend with colored markers
rotate3d on;  % Enable interactive rotation
view(3); % Set view angle to 3D
axis vis3d; % Make sure all axes have equal scaling






















% % Plot the clusters in 3D
% figure;
% hold on;
% colors = {'b', 'r', 'g', 'k', 'c', 'm', 'y', [0.5, 0.5, 0.5], [0.75, 0.25, 0], [0, 0.75, 0.75]};
% 
% for j = 1:L
%     cluster_indices = (L_cluser(:,1) == j); % Find indices for the current value of L
%     scatter3(L_cluser(cluster_indices,2), L_cluser(cluster_indices,3), L_cluser(cluster_indices,1), '.', colors{j}); % Scatter plot for each L with different colors
% end
% hold off;
% xlabel('X');
% ylabel('Y');
% zlabel('L');
% title('Cluster Plot for Different L');
% legend('L = 1', 'L = 2', 'L = 3');
% 
% 
% 








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
absy=abs(y);
[sorted_data, sortedindex ]= sort(absy, 'descend');
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




function noisy_vector = addNoiseToVector(vector, noise_type, varargin)
% Get the length of the vector
len = length(vector);

% Initialize noisy vector
noisy_vector = zeros(size(vector));

% Add noise based on noise type
switch lower(noise_type)
    case 'gaussian'
        % Parse optional arguments
        if nargin > 2
            mean = varargin{1};
            stddev = varargin{2};
        else
            mean = 0;
            stddev = 0.1; % default standard deviation
        end
        % Add Gaussian noise
        noise = mean + stddev * randn(size(vector));
        noisy_vector = vector + noise;
    case 'uniform'
        % Parse optional arguments
        if nargin > 2
            a = varargin{1};
            b = varargin{2};
        else
            a = -0.5;
            b = 0.5; % default range
        end
        % Add uniform noise
        noise = a + (b-a) * rand(size(vector));
        noisy_vector = vector + noise;
    case 'salt_and_pepper'
        % Parse optional argument
        if nargin > 2
            density = varargin{1};
        else
            density = 0.05; % default density
        end
        % Add salt and pepper noise
        noisy_indices = randperm(len, floor(density * len));
        noisy_vector(noisy_indices) = randi([0, 1], size(noisy_indices)) * 2 - 1;
        noisy_vector = vector + noisy_vector;
    otherwise
        error('Unknown noise type. Please choose from ''gaussian'', ''uniform'', or ''salt_and_pepper''.');
end
end







function features = extractGaborFeatures(image, pool_size)
    % Define parameters for Gabor filter bank
    numScales = 4; % Number of scales
    numOrientations = 6; % Number of orientations
    lambda = 3; % Wavelength of the sinusoidal factor
    gamma = 0.5; % Spatial aspect ratio
    psi = 0; % Phase offset
    sigma = 2; % Spatial standard deviation

    % Create Gabor filter bank
    gaborBank = cell(numScales, numOrientations);
    for s = 1:numScales
        for o = 1:numOrientations
            theta = (o - 1) * pi / numOrientations;
            g = createGaborFilter(lambda, theta, sigma, gamma, psi);
            gaborBank{s, o} = g;
        end
        lambda = lambda * 2; % Increase wavelength for next scale
    end

    % Apply Gabor filters to the input image
    [height, width] = size(image);
    pooled_height = floor(height / pool_size);
    pooled_width = floor(width / pool_size);
    features = zeros(pooled_height * pooled_width, numScales * numOrientations);
    for s = 1:numScales
        for o = 1:numOrientations
            g = gaborBank{s, o};
            filteredImage = imfilter(image, g, 'symmetric', 'conv');
            pooled_features = pooling(filteredImage, pool_size);
            features(:, (s - 1) * numOrientations + o) = reshape(pooled_features, [], 1);
        end
    end
end

function g = createGaborFilter(lambda, theta, sigma, gamma, psi)
    % Create Gabor filter
    halfSize = ceil(3 * sigma);
    [x, y] = meshgrid(-halfSize:halfSize);
    xTheta = x * cos(theta) + y * sin(theta);
    yTheta = -x * sin(theta) + y * cos(theta);
    g = exp(-(xTheta.^2 + (gamma * yTheta).^2) / (2 * sigma^2)) .* cos(2 * pi / lambda * xTheta + psi);
end

function pooled_features = pooling(image, pool_size)
    % Apply max pooling to the image
    [height, width] = size(image);
    pooled_height = floor(height / pool_size);
    pooled_width = floor(width / pool_size);
    pooled_features = zeros(pooled_height, pooled_width);
    for i = 1:pooled_height
        for j = 1:pooled_width
            row_start = (i - 1) * pool_size + 1;
            row_end = min(row_start + pool_size - 1, height);
            col_start = (j - 1) * pool_size + 1;
            col_end = min(col_start + pool_size - 1, width);
            pooled_features(i, j) = max(max(image(row_start:row_end, col_start:col_end)));
        end
    end
end



function random_vector = generate_random_vector(k, distribution_name, varargin)
    % Generate random vector of dimension k from a specified distribution
    
    % Check if distribution_name is valid
    valid_distributions = {'normal', 'uniform', 'exponential', 'poisson', 'lognormal'};
    if ~ismember(distribution_name, valid_distributions)
        error('Invalid distribution name. Choose from: normal, uniform, exponential, poisson, lognormal');
    end
    
    % Generate random vector
    switch distribution_name
        case 'normal'
            mu = varargin{1}; % Mean
            sigma = varargin{2}; % Standard deviation
            random_vector = normrnd(mu, sigma, [1, k]);
        case 'uniform'
            lower_bound = varargin{1}; % Lower bound
            upper_bound = varargin{2}; % Upper bound
            random_vector = unifrnd(lower_bound, upper_bound, [1, k]);
        case 'exponential'
            lambda = varargin{1}; % Rate parameter
            random_vector = exprnd(lambda, [1, k]);
        case 'poisson'
            lambda = varargin{1}; % Mean rate
            random_vector = poissrnd(lambda, [1, k]);
        case 'lognormal'
            mu = varargin{1}; % Mean of the underlying normal distribution
            sigma = varargin{2}; % Standard deviation of the underlying normal distribution
            random_vector = lognrnd(mu, sigma, [1, k]);
    end
end

