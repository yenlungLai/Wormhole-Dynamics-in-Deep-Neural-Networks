clear all
n = 5000;
L = 10;
cc = 10000;
N = 100;

for c = [cc]
    inp_vecs = cell(1, N);  % Cell array to store input vectors
    vecs = cell(1, N);      % Cell array to store output vectors

    % Generating random input vectors
    RandomfieldM = eye(length(inp_vecs{1}));  % Initialize RandomfieldM
    for i = 1:N
        vecsii = [];
        inp_vecs = generate_random_vector(2, 'normal', 0, 1)';
        ii = 1;
        while ii <= L
            [yi, frmat] = Encoding_mat(inp_vecs, n, length(inp_vecs));
            inp_vecs = yi;
            yi = [ii * c; inp_vecs]; % Apply scaling
            yi = yi / norm(yi); % Normalize
            vecsii = [vecsii; (yi')]; % Store the output vectors
            ii = ii + 1;
        end
        vecs{i} = vecsii;
    end












    % Plotting the trajectories
    colors = lines(N);  % Get N distinct colors for plotting

    for i = 1:N
        % Plotting the trajectory for each vector
        plot3(vecs{i}(:, 2), vecs{i}(:, 3), vecs{i}(:, 1), '--','linewidth', 1.5, 'color', colors(i, :));
        hold on;
        % Plotting the start and end points
        plot3(vecs{i}(1, 2), vecs{i}(1, 3), vecs{i}(1, 1), 'o', 'MarkerSize', 10, 'linewidth', 2, 'MarkerEdgeColor', 'k', 'MarkerFaceColor', colors(i, :)); % Start point
        plot3(vecs{i}(end, 2), vecs{i}(end, 3), vecs{i}(end, 1), 's', 'MarkerSize', 10, 'linewidth', 2, 'MarkerEdgeColor', 'k', 'MarkerFaceColor', colors(i, :)); % End point
        
        % Plotting arrows along the trajectory
        quiver3(vecs{i}(1:end-1, 2), vecs{i}(1:end-1, 3), vecs{i}(1:end-1, 1), ...
            diff(vecs{i}(:, 2)), diff(vecs{i}(:, 3)), diff(vecs{i}(:, 1)), ...
            'Color', colors(i, :), 'MaxHeadSize', 0.6, 'LineWidth', 1);
    end

% %   
%       xlabel('$\frac{\hat{i}_L}{\sqrt{(Lc)^{2}+\hat{i}^2_L+\hat{j}^2_L}}$', 'Interpreter', 'latex');
%         ylabel('$\frac{\hat{j}_L}{\sqrt{(Lc)^{2}+\hat{i}^2_L+\hat{j}^2_L}}$', 'Interpreter', 'latex');
%      zlabel('$\frac{Lc}{\sqrt{(Lc)^{2}+\hat{i}^2_L+\hat{j}^2_L}}$', 'Interpreter', 'latex');

    % Constructing legend labels
    legend_labels = cell(1, 3 * N);
    for i = 1:N
        legend_labels{(i - 1) * 3 + 1} = ['$x^{(', num2str(i), ')}$ Trajectory'];
        legend_labels{(i - 1) * 3 + 2} = ['$x^{(', num2str(i), ')}_{0}$'];
        legend_labels{(i - 1) * 3 + 3} = ['$x^{(', num2str(i), ')}_{L}$'];
    end

%   legend(legend_labels, 'Location', 'best', 'Interpreter', 'latex');
    hold off;
    zlim([0,1]);
    xlim([-1,1]);
    ylim([-1,1]);
    % Making the grid square
       % Making the grid square only for x and y axes
    daspect([1 1 0.7]); % Set data aspect ratio, we will manually adjust z
    pbaspect([1 1 0.5]); % Adjust relative aspect ratio

    grid on;
    hold off;






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






function distance = euclidean_distance(v1, v2)
    % Check if the vectors have the same dimensionality
    assert(length(v1) == length(v2), 'Vectors must have the same dimensionality');
    
    % Compute the squared differences of corresponding elements
    squared_diff = (v1 - v2).^2;
    
    % Compute the sum of squared differences
    sum_squared_diff = sum(squared_diff);
    
    % Compute the square root of the sum of squared differences
    distance = sqrt(sum_squared_diff);
end




% Function to compute angles between consecutive intervals
function angles = computeAngles(vec)
    angles = zeros(size(vec, 1) - 1, 1); % Initialize array to store angles

    for i = 1:size(vec, 1) - 1
        % Compute dot product between consecutive vectors
        dot_product = dot(vec(i, :), vec(i + 1, :));

        % Compute magnitudes of consecutive vectors
        magnitude1 = norm(vec(i, :));
        magnitude2 = norm(vec(i + 1, :));

        % Compute angle (in radians) between consecutive vectors
        angles(i) = acos(dot_product / (magnitude1 * magnitude2));
    end
end