clear all

n = 5;L=1; c=1;
inp_vec1 = randn(3,1);
L_cluster = [];




ii=1; 
while ii<=L
 
    for i = 1:5000
        inp_vec1 = generate_random_vector(3, 'uniform', -5000,5000)';
        [RandomfieldM,y1] =main_alg(inp_vec1,n,ii);

        %        y1=y1/norm(y1);
        L_cluster = [L_cluster; [ii*c y1']];
    end

    ii=ii+1;
end





% Plot the clusters in 3D with different colors for each unique L_cluster(:,1)
figure;
hold on;
unique_values = unique(L_cluster(:,1));
num_colors = numel(unique_values);
colors = parula(num_colors); % Change colormap for better visibility
for i = 1:num_colors
    indices = L_cluster(:,1) == unique_values(i);
    scatter3(L_cluster(indices,2), L_cluster(indices,3), L_cluster(indices,4), L_cluster(indices,1), '.', 'MarkerFaceColor', colors(i,:), 'DisplayName', ['$L$ = ', num2str(unique_values(i))]);
end
hold off;
grid on;
      xlabel('$\hat{i}_{L}$', 'Interpreter', 'latex');
        ylabel('$\hat{j}_{L}$', 'Interpreter', 'latex');
     zlabel('$\hat{k}_{L}$', 'Interpreter', 'latex');
% Set aspect ratio to be equal
axis equal;
% Set equal axes limits
min_lim = min(min(L_cluster(:,2:4)));
max_lim = max(max(L_cluster(:,2:4)));
xlim([min_lim, max_lim]);
ylim([min_lim, max_lim]);
zlim([min_lim, max_lim]);
% title('Cluster Plot for L Cluster');
legend('show', 'Location', 'Best','Interpreter','latex'); % Show legend with colored markers
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

