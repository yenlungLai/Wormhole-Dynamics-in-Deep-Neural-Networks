
clear all

avgs=[];

n=5000;

L=15;
N=100; % generate 50 random pair
k=100; %dimension of input vector x_0



for jj=1:N

    u=generate_random_vector(k, 'uniform', -6000,6000);
    v=generate_random_vector(k, 'uniform', -6000,6000);




    % we apply main alagorithm directly on u and take W_L W_L-1...W for v for each encoding time step
    pointr=[];

    initt=acos(dot(u, v))/pi; % compute their initital dot product

    ii=1; inI=u(:);
    RandomfieldM=eye(length(inI));
    while ii<=L
        [y1,frmat]=Encoding_mat(inI,n,length(inI));

        inI=y1;
        RandomfieldM=frmat*RandomfieldM;



        y2=RandomfieldM*v';


%         y1=[ii; y1]; y2=[ii; y2];

        y1 = y1/norm(y1);  % Normalize the vector
        y1=relu(y1);

        y2 = y2/norm(y2);  % Normalize the vector
        y2=relu(y2);

%         dis=acos(dot(y1, y2))/pi;% compute their pairwise dot product
                dis=norm(y1-y2)^2/4;
        %         dis=sum(sign(u)~=sign(v))/512;

        pointr=[pointr;dis];
        ii=ii+1;

    end

    pointrR{jj}=pointr;


end


% Create a figure
figure;

% Plot each set of values
for i = 1:numel(pointrR)
    plot(1:length(pointrR{i}), pointrR{i}, '-o', 'DisplayName', ['Time ' num2str(i)],'MarkerSize',6, 'MarkerIndices',1:1:L, 'LineWidth', 2.5);
    hold on;
end

xlabel(' $X=L$', 'Interpreter', 'latex');
ylabel('$Y= d_{\theta}$', 'Interpreter', 'latex');


ylim([0, 0.4]); xlim([1, L]);
% Hold off to stop superimposing new plots
hold off;


















function [yfil,frmat]=Encoding(x,n,t)
k=length(x);

rmat=randn(n,k);
% rmat=orth(rmat);
y=rmat*x;
absy=abs(y);
[sorted_data, sortedindex ]= sort(absy, 'descend');
topindex=(sortedindex(1:t));
frmat=rmat(topindex,:);

yfil=y(topindex);

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

function [RandomfieldM,yfil] =main_alg(Ib,n,L)


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



function y = relu(x)
    y = max(0, x);
end


function y = leaky_relu(x, alpha)
    y = max(alpha * x, x);
end




function y = tanh(x)
    y = (exp(x) - exp(-x)) ./ (exp(x) + exp(-x));
end


function y = softmax(x)
    exps = exp(x);
    y = exps ./ sum(exps);
end
