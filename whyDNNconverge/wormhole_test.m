clear all
n=500000;
for L=1:12
load("data/0.mat"); 
load("data/1.mat"); 
load("data/2.mat"); 
load("data/3.mat"); 
load("data/4.mat"); 
load("data/5.mat"); 
load("data/6.mat"); 
load("data/7.mat"); 
load("data/8.mat"); 
load("data/9.mat"); 

% Bob image
testimg1=one(:,:,3);
testimg2=two(:,:,3);
testimg3=three(:,:,3);
testimg4=four(:,:,3);
testimg5=five(:,:,3);
testimg6=six(:,:,3);
testimg7=seven(:,:,3);
testimg8=eight(:,:,3);
testimg9=nine(:,:,3);
testimg0=zero(:,:,3);



vin=m2vec(testimg0);



Ib=generate_random_vector(784, 'uniform', 0,255)'; % bob get random input Ib for unit vector

xpos=[];  % bob first generate RandomfieldM= WL WL-1 ... LW1
ii=1; inI=Ib(:);  
RandomfieldM=eye(length(inI)); 
while ii<=L
[yfil,frmat]=Encoding_mat(inI,n,length(inI));

% yfil=yfil/norm(yfil);
inI=yfil;
RandomfieldM=frmat*RandomfieldM;
xpos=[xpos;(abs(yfil))];
ii=ii+1;
end


alpha=100000;
bob_unit=(RandomfieldM*Ib)/norm((RandomfieldM*Ib));
cipher=vin-bob_unit*alpha; % here vin is the message bob want to tell alice



Ia=generate_random_vector(784, 'uniform', 0,255)'; % bob get random input Ib for unit vector
 % alice get random input Ia for unit vector


%  alice get her unit vector with agreed RandomfieldM
alice_unit=(RandomfieldM*Ia)/norm((RandomfieldM*Ia));

% alice, with agree alpha, recover message
alice_rec_m=cipher+alice_unit*alpha;
imshow(vec2m(alice_rec_m)')



% Define the output folder path
output_folder = 'output';

% Check if the output folder exists, if not, create it
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

% Save the reconstructed image as an image file in the output folder
% Save the reconstructed image with a filename dependent on parameter L
output_file = fullfile(output_folder, ['image L = ' num2str(L) '.png']);
imwrite(vec2m(alice_rec_m)', output_file);

disp(['Reconstructed image saved as: ' output_file]);


end





























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
