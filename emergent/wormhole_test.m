clear all
n=1000;
for L=[10 20 30 40 50 60 70 80 90 100]
load("4.mat"); 

% Bob image
testimg=four(:,:,2);
vin_noisy=randn(784,1);

vin=m2vec(testimg)/255;



Ib=randn(length(vin),1); % bob get random input Ib for unit vector

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


alpha=50;
bob_unit=(RandomfieldM*Ib)/norm((RandomfieldM*Ib));
cipher=bob_unit*alpha+vin; % here vin is the message bob want to tell alice



Ia=randn(length(vin),1); % alice get random input Ia for unit vector


%  alice get her unit vector with agreed RandomfieldM
alice_unit=(RandomfieldM*Ia)/norm((RandomfieldM*Ia));

% alice, with agree alpha, recover message
alice_rec_m=cipher-alice_unit*alpha;
imshow(vec2m(alice_rec_m))



% Define the output folder path
output_folder = 'output';

% Check if the output folder exists, if not, create it
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

% Save the reconstructed image as an image file in the output folder
% Save the reconstructed image with a filename dependent on parameter L
output_file = fullfile(output_folder, ['image L = ' num2str(L) '.png']);
imwrite(vec2m(alice_rec_m), output_file);

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

