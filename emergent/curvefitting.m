clear all
n=50;
L=10;



vin=abs(randn(1,1));
vin2=abs(randn(1,1));


xpos=[];  % use 0 to get yfil (transformed 0 using force)
ii=1; inI=vin(:);  
frmat_c=eye(length(inI)); 
while ii<=L
[yfil,frmat]=Encoding_mat(inI,n,length(inI));

% yfil=yfil/norm(yfil);
inI=(yfil);
frmat_c=frmat*frmat_c;
xpos=[xpos;(abs(yfil))];
ii=ii+1;
end

yfil=yfil/norm(yfil);  
yfil2=frmat_c*vin2/norm(frmat_c*vin2);


t = numel(xpos); % Get the number of elements in xpos

% Create an array for the x-axis ranging from 1 to t
x_axis = 1:t;

% Plot the graph
plot(x_axis, xpos, '-o', 'Color', [0.1, 0.5, 0.9], 'LineWidth', 2, 'MarkerSize', 10, 'DisplayName', 'Data');


hold on 

k = 1; % Growth rate constant

% Calculate x values using the exponential growth equation
x = vin * exp(k * x_axis); x = x';

% Plot the exponential growth
% Plot the exponential growth
plot(x_axis, x, '-s', 'Color', [0.9, 0.2, 0.1], 'LineWidth', 2, 'MarkerSize', 10, 'DisplayName', 'Exponential Growth');

% Add legend with LaTeX-formatted text
xlabel('Layer, $L$', 'Interpreter', 'latex');
ylabel('Position, $|x_L|$','Interpreter', 'latex');
title('Position, $|x_L|$ vs Layer ($L$)', 'Interpreter', 'latex');
grid on;

% Add legend
legend('$|x_L|=|H_{\textnormal{Linear}}(x_0)|=|W_LW_{L-1}\ldots W_1 x_0|$','$|x_L|=|x_0 \exp(\Lambda L)|$', 'Interpreter', 'latex')






























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


function convolved_features = convolution(image, kernel)
    % Get dimensions of input image and kernel
    [image_height, image_width] = size(image);
    [kernel_height, kernel_width] = size(kernel);
    
    % Calculate output feature map size
    output_height = 28;
    output_width = 28;
    
    % Initialize output feature map
    convolved_features = zeros(output_height, output_width);
    
    % Pad the input image
    padded_image = padarray(image, [floor(kernel_height/2), floor(kernel_width/2)]);
    
    % Perform convolution
    for i = 1:output_height
        for j = 1:output_width
            % Extract patch from padded input image
            image_patch = padded_image(i:i+kernel_height-1, j:j+kernel_width-1);
            
            % Compute dot product between image patch and kernel
            convolved_features(i, j) = sum(sum(image_patch .* kernel));
        end
    end
end




function pooled_features = max_pooling(features, pool_size)
    % Get dimensions of input feature map
    [feature_height, feature_width, num_channels] = size(features);
    
    % Calculate size of pooled feature map
    pooled_height = floor(feature_height / pool_size);
    pooled_width = floor(feature_width / pool_size);
    
    % Initialize pooled feature map
    pooled_features = zeros(pooled_height, pooled_width, num_channels);
    
    % Perform max pooling
    for k = 1:num_channels
        for i = 1:pooled_height
            for j = 1:pooled_width
                % Extract patch from input feature map
                feature_patch = features((i-1)*pool_size+1:i*pool_size, (j-1)*pool_size+1:j*pool_size, k);
                
                % Compute maximum value in the patch
                pooled_features(i, j, k) = max(feature_patch(:));
            end
        end
    end
end




function output_features = forward_convolution_pooling(input_image, kernels)
    num_kernels = numel(kernels);
    % Apply convolution
    convolved_features = zeros(size(input_image, 1), size(input_image, 2), num_kernels);
    for k = 1:num_kernels
        convolved_features(:,:,k) = convolution(input_image, kernels{k});
    end
    
    % Apply max pooling
    pooled_features = max_pooling(convolved_features, 2);
    
    % Flatten the pooled features
    output_features = reshape(pooled_features, [], size(pooled_features, 3));
end

