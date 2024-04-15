u=randn(1,2);
v=randn(1,2);


n=5000; k=2; L=500;


figure; % Create a new figure
hold on; % Hold the plot
ii=1; nter=L;
% Determine the color for the start markers
start_u_color = [0.8500, 0.3250, 0.0980]; % red for w
start_v_color = [0, 0.4470, 0.7410]; % Blue for w'

% Plot starting points
start_u = plot(u(1), u(2), 'o', 'MarkerSize',8, 'MarkerFaceColor', start_u_color, 'MarkerEdgeColor', start_u_color, 'LineWidth', 2, 'DisplayName', '$x''_0$ Start'); % Plot v1 start marker
start_v = plot(v(1), v(2), 'o', 'MarkerSize',8, 'MarkerFaceColor', start_v_color, 'MarkerEdgeColor', start_v_color, 'LineWidth', 2, 'DisplayName', '$x_0$ Start'); % Plot v2 start marker

% Create variables to store old values
old_u = u;
old_v = v;
dis=acos(dot(u, v) / (norm(u) * norm(v))) / pi;
distpoint=[];



% entanglement take place here
% we apply encoding directly on u and v for each encoding time step
    
while ii<nter
    % Encoding for u
    [reduced_codeword1, reduced_matrix] = Encoding(u', n, k);
    fu1 = reduced_codeword1';
    u = fu1;u=u/norm(u);
    

    % Encoding for v
    reduced_codeword2 = reduced_matrix * (v');
    fu2 = reduced_codeword2';
    v = fu2; v=v/norm(v);

    ii = ii + 1;


    % Plot updated vectors
    % Draw lines from old points to new points
    line([old_u(1), u(1)], [old_u(2), u(2)], 'Color', start_u_color, 'LineWidth', 1);
    line([old_v(1), v(1)], [old_v(2), v(2)], 'Color', start_v_color, 'LineWidth', 1);

    % Store current points as old points for the next iteration
    old_u = u;
    old_v = v;

end



grid on; % Turn the grid on
xlabel('$x$',Interpreter='latex'); % Label X-axis
ylabel('$y$',Interpreter='latex'); % Label Y-axis

% Dummy lines for legend
dummy_u = line([NaN], [NaN], 'Color', start_u_color, 'LineWidth', 1, 'DisplayName', '$x''_{\ell}$ Path');
dummy_v = line([NaN], [NaN], 'Color', start_v_color, 'LineWidth', 1, 'DisplayName', '$x_{\ell}$ Path');

% Plot end points
end_u = plot(u(1), u(2), 'ks', 'MarkerSize',14, 'MarkerFaceColor', 'none', 'LineWidth', 2, 'DisplayName', '$x''_L$  End'); % Plot v1 end as black square
end_v = plot(v(1), v(2), 'kd', 'MarkerSize',14, 'MarkerFaceColor', 'none', 'LineWidth', 2, 'DisplayName', '$x_L$ End'); % Plot v2 end as black diamond

legend([start_u, start_v, dummy_u, dummy_v, end_u, end_v],'Interpreter', 'latex'); % This will automatically grab the 'DisplayName' properties of your plot objects
hold off; % Release the plot






% 
% 
% function [yfil,frmat]=pfilter2(x,n,t)
% k=length(x);
% 
% rmat=randn(n,k);
% y=rmat*x;
% absy=abs(y);
% [sorted_data, sortedindex ]= sort(absy, 'ascend');
% topindex=(sortedindex(1:t));
% frmat=rmat(topindex,:);
% yfil=y(topindex);
% 
% end










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





function [targetVec, vecs] = generateVectorsLessThan90Degrees(m)
    targetVec = randn(1, 2);  % Generate target vector
    targetVec = targetVec / norm(targetVec);  % Normalize the vector

    vecs = zeros(m, 2);  % Preallocate matrix for efficiency

    for i = 1:m
        % Generate a random angle in radians less than 90 degrees
        angle = rand()*pi/2;
        % Create rotation matrix
        R = [cos(angle) -sin(angle); sin(angle) cos(angle)];
        % Rotate target vector
        rotatedVec = (R * targetVec')';
        % Store the rotated vector in the matrix
        vecs(i, :) = rotatedVec;
    end


end


function [targetVec, vecs] = generateVectorsMoreThan90Degrees(m)
    targetVec = randn(1, 2);  % Generate target vector
    targetVec = targetVec / norm(targetVec);  % Normalize the vector

    vecs = zeros(m, 2);  % Preallocate matrix for efficiency

    for i = 1:m
        % Generate a random angle in radians more than 90 degrees but less than 180 degrees
        angle = pi/2 + rand()*pi/2;
        % Create rotation matrix
        R = [cos(angle) -sin(angle); sin(angle) cos(angle)];
        % Rotate target vector
        rotatedVec = (R * targetVec')';
        % Store the rotated vector in the matrix
        vecs(i, :) = rotatedVec;
    end

end


