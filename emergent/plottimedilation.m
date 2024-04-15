% Constants
c = 299792458; % Speed of light in m/s
hbar = 1.0546e-34; % Reduced Planck constant in J*s

% Initialize arrays to store values
n_values = 1:1:100;
s_values = zeros(size(n_values));
loretz= zeros(size(n_values));
pec=zeros(size(n_values));
v=zeros(size(n_values));
% Calculate s for each value of n
for i = 1:numel(n_values)
    n = n_values(i);
    s_values(i) =sqrt( sqrt(n / hbar) / c^2);
    s_values(i)=(s_values(i)-1)/s_values(i);
%     pec(i)=(n/size(n_values,2));
 
    v(i)=sqrt((1-(s_values(i)*s_values(i)))*c^2);
    loretz(i)=sqrt(1-(v(i)/c)^2);
end

% Plot
xlabel('$F$'  ,Interpreter="latex");
ylabel('$c''$',Interpreter="latex");
hold on 
% plot(n_values, loretz, 'Marker', '*'); % Changed marker to '*'
title('Plot of  $F$ vs $c''$', Interpreter='latex');
plot(n_values,v, 'Marker', 'o', 'Color', 'blue', 'LineWidth', 2, 'MarkerIndices', 1:5:numel(n_values)); % Changed marker to 'o', line color to red, line width to 2, and set marker interval

