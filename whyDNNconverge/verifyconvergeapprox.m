clear all
dd = 20000;
layer = 10;
normalized_layer = zeros(1, dd); % Preallocate array to store normalized layer values

points = [];

for i = -dd/2:1:dd/2
    x = [layer, i];
    normalizex = x / norm(x);
    points=[points;normalizex];
end

scatter(points(:,2),points(:,1))


xlabel('$\frac{\hat{i}_L}{\sqrt{(Lc)^{2}+\hat{i}^2_L+\hat{j}^2_L}}$', 'Interpreter', 'latex');
ylabel('$\frac{Lc}{\sqrt{(Lc)^{2}+\hat{i}^2_L+\hat{j}^2_L}}$', 'Interpreter', 'latex');







dd=1/dd;
% Define a vector of probabilities (normalized y values)
bev=[];
for p=0:dd:1
p
    % Compute binary entropy for each probability
    binary_entropy = -p .* log2(p) - (1 - p) .* log2(1 - p);
    bev=[bev;binary_entropy];

end

figure


plot(points(:,2),points(:,1),  '-','LineWidth',2.5)
hold on
plot(0:dd:1,bev,'color','r','LineWidth',2.5)

legend('$\frac{\hat{x}_{L}}{||\hat{x}_{L}||}$','$H_2(\hat{\theta})$', 'Interpreter', 'latex')

ylabel('$\frac{Lc}{\sqrt{(Lc)^{2}+\hat{i}^2_L+\hat{j}^2_L}}$', 'Interpreter', 'latex');
xlabel('$\frac{{x}_{L}}{\sqrt{(Lc)^{2}+||x_L||}}$', 'Interpreter', 'latex');






