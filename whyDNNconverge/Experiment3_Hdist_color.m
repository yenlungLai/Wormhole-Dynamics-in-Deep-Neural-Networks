
clear all

avgs=[];

n=100;

L=15;
N=100; % generate 50 random pair
k=3; %dimension of input vector x_0



for jj=1:N

    u=randn(1,k);
    v=randn(1,k);



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

        y1=[ii; y1]; y2=[ii; y2];


        y1 = y1/norm(y1);  % Normalize the vector
        y2 = y2/norm(y2);  % Normalize the vector


%         dis=acos(dot(y1, y2))/pi;% compute their pairwise dot product
                dis=norm(y1-y2)^2/4;
%         dis=sum((abs(y1)-abs(y2)))/k;
%         dis=sum(sign(y1)~=sign(y2))/k;

        pointr=[pointr;dis];
        ii=ii+1;

    end

    pointrR{jj}=pointr;


end


% Create a figure
figure;

% Plot each set of values
for i = 1:numel(pointrR)
    plot(1:length(pointrR{i}), pointrR{i}, '-o', 'DisplayName', ['Time ' num2str(i)],'MarkerSize',6, 'MarkerIndices',1:3:L, 'LineWidth', 2.5);
    hold on;
end

xlabel(' $L$', 'Interpreter', 'latex');
ylabel('$d_{01}$', 'Interpreter', 'latex');


ylim([0,1]); xlim([1, L]);
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