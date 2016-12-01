%By Zeyu Zhao
%Input image must be 256*256
function [X,U,V]=feature_extract(img)
    [y,u,v,~]=rgb2yuv(img);
    %Downsampling
    y=scale_matrix(y,0.25);
    u=scale_matrix(u,0.25);
    v=scale_matrix(v,0.25);
    points = zeros(64*64,2);
    U=[];
    V=[];
    Y=[];
    for i = 1:64
        for j = 1:64
            points((i-1)*64+j,:)=[i+8,j+8];
            U=[U;u(i,j)];
            V=[V;v(i,j)];
            Y=[Y;y(i,j)];
        end
    end
    y=matrix_padding(y,8,8);
    [X,~]=  extractHOGFeatures(y,points,'BlockSize',[2,2]);
    X=cat(2,X,Y);
end