%By Zeyu Zhao
%Input image must be 256*256
function X=feature_extract_y(ymtx)
    %Downsampling
    y=scale_matrix(ymtx,0.25);
    points = zeros(64*64,2);
    Y=[];
    for i = 1:64
        for j = 1:64
            points((i-1)*64+j,:)=[i+8,j+8];
            Y=[Y;y(i,j)];
        end
    end
    y=matrix_padding(y,8,8);
    [X,~]=  extractHOGFeatures(y,points,'BlockSize',[2,2]);
    X=cat(2,X,Y);
end