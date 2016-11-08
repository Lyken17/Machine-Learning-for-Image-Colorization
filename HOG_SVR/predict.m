img=imread('F:\b.jpg');
[y,~,~,~]=rgb2yuv(img);
X=feature_extract_y(y);
u=predict(mdl_u,X);
v=predict(mdl_v,X);
U=double(zeros(64,64));
V=double(zeros(64,64));
for i=1:64
    for j=1:64
        U(i,j)=u((i-1)*64+j,1);
        V(i,j)=v((i-1)*64+j,1);
    end
end
y=scale_matrix(y,0.25);
[~,~,~,imgn]=yuv2rgb(y,U,V);
imshow(imgn)