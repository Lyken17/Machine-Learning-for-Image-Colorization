img=imread('F:\b.jpg');
[y,~,~,~]=rgb2yuv(img);
X=feature_extract_y(y);
u=predict(mdl_u,X);
v=predict(mdl_v,X);
U=double(zeros(256,256));
V=double(zeros(256,256));
for i=1:256
    for j=1:256
        U(i,j)=u((i-1)*256+j,1);
        V(i,j)=v((i-1)*256+j,1);
    end
end
[~,~,~,imgn]=yuv2rgb(y,U,V);
imshow(imgn)