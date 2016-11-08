img=imread('F:\b.jpg');
[y,~,~,~]=rgb2yuv(imresize(img,[256,256]));
X=feature_extract_y(y);
u_p=predict(mdl_u,X);
v_p=predict(mdl_v,X);
u_pp=double(zeros(64,64));
v_pp=double(zeros(64,64));
for i=1:64
    for j=1:64
        u_pp(i,j)=u_p((i-1)*64+j);
        v_pp(i,j)=v_p((i-1)*64+j);
    end
end

u_p=scale_matrix(u_pp,4);
v_p=scale_matrix(v_pp,4);
[~,~,~,pimg]=yuv2rgb(y,u_p,v_p);
imshow(pimg);
