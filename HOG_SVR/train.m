%By Zeyu Zhao
%Input image must be 256*256
files=dir('train2014/*.jpg');
X_F=[];
U_F=[];
V_F=[];
for file = files'
    img=imread(strcat('train2014/',file.name));
    if size(img,3) < 3
        continue
    end
    file.name
    [X,U,V]=feature_extract(imresize(img,[256,256]));
    X_F=[X_F;X];
    U_F=[U_F;U];
    V_F=[V_F;V];
end
mdl_u=fitrlinear(X_F,U_F);
mdl_v=fitrlinear(X_F,V_F);
save('model.mat','mdl_u','mdl_v');