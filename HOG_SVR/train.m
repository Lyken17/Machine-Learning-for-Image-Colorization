 files=dir('F:/*.jpg');
X_F=[];
U_F=[];
V_F=[];
for file = files'
    [X,U,V]=feature_extract(imresize(imread(strcat('F:/',file.name)),[256,256]));
    X_F=[X_F;X];
    U_F=[U_F;U];
    V_F=[V_F;V];
end
mdl_u=fitrlinear(X_F,U_F);
mdl_v=fitrlinear(X_F,V_F);
save('model.mat','mdl_u','mdl_v');