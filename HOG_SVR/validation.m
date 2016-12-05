%By Zeyu Zhao
load('model.mat');
files=dir('val2014/*.jpg');
error=0.0;
count=0;
for file = files'
    img=imread(strcat('val2014/',file.name));
    if size(img,3) < 3
        continue
    end
    file.name
    [y,u,v,~]=rgb2yuv(imresize(img,[256,256]));
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
    [y_p,u_p,v_p,~]=rgb2yuv(pimg);
    error=error+mean(mean((y_p-y).^2))+mean(mean((u_p-u).^2))+mean(mean((v_p-v).^2));
    count=count+1;
end

%Normalize YUV from [0,255] to [0,1]
error=error/double(count)/2.0;
save('error.mat','error');
