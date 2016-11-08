img=imread('F:\a.jpg');
R=img(:,:,1);
G=img(:,:,2);
B=img(:,:,3);
Y=[];
U=[];
V=[];
for i=1:256
    for j=1:256
        Y
    end
end
Y = 0.299   * R + 0.587   * G + 0.114 * B;
U =128 - 0.168736 * R - 0.331264 * G + 0.5 * B;
V =128 + 0.5 * R - 0.418688 * G - 0.081312 * B;
points = zeros(240*240,2);
for i = 9:248 
    for j = 9:248
        points((i-9)*240+(j-8),:)=[i,j];
    end
end
[features,v]=  extractHOGFeatures(b,points,'BlockSize',[2,2]);
U_shaped=[];
for i = 9:248
    for j=9:248
        U_shaped=[U_shaped;U(i,j)];
    end
end
modelU=fitrsvm(features,U_shaped);



B_E = 1.164*(Y - 16) + 2.018*(U - 128);

G_E = 1.164*(Y - 16) - 0.813*(V - 128) - 0.391*(U - 128);

R_E = 1.164*(Y - 16) + 1.596*(V - 128);
img_cons=zeros(256,256,3);
for i=1:256
    for j=1:256
        img_cons(i,j)=[R_E(i,j),G_E(i,j),B_E(i,j)];
    end
end