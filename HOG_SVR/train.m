img=imread('F:\a.jpg');
[X,U,V]=feature_extract(img);
mdl_u=fitrlinear(X,U);
mdl_v=fitrlinear(X,V);