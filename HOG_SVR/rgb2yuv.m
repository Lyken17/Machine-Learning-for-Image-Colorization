%By Zeyu Zhao
function [y,u,v,yuv]=rgb2yuv(image)
	rgb = double(image);
	R=rgb(:,:,1) / 255.0;
	G=rgb(:,:,2) / 255.0;
	B=rgb(:,:,3) / 255.0;
	y = 0.299 * R + 0.587 * G + 0.114 * B;
	u = -0.14713 * R - 0.28886 * G + 0.436 * B;
	v = 0.615 * R - 0.51499 * G - 0.10001 * B;
	yuv=cat(3,y,u,v);
end
