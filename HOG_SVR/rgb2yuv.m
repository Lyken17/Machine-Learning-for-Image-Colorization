%By Zeyu Zhao
function [y,u,v,yuv]=rgb2yuv(image)
	rgb = double(image);
	R=rgb(:,:,1);
	G=rgb(:,:,2);
	B=rgb(:,:,3);
	y = 0.299 * R + 0.587 * G + 0.114 * B;
	u =128 - 0.168736 * R - 0.331264 * G + 0.5 * B;
	v =128 + 0.5 * R - 0.418688 * G - 0.081312 * B;
	yuv=cat(3,y,u,v);
end