%By Zeyu Zhao
function [r,g,b,rgb]=yuv2rgb(y,u,v)
	r=round(y+1.13983*(v-128));
    g=round(y-0.39465*(u-128)-0.58060*(v-128));
    b=round(y+2.03211*(u-128));
    for i=1:numel(r)
        if r(i)<0
            r(i)=0;
        end
        if r(i)>255
            r(i)=255;
        end
        if g(i)<0
            g(i)=0;
        end
        if g(i)>255
            g(i)=255;
        end
        if b(i)<0
            b(i)=0;
        end
        if b(i)>255
            b(i)=255;
        end
    end
    r=uint8(r);
    g=uint8(g);
    b=uint8(b);
    rgb=cat(3,r,g,b);
end