%By Zeyu Zhao
%padding matrix M*N left x right x top y bottom y with value 0
%for HOG extraction
function m_new=matrix_padding(m_old,x,y)
	p=zeros(size(m_old,1),x);
    m_new=cat(2,p,m_old,p);
    q=zeros(y,size(m_new,2));
    m_new=cat(1,q,m_new,q);
end