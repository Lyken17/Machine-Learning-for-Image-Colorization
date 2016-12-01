%By Zeyu Zhao
function m_new=scale_matrix(m_old,x)
	m_new=imresize(m_old,x,'bilinear');
end