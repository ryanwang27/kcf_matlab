function out = get_subwindow(im, pos, sz)
%GET_SUBWINDOW Obtain sub-window from image, with replication-padding.     ͨ����������ͼ���ȡ�Ӵ��ڡ�
%   Returns sub-window of image IM centered at POS ([y, x] coordinates),   ������POS��[y��x]���꣩Ϊ���ĵ�ͼ��IM���Ӵ��ڣ�
%   with size SZ ([height, width]). If any pixels are outside of the image,��СΪSZ��[height��width]����
%   they will replicate the values at the borders.                         ����κ�������ͼ��֮�⣬���ǽ��ڱ߽縴��ֵ��
%
%   Joao F. Henriques, 2014
%   http://www.isr.uc.pt/~henriques/

	if isscalar(sz),  %square sub-window                                   �Ӵ��ڵ�ƽ��
		sz = [sz, sz];
	end
	
	xs = floor(pos(2)) + (1:sz(2)) - floor(sz(2)/2);
	ys = floor(pos(1)) + (1:sz(1)) - floor(sz(1)/2);
	
	%check for out-of-bounds coordinates, and set them to the values at    ��鳬���߽�����꣬������������Ϊ�߽�ֵ
	%the borders
	xs(xs < 1) = 1;
	ys(ys < 1) = 1;
	xs(xs > size(im,2)) = size(im,2);
	ys(ys > size(im,1)) = size(im,1);
	
	%extract image                                                         ��ȡͼ��
	out = im(ys, xs, :);
    %imshow(out)
    %hello()
end

