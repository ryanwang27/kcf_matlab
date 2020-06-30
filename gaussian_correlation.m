function kf = gaussian_correlation(xf, yf, sigma)
%GAUSSIAN_CORRELATION Gaussian Kernel at all shifts, i.e. kernel correlation.GAUSSIAN_CORRELATION������λ�ĸ�˹�ںˣ����ں���ء�
%   Evaluates a Gaussian kernel with bandwidth SIGMA for all relative      �������ͼ��X��Y֮����������λ�ƣ�
%   shifts between input images X and Y, which must both be MxN. They must �������д���SIGMA�ĸ�˹�ںˣ�����붼��M��N��
%   also be periodic (ie., pre-processed with a cosine window). The result ����Ҳ�����������Եģ����������Ҵ�Ԥ������
%   is an MxN map of responses.                                            �����һ��MxN��Ӧͼ��
%
%   Inputs and output are all in the Fourier domain.                       �����������ڸ���Ҷ��
%
%   Joao F. Henriques, 2014
%   http://www.isr.uc.pt/~henriques/
	 
	N = size(xf,1) * size(xf,2);
	xx = xf(:)' * xf(:) / N;  %squared norm of x                           x��ƽ���淶
	yy = yf(:)' * yf(:) / N;  %squared norm of y                           y��ƽ���淶
	
	%cross-correlation term in Fourier domain                              ����Ҷ���еĻ������
	xyf = xf .* conj(yf);
	xy = sum(real(ifft2(xyf)), 3);  %to spatial domain                     ת�����ռ���
	
	%calculate gaussian response for all positions, then go back to the    ��������λ�õĸ�˹��Ӧ��Ȼ�󷵻ظ���Ҷ��
	%Fourier domain
	kf = fft2(exp(-1 / sigma^2 * max(0, (xx + yy - 2 * xy) / numel(xf))));
    %imagesc(abs(fftshift(kf)))
    %hello()
end

