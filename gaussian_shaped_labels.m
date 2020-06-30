function labels = gaussian_shaped_labels(sigma, sz)
%GAUSSIAN_SHAPED_LABELS
%   Gaussian-shaped labels for all shifts of a sample.                     ������������λ�Ƶĸ�˹�α�ǩ
%
%   LABELS = GAUSSIAN_SHAPED_LABELS(SIGMA, SZ)
%   Creates an array of labels (regression targets) for all shifts of a    Ϊ�ߴ�ΪSZ��������������λ����һ���ǩ���ع�Ŀ�꣩��
%   sample of dimensions SZ. The output will have size SZ, representing    ����ĳߴ�ΪSZ������ÿ�����ܵİ�ε�һ����ǩ��
%   one label for each possible shift. The labels will be Gaussian-shaped, ��ǩ��Ϊ��˹��״����ֵΪ0��λ�����е����Ͻ�Ԫ�أ���
%   with the peak at 0-shift (top-left element of the array), decaying     ��������Ӷ�˥�������ڱ߽紦���ơ�
%   as the distance increases, and wrapping around at the borders.
%   The Gaussian function has spatial bandwidth SIGMA.                     ��˹�������пռ����SIGMA��
%
%   Joao F. Henriques, 2014
%   http://www.isr.uc.pt/~henriques/


% 	%as a simple example, the limit sigma = 0 would be a Dirac delta,
% 	%instead of a Gaussian:
% 	labels = zeros(sz(1:2));  %labels for all shifted samples              ������λ�����ı�ǩ
% 	labels(1,1) = magnitude;  %label for 0-shift (original sample)         0�ƶ��ı�ǩ��ԭʼ����) 
	

	%evaluate a Gaussian with the peak at the center element               ������Ԫ�ش��ķ�ֵ������˹
	[rs, cs] = ndgrid((1:sz(1)) - floor(sz(1)/2), (1:sz(2)) - floor(sz(2)/2));
	labels = exp(-0.5 / sigma^2 * (rs.^2 + cs.^2));
    %mesh(labels);
   
	%move the peak to the top-left, with wrap-around                       ���嶥�������Ͻǣ������л���
	labels = circshift(labels, -floor(sz(1:2) / 2) + 1);
    %figure
    %mesh(labels);
    %hello()
	%sanity check: make sure it's really at top-left                       �����Լ�飺ȷ������������Ͻ�
	assert(labels(1,1) == 1)

end

