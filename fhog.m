function H = fhog( I, binSize, nOrients, clip, crop )
% Efficiently compute Felzenszwalb's HOG (FHOG) features.                  ��Ч����Felzenszwalb��HOG��FHOG��������
%
% A fast implementation of the HOG variant used by Felzenszwalb et al.     Felzenszwalb���������ǵĹ����У��Կ���ʵ�ֵ�HOG����
% in their work on discriminatively trained deformable part models.        �����˿ɱ������ģ���������ѵ��
%  http://www.cs.berkeley.edu/~rbg/latent/index.html
% Gives nearly identical results to features.cc in code release version 5  �ڴ��뷢���汾5�л����features.cc������ͬ�Ľ����
% but runs 4x faster (over 125 fps on VGA color images).                   �������ٶ������4������VGA��ɫͼ���ϳ���125 fps��
%
% The computed HOG features are 3*nOrients+5 dimensional. There are        �����HOG������3 * nOrients + 5ά��
% 2*nOrients contrast sensitive orientation channels, nOrients contrast    ��2 * nOrients�Ա����ж�λͨ����
% insensitive orientation channels, 4 texture channels and 1 all zeros     nOrients�ԱȲ����ж���ͨ����
% channel (used as a 'truncation' feature). Using the standard value of    4������ͨ����1��ȫ��ͨ��������'�ض�'���ܣ���
% nOrients=9 gives a 32 dimensional feature vector at each cell. This      ʹ��nOrients�ı�׼ֵ= 9������ÿ����Ԫ��32ά����������
% variant of HOG, refered to as FHOG, has been shown to achieve superior   HOG�����ֱ��壬����ΪFHOG���ѱ�֤������ʵ������ԭʼHOG���������ܡ�
% performance to the original HOG features. For details please refer to    ���������Felzenszwalb���˵Ĺ�����
% work by Felzenszwalb et al. (see link above).
%
% This function is essentially a wrapper for calls to gradientMag()        ��������������ǵ���gradientMag������gradientHist�����İ�װ����
% and gradientHist(). Specifically, it is equivalent to the following:     ������ԣ����൱���������ݣ�
%  [M,O] = gradientMag( I,0,0,0,1 ); softBin = -1; useHog = 2;             [M��O] = gradientMag��I��0,0,0,1��; softBin = -1;useHog = 2;
%  H = gradientHist(M,O,binSize,nOrients,softBin,useHog,clip);             H = gradientHist��M��O��binSize��nOrients��softBin��useHog��clip��; 
% See gradientHist() for more general usage.                               �йظ��ೣ���÷��������gradientHist������
%
% This code requires SSE2 to compile and run (most modern Intel and AMD    �˴���Ҫ��SSE2��������У�������ִ�Ӣ�ض���AMD������֧��SSE2����
% processors support SSE2). Please see: http://en.wikipedia.org/wiki/SSE2.
%
% USAGE
%  H = fhog( I, [binSize], [nOrients], [clip], [crop] )
%
% INPUTS
%  I        - [hxw] color or grayscale input image (must have type single) ��ɫ��Ҷ�����ͼ�񣨱����е�һ�����ͣ�
%  binSize  - [8] spatial bin size                                         �ռ�ִ�С
%  nOrients - [9] number of orientation bins                               �����������
%  clip     - [.2] value at which to clip histogram bins                   ���ڼ���ֱ��ͼ���ֵ
%  crop     - [0] if true crop boundaries                                  ������棬���߽�
%
% OUTPUTS
%  H        - [h/binSize w/binSize nOrients*3+5] computed hog features     �����hog����
%
% EXAMPLE
%  I=imResample(single(imread('peppers.png'))/255,[480 640]);
%  tic, for i=1:100, H=fhog(I,8,9); end; disp(100/toc) % >125 fps
%  figure(1); im(I); V=hogDraw(H,25,1); figure(2); im(V)
%
% EXAMPLE
%  % comparison to features.cc (requires DPM code release version 5)
%  I=imResample(single(imread('peppers.png'))/255,[480 640]); Id=double(I);
%  tic, for i=1:100, H1=features(Id,8); end; disp(100/toc)
%  tic, for i=1:100, H2=fhog(I,8,9,.2,1); end; disp(100/toc)
%  figure(1); montage2(H1); figure(2); montage2(H2);
%  D=abs(H1-H2); mean(D(:))
%
% See also hog, hogDraw, gradientHist
%
% Piotr's Image&Video Toolbox      Version 3.23
% Copyright 2013 Piotr Dollar.  [pdollar-at-caltech.edu]
% Please email me if you find bugs, or have suggestions or questions!
% Licensed under the Simplified BSD License [see external/bsd.txt]

%Note: modified to be more self-contained

if( nargin<2 ), binSize=8; end
if( nargin<3 ), nOrients=9; end
if( nargin<4 ), clip=.2; end
if( nargin<5 ), crop=0; end

softBin = -1; useHog = 2; b = binSize;

[M,O]=gradientMex('gradientMag',I,0,1);

H = gradientMex('gradientHist',M,O,binSize,nOrients,softBin,useHog,clip);

if( crop ), e=mod(size(I),b)<b/2; H=H(2:end-e(1),2:end-e(2),:); end

end
