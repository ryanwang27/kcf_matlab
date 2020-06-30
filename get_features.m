function x = get_features(im, features, cell_size, cos_window)
%GET_FEATURES
%   Extracts dense features from image.                                    ��ͼ������ȡ�ܼ���������
%
%   X = GET_FEATURES(IM, FEATURES, CELL_SIZE)
%   Extracts features specified in struct FEATURES, from image IM. The     ��ͼ��IM����ȡstruct FEATURES��ָ����������
%   features should be densely sampled, in cells or intervals of CELL_SIZE.��Щ����Ӧ����CELL_SIZE�ĵ�Ԫ����������ܼ�������
%   The output has size [height in cells, width in cells, features].       ������д�С[��Ԫ��߶ȣ���Ԫ���ȣ�����]��
%
%   To specify HOG features, set field 'hog' to true, and                  Ҫָ��HOG�������뽫�ֶΡ�hog������Ϊtrue��
%   'hog_orientations' to the number of bins.                              ������hog_orientations������Ϊ��������
%
%   To experiment with other features simply add them to this function     Ҫ�����������ܣ�ֻ�轫������ӵ��˹����У�
%   and include any needed parameters in the FEATURES struct. To allow     ����FEATURES�ṹ�а���������κβ�����
%   combinations of features, stack them with x = cat(3, x, new_feat).     Ҫ���������������ʹ��x = cat��3��x��new_feat�������Ƕѵ�������
%
%   Joao F. Henriques, 2014
%   http://www.isr.uc.pt/~henriques/



    %HOG features, from Piotr's Toolbox                                HOG���ܣ�����Piotr�Ĺ�����
    x = double(fhog(single(im) / 255, cell_size, features.hog_orientations));
    x(:,:,end) = [];  %remove all-zeros channel ("truncation feature") ɾ��ȫ��ͨ�������ضϹ��ܡ���


	
	%process with cosine window if needed                                  �����Ҫ�������Ҵ��ڴ���
	if ~isempty(cos_window),
		x = bsxfun(@times, x, cos_window);
	end
	
end
