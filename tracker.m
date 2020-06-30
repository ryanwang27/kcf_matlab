%��ȡ��Ƶ֡
img_path = 'C:/Users/Admin/Desktop/wzy_kcf_matlab/data/Basketball/img/'; 
D = dir([img_path, '*.jpg']);
img_path_list = dir(strcat(img_path, '*.jpg')); % ��ȡ���ļ���������jpg��ʽ��ͼ��
img_num = length(img_path_list);    % ��ȡͼ��������
img_files = cell(1,img_num);   %img_files��ͼ���ļ����Ƶĵ�Ԫ���С�
if img_num > 0 %������������ͼ��
    for j = 1:img_num %��һ��ȡͼ��
        image_name = img_path_list(j).name;% ͼ����
        %fprintf('��ǰ�ҵ�ָ�����ļ� %s\n', strcat(img_path,image_name));% ��ʾɨ�赽��ͼ��·����
        img_files{j} = image_name;
    end
end


if exist([img_path num2str(1, '%04i.jpg')], 'file'),
    img_files_2 = num2str((1:img_num)', [img_path '%04i.jpg']);
else
    error('No image files found in the directory.');
end


%ѡ��Ҫ���ٵ�Ŀ��
im = imread(img_files_2(1,:));
f = figure('Name', 'Select object to track'); imshow(im);
rect = getrect;
close(f); clear f;
pos = [rect(2)+rect(4)/2 rect(1)+rect(3)/2];  %posΪĿ�������������[y x]
target_sz = [rect(4) rect(3)];  %target_szΪĿ���Ŀ�͸�[width height]   


%���Ŀ��ܴ󣬽��ͷֱ���
resize_image = (sqrt(prod(target_sz)) >= 100);  %�Խ��ߴ�С> =��ֵ
	if resize_image,
		pos = floor(pos / 2);  %Ŀ��ߴ����ͽ�����С1/2
		target_sz = floor(target_sz / 2); %������ʾ���õ�target_sz�����ڼ���Ķ���window_sz
    end
    
%floor����ȡ����Ŀ���������չ1.5����Ϊwindow_sz
padding = 1.5
window_sz = floor(target_sz * (1 + padding));                          
%�������еĴ�����window_sz��������Ŀ��ͱ���



%������˹��״�Ļع��ǩ����˹��״��������Ŀ��ĳߴ�ɱ�����Ŀ��ߴ�Խ�󣬴���Խ��
output_sigma_factor = 0.1;  %�ռ������Ŀ��ɱ�����
cell_size = 4;
output_sigma = sqrt(prod(target_sz)) * output_sigma_factor / cell_size;%prod��������Ԫ�ص����˻�
%output_sigma Ϊ����delta��  cell_sizeÿһ��ϸ�������ص�������HOG),������HOG��Ϊ1
yf = fft2(gaussian_shaped_labels(output_sigma, floor(window_sz / cell_size)));%fft2 2ά��ɢ����Ҷ�任��yf��Ƶ���ϵĻع�ֵ
%imagesc(abs(fftshift(yf)))
%hello()

 
%�洢Ԥ�ȼ�������Ҵ���
cos_window = hann(size(yf,1)) * hann(size(yf,2))';  %hann��������ʹ��yf�ĳߴ���������Ӧ�����Ҵ�
%�������Ҵ���size(yf,1)����������size(yf,2)��������
%mesh(cos_window)
%hello()


time = 0;  %Ϊ�˼���FPS
%��ʼ��n��2�еľ����������ÿһ֡�������λ��

positions = zeros(numel(img_files), 2);  %numel(img_files)��Ƶ��֡��
features.hog = true;
features.hog_orientations = 9;
kernel.sigma = 0.5;
lambda = 1e-4;  %���򻯲���
interp_factor = 0.02;
for frame = 1:numel(img_files),
    
    %��ͼ��
    im = imread([img_path img_files{frame}]); %��ȡһ֡ͼ��

    if size(im,3) > 1,
        im = rgb2gray(im);  %�Ѳ�ɫͼת��Ϊ�Ҷ�ͼ
    end
    if resize_image,
        im = imresize(im, 0.5); %��Ŀ����󣬰�����ͼ��Ϊԭ����1/2��С
    end

    tic()  %��ʼ��ʱ����toc�������ʹ��

    if frame > 1,
        %�����һ֡��λ�û�����ڼ����Ӵ��ڣ�
        %��ת��������Ҷ�����С���䣩
        patch = get_subwindow(im, pos, window_sz);
        zf = fft2(get_features(patch, features, cell_size, cos_window));%zf�ǲ�������

        %�����������������ѭ��λ�ƺ����������Ӧ

        kzf = gaussian_correlation(zf, model_xf, kernel.sigma);    %ͨ���Բ��������ĺ˱任��õ�kzf
        
        response = real(ifft2(model_alphaf .* kzf));  %������Ӧ
%real->����ʵ������������ifft2->������Ҷ�任��model_alphaf->ģ�ͣ�* ->Ԫ�ص��

        %Ŀ��λ�ô��������Ӧ(��Щ��Ӧ�ܶ���ʼ)��
        %���Ǳ��뿼�ǵ�����һ����ʵ�����Ŀ��û���ƶ���
        %��ֵ�����������Ͻǣ����������ģ��������������۹�����
        [vert_delta, horiz_delta] = find(response == max(response(:)), 1);%�ҵ���Ӧ����λ��
        if vert_delta > size(zf,1) / 2,  %�Ƶ�����ĸ���ռ�
            vert_delta = vert_delta - size(zf,1);
        end
        if horiz_delta > size(zf,2) / 2,  %�������ͬ
            horiz_delta = horiz_delta - size(zf,2);
        end
        pos = pos + cell_size * [vert_delta - 1, horiz_delta - 1];     %���³�Ŀ�����λ��
    end
%if frame>1  �Ľ�β����

    %���¹��Ƶ�Ŀ��λ�û��һ������ѵ�����Ӵ���
    patch = get_subwindow(im, pos, window_sz);                         %��ȡĿ���λ�úʹ��ڴ�С
    xf = fft2(get_features(patch, features, cell_size, cos_window));   %���µĽ������ѵ��������

    %�ں���ع飬���ڸ���Ҷ�򣩼���alphas(Ȩֵ)
    
    kf = gaussian_correlation(xf, xf, kernel.sigma);
    %imagesc(abs(fftshift(kf)))
    %hello()
    alphaf = yf ./ (kf + lambda);   %equation for fast training        ѵ�����ÿ��������Ӧ��Ȩֵ

    %����ģ���Ȩֵ
    if frame == 1,  % ��һ֡���õ���ͼ��ѵ����
        model_alphaf = alphaf; %��һ֡�о�ֱ����ѵ������Ȩֵ��ģ��
        model_xf = xf;
    else
        %����֡����ֵģ��
        model_alphaf = (1 - interp_factor) * model_alphaf + interp_factor * alphaf;%����֡�еĸ���ʹ�ñ�֡��ǰһ֡�н���ļ�Ȩ
        model_xf = (1 - interp_factor) * model_xf + interp_factor * xf;
    end                           %model_xf ��һ֡ ��  interp_factor * xf ��һ֡��  

    %����λ�ú�ʱ��
    positions(frame,:) = pos; %����ÿһ֡�е�Ŀ��λ��
    time = time + toc();   %���洦�����ĵ�ʱ��

    %��ÿһ֡�Ľ����ʾ����
    text_str = ['Frame: ' num2str(frame)];
    box_color = 'green';
    box = [pos([2,1]) - target_sz([2,1])/2, target_sz([2,1])];  
    result = insertShape(im, 'Rectangle', box, 'LineWidth', 3);
    imshow(result);

 
end%  �͵�80�е�for ���һ��end of for ѭ��

