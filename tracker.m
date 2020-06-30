%读取视频帧
img_path = 'C:/Users/Admin/Desktop/wzy_kcf_matlab/data/Basketball/img/'; 
D = dir([img_path, '*.jpg']);
img_path_list = dir(strcat(img_path, '*.jpg')); % 获取该文件夹中所有jpg格式的图像
img_num = length(img_path_list);    % 获取图像总数量
img_files = cell(1,img_num);   %img_files是图像文件名称的单元阵列。
if img_num > 0 %有满足条件的图像
    for j = 1:img_num %逐一读取图像
        image_name = img_path_list(j).name;% 图像名
        %fprintf('当前找到指定的文件 %s\n', strcat(img_path,image_name));% 显示扫描到的图像路径名
        img_files{j} = image_name;
    end
end


if exist([img_path num2str(1, '%04i.jpg')], 'file'),
    img_files_2 = num2str((1:img_num)', [img_path '%04i.jpg']);
else
    error('No image files found in the directory.');
end


%选择要跟踪的目标
im = imread(img_files_2(1,:));
f = figure('Name', 'Select object to track'); imshow(im);
rect = getrect;
close(f); clear f;
pos = [rect(2)+rect(4)/2 rect(1)+rect(3)/2];  %pos为目标框正中心坐标[y x]
target_sz = [rect(4) rect(3)];  %target_sz为目标框的宽和高[width height]   


%如果目标很大，降低分辨率
resize_image = (sqrt(prod(target_sz)) >= 100);  %对角线大小> =阈值
	if resize_image,
		pos = floor(pos / 2);  %目标尺寸过大就将其缩小1/2
		target_sz = floor(target_sz / 2); %除了显示是用的target_sz，用于计算的都是window_sz
    end
    
%floor向下取整，目标框向外扩展1.5倍作为window_sz
padding = 1.5
window_sz = floor(target_sz * (1 + padding));                          
%后面所有的处理都用window_sz，即包含目标和背景



%创建高斯形状的回归标签，高斯形状，其带宽和目标的尺寸成比例（目标尺寸越大，带宽越宽）
output_sigma_factor = 0.1;  %空间带宽（与目标成比例）
cell_size = 4;
output_sigma = sqrt(prod(target_sz)) * output_sigma_factor / cell_size;%prod计算数组元素的连乘积
%output_sigma 为带宽delta；  cell_size每一个细胞中像素的数量（HOG),若不用HOG则为1
yf = fft2(gaussian_shaped_labels(output_sigma, floor(window_sz / cell_size)));%fft2 2维离散傅里叶变换，yf是频域上的回归值
%imagesc(abs(fftshift(yf)))
%hello()

 
%存储预先计算的余弦窗口
cos_window = hann(size(yf,1)) * hann(size(yf,2))';  %hann汉宁窗，使用yf的尺寸来生成相应的余弦窗
%主体余弦窗，size(yf,1)返回行数，size(yf,2)返回列数
%mesh(cos_window)
%hello()


time = 0;  %为了计算FPS
%初始化n行2列的矩阵用来存放每一帧计算出的位置

positions = zeros(numel(img_files), 2);  %numel(img_files)视频的帧数
features.hog = true;
features.hog_orientations = 9;
kernel.sigma = 0.5;
lambda = 1e-4;  %正则化参数
interp_factor = 0.02;
for frame = 1:numel(img_files),
    
    %读图像
    im = imread([img_path img_files{frame}]); %读取一帧图像

    if size(im,3) > 1,
        im = rgb2gray(im);  %把彩色图转换为灰度图
    end
    if resize_image,
        im = imresize(im, 0.5); %若目标过大，把整幅图变为原来的1/2大小
    end

    tic()  %开始计时，和toc（）配合使用

    if frame > 1,
        %从最后一帧的位置获得用于检测的子窗口，
        %并转换到傅里叶域（其大小不变）
        patch = get_subwindow(im, pos, window_sz);
        zf = fft2(get_features(patch, features, cell_size, cos_window));%zf是测试样本

        %计算分类器对于所有循环位移后的样本的响应

        kzf = gaussian_correlation(zf, model_xf, kernel.sigma);    %通过对测试样本的核变换后得到kzf
        
        response = real(ifft2(model_alphaf .* kzf));  %计算响应
%real->返回实部（虚数），ifft2->反傅里叶变换，model_alphaf->模型，* ->元素点乘

        %目标位置处于最大响应(这些响应周而复始)。
        %我们必须考虑到这样一个事实，如果目标没有移动，
        %峰值将出现在左上角，而不是中心（这在文章中讨论过）。
        [vert_delta, horiz_delta] = find(response == max(response(:)), 1);%找到响应最大的位置
        if vert_delta > size(zf,1) / 2,  %绕到纵轴的负半空间
            vert_delta = vert_delta - size(zf,1);
        end
        if horiz_delta > size(zf,2) / 2,  %与横轴相同
            horiz_delta = horiz_delta - size(zf,2);
        end
        pos = pos + cell_size * [vert_delta - 1, horiz_delta - 1];     %更新出目标的新位置
    end
%if frame>1  的结尾在这

    %在新估计的目标位置获得一个用于训练的子窗口
    patch = get_subwindow(im, pos, window_sz);                         %获取目标的位置和窗口大小
    xf = fft2(get_features(patch, features, cell_size, cos_window));   %用新的结果重新训练分类器

    %内核岭回归，（在傅里叶域）计算alphas(权值)
    
    kf = gaussian_correlation(xf, xf, kernel.sigma);
    %imagesc(abs(fftshift(kf)))
    %hello()
    alphaf = yf ./ (kf + lambda);   %equation for fast training        训练算出每个样本对应的权值

    %更新模板的权值
    if frame == 1,  % 第一帧，用单幅图像训练。
        model_alphaf = alphaf; %第一帧中就直接用训练出的权值和模板
        model_xf = xf;
    else
        %后续帧，插值模型
        model_alphaf = (1 - interp_factor) * model_alphaf + interp_factor * alphaf;%后续帧中的更新使用本帧和前一帧中结果的加权
        model_xf = (1 - interp_factor) * model_xf + interp_factor * xf;
    end                           %model_xf 上一帧 ，  interp_factor * xf 这一帧的  

    %保存位置和时间
    positions(frame,:) = pos; %保存每一帧中的目标位置
    time = time + toc();   %保存处理所耗的时间

    %将每一帧的结果显示出来
    text_str = ['Frame: ' num2str(frame)];
    box_color = 'green';
    box = [pos([2,1]) - target_sz([2,1])/2, target_sz([2,1])];  
    result = insertShape(im, 'Rectangle', box, 'LineWidth', 3);
    imshow(result);

 
end%  和第80行的for 组成一个end of for 循环

