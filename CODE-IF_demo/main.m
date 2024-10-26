close all; clear;
addpath(genpath('.\dataset')); 
addpath(genpath('.\function'));
rng(1);
normColor = @(R)max(min((R-mean(R(:)))/std(R(:)),2),-2)/3+0.5;

%% Select the Mode (Simulated Data: mode=1; Real Data: mode=2)
mode = 1;

if mode == 1
    %% Load Related data
    load Chikusei.mat; ratio = 4; %Sampling Factor
    [GT,MAX,MIN] = normalize(GT);
    load SRF_Chikusei.mat; %SRF
    PSF = fspecial('gaussian',[ratio,ratio],2); %PSF
    %Permu=simple_permutation(Ym,ratio);
    load perm_Chikusei; %Permutation Matrix

    %% Small Data Learning (DE)
    system(['activate & conda activate env & python function/test_Chikusei.py']);
    load Small_Data_Result/Chikusei.mat;
    Z_DE = double(Z_DE);

    %% CODE-IF Algorithm (CO)
    tic;
    [Z_CODE] = CODE_IF(Yh,Ym,D,Z_DE,Permu,PSF,ratio);
    toc;

    %% Quantitative Measurement
    QI_Z_DE = QualityIndices(Z_DE,GT,ratio, MAX, MIN);
    QI_Z = QualityIndices(Z_CODE,GT,ratio, MAX, MIN);

    %% Show Results
    figure;
    subplot(2,2,1); imshow(normColor(Ym(:,:,[3, 2, 1]))); title("Ym");
    subplot(2,2,2); imshow(normColor(Yh(:,:,[55, 35, 11]))); title("Yh");
    subplot(2,2,3); imshow(normColor(Z_DE(:,:,[55, 35, 11]))); title("Z_{DE}");
    xlabel(sprintf("PSNR=%0.4f\nSAM=%0.4f\nRMSE=%0.4f\nERGAS=%0.4f",QI_Z_DE.psnr,QI_Z_DE.sam,QI_Z_DE.rmse,QI_Z_DE.ergas));
    subplot(2,2,4); imshow(normColor(Z_CODE(:,:,[55, 35, 11]))); title("Z");
    xlabel(sprintf("PSNR=%0.4f\nSAM=%0.4f\nRMSE=%0.4f\nERGAS=%0.4f",QI_Z.psnr,QI_Z.sam,QI_Z.rmse,QI_Z.ergas));

elseif mode==2
    %% Load Related data
    load Houston.mat; ratio = 20; %Sampling Factor
    load SRF_Houston.mat; %SRF
    PSF = fspecial('gaussian',[ratio,ratio],2); %PSF
    %Permu=simple_permutation(Ym,ratio);
    load perm_Houston; %Permutation Matrix

    %% Small Data Learning (DE)
    system(['activate & conda activate env & python function/test_Houston.py']);
    load Small_Data_Result/Houston.mat;
    Z_DE = double(Z_DE);

    %% CODE-IF Algorithm (CO)
    tic;
    [Z_CODE] = CODE_IF(Yh,Ym,D,Z_DE,Permu,PSF,ratio);
    toc;

    %% Show Results
    figure;
    imshow(normColor(Ym(:,:,[3, 2, 1]))); title("Ym");

    figure;
    subplot(2,2,1); imshow(Yh(:,:,6),[]); title("Yh (band 6)");
    subplot(2,2,2); imshow(Yh(:,:,13),[]); title("Yh (band 13)");
    subplot(2,2,3); imshow(Z_CODE(:,:,6),[]); title("Z (band 6)");
    subplot(2,2,4); imshow(Z_CODE(:,:,13),[]); title("Z (band 13)");

end


