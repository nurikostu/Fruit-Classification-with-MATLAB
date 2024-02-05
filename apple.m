    rgbImage = imread('apple.jpg');
    
    hsvImage = rgb2hsv(rgbImage);
    
    hueComponent = hsvImage(:, :, 1);
    backgroundSubtracted = imopen(hueComponent, strel('disk', 15));
    threshold = graythresh(backgroundSubtracted);
    binaryROI = imbinarize(backgroundSubtracted, threshold);
    filledROI = imfill(binaryROI, 'holes');
    roiImage = rgbImage;
    roiImage(repmat(~filledROI, [1, 1, 3])) = 0;
    
    % Display the results
    figure;
    subplot(2, 3, 1); imshow(rgbImage); title('Original Image');
    subplot(2, 3, 2); imshow(hueComponent); title('Hue Component');
    subplot(2, 3, 3); imshow(backgroundSubtracted); title('Background Subtracted');
    subplot(2, 3, 4); imshow(binaryROI); title('Binary ROI');
    subplot(2, 3, 5);imshow(filledROI); title('Filled ROI');
    subplot(2, 3, 6);imshow(roiImage); title('Image with ROI');

    grayImage = rgb2gray(roiImage);
    [LL, LH, HL, HH] = dwt2(grayImage, 'haar');
    statisticalFeatures = [mean2(LL), std2(LL), entropy(LL)];
    textureFeatures = graycomatrix(LL);
    textureFeatures = reshape(textureFeatures, 1, []);
    allFeatures = [statisticalFeatures, textureFeatures];
    label = 'apple';
    disp('Statistical Features:');
    disp(statisticalFeatures);
    disp('Texture Features:');
    disp(textureFeatures);
    disp('All Features:');
    disp(allFeatures);
    inputData = allFeatures;
    labels = cellstr(repmat(label, size(inputData, 1), 1));
    svmModel = fitcsvm(inputData, labels);

    new = imread('apple4.jpg');
    
    newhsvImage = rgb2hsv(new);
    
    hueComponent2 = newhsvImage(:, :, 1);
    backgroundSubtracted2 = imopen(hueComponent2, strel('disk', 15));
    threshold2 = graythresh(backgroundSubtracted2);
    binaryROI2 = imbinarize(backgroundSubtracted2, threshold2);
    filledROI2 = imfill(binaryROI2, 'holes');
    roiImage2 = new;
    roiImage2(repmat(~filledROI2, [1, 1, 3])) = 0;
    newBW = rgb2gray(roiImage2);
    [LL_new, LH_new, HL_new, HH_new] = dwt2(newBW, 'haar');
    statisticalFeatures2 = [mean2(LL_new), std2(LL_new), entropy(LL_new)];
    textureFeatures2 = graycomatrix(LL_new);
    textureFeatures2 = reshape(textureFeatures2, 1, []);
    allFeatures2 = [statisticalFeatures2, textureFeatures2];
    predictedLabel = predict(svmModel, allFeatures2);
    disp(['Predicted Label: ', char(predictedLabel)]);
    