function preprocessedImg = preprocessResistorImage(img, targetSize)
% PREPROCESSRESISTORIMAGE Preprocess resistor images for better recognition
%   This function applies various preprocessing steps to enhance resistor
%   color band visibility and improve model accuracy
%
%   Inputs:
%       img - Input image (RGB or grayscale)
%       targetSize - [height, width] target size for resizing
%
%   Output:
%       preprocessedImg - Preprocessed image ready for model input

    % Convert to RGB if grayscale
    if size(img, 3) == 1
        img = cat(3, img, img, img);
    end
    
    % Resize image
    img = imresize(img, targetSize);
    
    % Apply contrast enhancement
    if size(img, 3) == 3
        % Convert to LAB color space for better color preservation
        lab = rgb2lab(img);
        
        % Enhance luminosity channel
        L = lab(:,:,1);
        L = adapthisteq(L / 100) * 100;
        lab(:,:,1) = L;
        
        % Convert back to RGB
        preprocessedImg = lab2rgb(lab);
    else
        preprocessedImg = adapthisteq(img);
    end
    
    % Optional: Apply denoising
    preprocessedImg = imgaussfilt(preprocessedImg, 0.5);
    
    % Ensure proper data type
    if ~isa(preprocessedImg, 'uint8')
        preprocessedImg = im2uint8(preprocessedImg);
    end
end
