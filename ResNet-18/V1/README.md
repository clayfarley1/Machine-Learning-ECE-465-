# Resistor Identification Program - MATLAB

A deep learning-based system for identifying resistor values from images using color band recognition.

## Requirements

- MATLAB R2019b or later
- Deep Learning Toolbox
- Computer Vision Toolbox
- Image Processing Toolbox
- (Optional) Parallel Computing Toolbox for GPU acceleration

## Dataset Organization

Organize your resistor images into folders by their values:

```
resistor_dataset/
├── 10ohm/
│   ├── img001.jpg
│   ├── img002.jpg
│   └── ...
├── 100ohm/
│   ├── img001.jpg
│   └── ...
├── 1kohm/
│   └── ...
└── ...
```

Each folder name represents a resistor value/class, and should contain multiple images of that resistor type.

## Setup Instructions

1. **Prepare Your Dataset**
   - Organize images as shown above
   - Recommended: At least 50-100 images per resistor value
   - Images should be clear, well-lit, and show the color bands

2. **Update File Paths**
   - Open `train_resistor_model.m`
   - Update line 11: `datasetPath = 'path/to/your/resistor/images';`
   - Open `predict_resistor.m`
   - Update line 14: `imagePath = 'path/to/test/resistor/image.jpg';`

## Usage

### Training the Model

1. Open MATLAB
2. Navigate to the project directory
3. Run the training script:
   ```matlab
   train_resistor_model
   ```

4. The script will:
   - Load and split your dataset (80% training, 20% validation)
   - Apply data augmentation
   - Train a ResNet-18 model using transfer learning
   - Display training progress
   - Show validation accuracy and confusion matrix
   - Save the trained model as `resistor_identification_model.mat`

### Making Predictions

**Single Image Prediction:**
```matlab
predict_resistor
```

**Batch Prediction:**
Uncomment the batch prediction section in `predict_resistor.m` (lines 36-59) and update the test folder path.

**Webcam Real-time Prediction:**
Uncomment the webcam section (lines 61-82) for live predictions.

## Training Parameters

You can adjust these in `train_resistor_model.m`:

- `initialLearningRate`: 0.001 (default)
- `maxEpochs`: 20 (increase for better accuracy, but watch for overfitting)
- `miniBatchSize`: 32 (adjust based on GPU memory)
- Input size: 224×224 (standard for ResNet-18)

## Performance Tips

1. **GPU Acceleration**: Ensure MATLAB can access your GPU for faster training
   ```matlab
   gpuDevice % Check GPU availability
   ```

2. **More Data**: Collect more images per class for better accuracy

3. **Data Augmentation**: Already enabled - helps model generalize better

4. **Hyperparameter Tuning**: Experiment with learning rate and epochs

5. **Different Architectures**: Try ResNet-50 or GoogLeNet for potentially better accuracy:
   ```matlab
   net = resnet50; % Instead of resnet18
   ```

## Expected Accuracy

- With good quality images: 85-95% accuracy
- Factors affecting accuracy:
  - Image quality and lighting
  - Number of training images
  - Similarity between resistor values
  - Consistency in color bands

## Troubleshooting

**Out of Memory Error:**
- Reduce `miniBatchSize` to 16 or 8
- Use a smaller network (ResNet-18 instead of ResNet-50)
- Reduce image input size

**Low Accuracy:**
- Collect more training images
- Ensure images are well-lit and clear
- Check that dataset is balanced (similar number of images per class)
- Increase `maxEpochs`

**GPU Not Being Used:**
- Check GPU availability: `gpuDevice`
- Install MATLAB GPU support packages
- Set `'ExecutionEnvironment', 'gpu'` explicitly in training options

## File Descriptions

- `train_resistor_model.m` - Main training script
- `predict_resistor.m` - Prediction script for trained model
- `preprocessResistorImage.m` - Image preprocessing helper function
- `resistor_identification_model.mat` - Saved trained model (generated after training)
- `model_config.mat` - Model configuration (generated after training)

## Next Steps

1. Collect and organize your resistor image dataset
2. Run the training script
3. Evaluate the model performance
4. Use the prediction script on new images
5. Fine-tune parameters if needed

## Advanced: Color Band Detection

For more advanced resistor identification that reads individual color bands:

1. Implement color band segmentation
2. Extract each band's color
3. Map colors to resistor value code
4. Calculate resistance value

This would require additional image processing beyond classification.
