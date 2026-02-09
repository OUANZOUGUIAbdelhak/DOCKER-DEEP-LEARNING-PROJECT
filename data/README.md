# Dataset Instructions

## Downloading the Dataset

This project uses the **CIFAR-100** dataset for image classification.

### Kaggle Dataset URL

**https://www.kaggle.com/datasets/pankrzysiu/cifar100**

### Download Instructions

1. **Option 1: Using Kaggle API (Recommended)**
   ```bash
   # Install Kaggle API (if not already installed)
   pip install kaggle
   
   # Set up Kaggle credentials (place kaggle.json in ~/.kaggle/)
   # Download dataset
   kaggle datasets download -d pankrzysiu/cifar100
   
   # Extract to data/raw/
   unzip cifar100.zip -d data/raw/
   ```

2. **Option 2: Manual Download**
   - Visit the Kaggle dataset URL above
   - Click "Download" button
   - Extract the downloaded ZIP file to `data/raw/` directory

### Expected Directory Structure

After downloading and extracting, your `data/raw/` directory should look like:

```
data/raw/
├── train/
│   ├── class1/
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   └── ...
│   ├── class2/
│   │   └── ...
│   └── ...
└── test/
    ├── class1/
    ├── class2/
    └── ...
```

### Dataset Information

- **Name**: CIFAR-100
- **Task**: Image Classification
- **Classes**: 100 fine-grained classes
- **Training Samples**: 50,000 images
- **Test Samples**: 10,000 images
- **Image Size**: 32x32 pixels
- **Channels**: RGB (3 channels)

### Alternative Datasets

If you prefer a different dataset, you can use any image classification dataset with:
- At least 5,000+ samples
- Organized in class folders
- Common image formats (JPG, PNG)

Some alternatives:
- **CIFAR-10**: https://www.kaggle.com/datasets/pankrzysiu/cifar10
- **Flowers Dataset**: https://www.kaggle.com/datasets/alxmamaev/flowers-recognition
- **Animals Dataset**: https://www.kaggle.com/datasets/andrewmvd/animal-faces

### Notes

- The dataset files are NOT included in the repository (see `.gitignore`)
- You must manually download and place the dataset in `data/raw/`
- The data loader will automatically detect the directory structure
