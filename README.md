# Micro Expression Recognition Using Local Binary Pattern (LBP)

![icon_diagram](https://user-images.githubusercontent.com/48756138/172045399-62a1a1d3-0b19-4f6c-a251-b65d98f0c9d8.png)


## Algorithms Implemented
  - Eigenfaces
  - Localbinary Pattern Histograms[LBPH]
  - Fisherfaces

# How to use?
 1. Download miniconda/anaconda.
 2. Create environment.
 3. Installation.	
 4. Clone repository.	
 5. Execute.

### 1. Download
 - Download [Mininconda](https://conda.io/miniconda.html).
 - Download [Anaconda](https://www.anaconda.com/).

### 2. Create Environment
 - ```$ conda create -n cv python=3.*```
 - ```$ conda activate cv```

### 3. Package Installation
 - ```$ conda install pyqt=5.*```
 - ```$ conda install opencv=*.*```
 - ```$ conda install -c michael_wild opencv-contrib```

### 4. Execute Application
 - Execute  ```$ python main.py```

	Note:Generate atleat two datasets to work properly.
  
  1. Enter name,and unique key.
  2. Check algorithm radio button which you want to train.
  3. Click recognize button.
  4. Click save button to save current displayed image.
  5. Click record button to save video.

## Resources
  - [OpenCV face Recognition](https://docs.opencv.org/2.4/modules/contrib/doc/facerec/facerec_tutorial.html)
  - [PyQt5 Documentation](http://pyqt.sourceforge.net/Docs/PyQt5/)

## LBP reference
https://www.researchgate.net/publication/220939185_Iris_Extraction_Based_on_Intensity_Gradient_and_Texture_Difference
