# Malaria Parasite Detection

The project aims to detect malaria parasite in thin blood scans using machine learning. Model architectures like VGG16, InceptionV3, Inception-ResnetV2, Resnet50 and HHNA-Net (recent architecture developed in 2022) were explored. Data pre-processing steps included image resizing, image contrast enhancement & brightness rectification and normalization. Data Augmentation was done using horizontal & vertical flip, height & width shift, shear and zoom.

The project consists of 5 files:
- Config.py : This file contains all the project variables and parameters.
- Architectures.py : This file contains different model architectures discussed above.
- Data_generation.py : This file contains functions related to the train and validation generator.
- Utilities.py : This file contains code for performing performance evaluation, graph plotting and enhancements related to code readability.
- Main.py : This is the main file. It downloads the data and performs model training.

The project has been designed to be a one-shot solution for model training and evaluation (after your preferences have been added to the config file) and thus does not require human intervention or interaction once initiated. This makes it most suited for High Performance Computing (HPC) environments. For environments like single GPU/Google Colab, the main file will need to be modified to break processing time into smaller chunks.

While all models performed reasonably well, HHNA performed the best with an F1-score of 0.972 and a Recall of 0.981 when averaged over 5 trials. Techniques like hyper-parameter tuning, data augmentation (using ImageDataGenerator and GANs) and more exotic model architecture design could be used to further improve performance.

To initiate, add your preferences to the config.py file and then run main.py.
