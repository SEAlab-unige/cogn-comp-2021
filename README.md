# Design and Deployment of an Image Polarity Detector with Visual Attention
This repository concerns the experimental campaign of the paper **"Design and Deployment of an Image Polarity Detector with Visual Attention"** by Ragusa et al., [DOI](https://doi.org/10.1007/s12559-021-09829-6)

## Authors
* Edoardo Ragusa: edoardo.ragusa@edu.unige.it
* Tommaso Apicella: tommaso.apicella@edu.unige.it
* Christian Gianoglio: christian.gianoglio@edu.unige.it
* Rodolfo Zunino: rodolfo.zunino@unige.it
* Paolo Gastaldo: paolo.gastaldo@unige.it

## Table of contents
* [Python project](#python-project)
* [Android application](#android-application)
* [Reference](#reference)

## Python project
### Requirements
The requirements to run the python code are the following:
* Python 3.6
* Tensorflow 1.14
* Tensorflow Slim
* Tensorflow Object Detection API
* Numpy
* OpenCV
* Pillow
* scikit-learn

### Description
The **pythonProject** enables to run the algorithm described in the **paper** on a test set.
The module consists of 3 files:
* `box_detection.py`: methods to perform object detection for a single image. These functions can return boxes images
  and/or boxes coordinates with a score greater than threshold (in some cases set to 0.3) .
* `write_file.py`: methods to clear a file, write a text, write accuracy or confusion matrix into a text file.
* `polarity_detection_visual_attention.py`: this script employs the algorithm described in the **paper** to detect image polarity using visual attention.

**Model checkpoints** are available:
* `mobilenet_v1_1.0_224_anp40`: entire image model.
* `mobilenet_v1_1.0_224_anp40_patches`: patches model.
* `ssd_mobilenet_v1_ILSVRC`: saliency detector ILSVRC.
* `ssd_mobilenet_v1_SOS`: saliency detector SOS.


## Android application
### Requirements
Currently, the app settings in `build.gradle` file are:
* compileSdkVersion 29
* minSdkVersion 19
* targetSdkVersion 29

The minimum SDK 19 indicates that smartphones with an Android version higher (or equal) than Android 4.4 (KitKat) can run the application. 

### Description
This demo enables to run the algorithm described in the **paper** on Android smartphone.\
User can select an image from the asset or from storage.\
The app can run inference using only the polarity classifier or using the saliency detector in combination with polarity classifier.\
For each performed inference phase the app displays:
* *Inference time*: time spent to perform forward phase.
* *Inference result*: eventual polarity output, "Positive" or "Negative".
* *Inference confidence*: confidence associated with the prediction. A number in [0, 1] range if the final polarity class is established by the entire image classifier, "---" otherwise. The rationale behind this is that the algorithm's operations can change the range and the meaning of the confidence value. For more details, please refer to the **paper**.

Each model of the **pythonProject** was converted to *TFLite* exploiting post-training quantization process for *FP32*, *FP16* and *INT8* formats. Models are available in the asset folder.\
The most important directories are:
* `assets`: collection of test image, saliency and polarity models, labels file.
* `utils`: holds java classes of algorithm, file and general utilities.
* `activity`: holds MainActivity, which is the only activity of the application.
* `classes`: holds java classes concerning the prediction, the polarity classifier and saliency detector.

## Reference
If you find the code or pre-trained models useful, please cite the following paper:

**Design and Deployment of an Image Polarity Detector with Visual Attention.** E. Ragusa, T. Apicella, C. Gianoglio, R. Zunino and P. Gastaldo. Cognitive Computation, 2021. [DOI](https://doi.org/10.1007/s12559-021-09829-6)

    @article{ragusa2021design,
      title={Design and Deployment of an Image Polarity Detector with Visual Attention},
      author={Ragusa, Edoardo and Apicella, Tommaso and Gianoglio, Christian and Zunino, Rodolfo and Gastaldo, Paolo},
      journal={Cognitive Computation},
      pages={1--13},
      year={2021},
      publisher={Springer}
    }
