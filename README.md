# Projects for students
* Final update: 2019. 02. 17.
* All right reserved @ ModuLabs 2019


## Getting Started

### Prerequisites
* [`TensorFlow`](https://www.tensorflow.org) 1.12.0
* Python 3.6
* Python libraries:
  * `numpy`, `matplotblib`, `pandas`
  * `PIL`, `imageio` for images
  * `fix_yahoo_finance` for stock market prediction
* Jupyter notebook
* OS X and Linux (Not validated on Windows OS)



## CNN projects

### Image Classification

#### Task
* Overftting을 피하며, accuracy를 높혀 보자
* 밑에 제시된 여러가지 시도를 해보자

#### Dataset
* [Google flower datasets](https://github.com/tensorflow/models/blob/master/research/inception/inception/data/download_and_preprocess_flowers.sh)
* 5개의 클래스(daisy, dandelion, roses, sunflowers, tulips)로 이루어진 꽃 이미지 데이터를 분류

#### Base code
* Dataset: train, validation, test로 split
* Input data shape: (`batch_size`, 150, 150, 3)
* Output data shape: (`batch_size`, `num_classes`=5)
* Architecture: 
  * `Conv2D` (x3) - `Dense` - `Softmax`
  * [`tf.keras.layers`](https://www.tensorflow.org/api_docs/python/tf/keras/layers) 사용
* Training
  * `model.fit_generator` 사용
  * `tf.keras.preprocessing.image.ImageDataGenerator` 사용 for data augmentation
* Evaluation
  * `model.evaluate_generator` 사용 for test dataset

#### Try some techniques
* Change model architectures (Custom model)
  * Or use pretrained models
* Data augmentation
* Various regularization methods


### Image Segmentation

#### Task
* GIANA dataset으로 위내시경 이미지에서 용종을 segmentation 해보자
* 밑에 제시된 여러가지 시도를 해보자
* This code is borrowed from [TensorFlow tutorials/Image Segmentation](https://github.com/tensorflow/models/blob/master/samples/outreach/blogs/segmentation_blogpost/image_segmentation.ipynb) which is made of `tf.keras.layers` and `tf.enable_eager_execution()`.
* You can see the detail description [tutorial link](https://github.com/tensorflow/models/blob/master/samples/outreach/blogs/segmentation_blogpost/image_segmentation.ipynb)  

#### Dataset
* I use below dataset instead of [carvana-image-masking-challenge dataset](https://www.kaggle.com/c/carvana-image-masking-challenge/rules) in TensorFlow Tutorials which is a kaggle competition dataset.
  * carvana-image-masking-challenge dataset: Too large dataset (14GB)
* [Gastrointestinal Image ANAlys Challenges (GIANA)](https://giana.grand-challenge.org) Dataset (345MB)
  * Train data: 300 images with RGB channels (bmp format)
  * Train lables: 300 images with 1 channels (bmp format)
  * Image size: 574 x 500

#### Base code
* Dataset: train, test로 split
* Input data shape: (`batch_size`, 64, 64, 3)
* Output data shape: (`batch_size`, 64, 64, 1)
* Architecture: 
  * 간단한 U-Net 구조
  * [`tf.keras.layers`](https://www.tensorflow.org/api_docs/python/tf/keras/layers) 사용
* Training
  * `tf.data.Dataset` 사용
  * `tf.GradientTape()` 사용 for weight update
* Evaluation
  * MeanIOU: Image Segmentation에서 많이 쓰이는 evaluation measure
  * tf.version 1.12 API: [`tf.metrics.mean_iou`](https://www.tensorflow.org/api_docs/python/tf/metrics/mean_iou)
    * `tf.enable_eager_execution()`이 작동하지 않음
    * 따라서 예전 방식대로 `tf.Session()`을 이용하여 작성하거나 아래와 같이 2.0 version으로 작성하여야 함
  * tf.version 2.0 API: [`tf.keras.metrics.MeanIoU`](https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/keras/metrics/MeanIoU)

#### Try some techniques
* Change model architectures (Custom model)
  * Try another models (DeepLAB, Hourglass, Encoder-Decoder 모델)
* Data augmentation
* Various regularization methods





## RNN projects


### Sentiment Analysis


### Stock Market Prediction

#### Try some techniques
 1. 모델구조 변화
 2. Early stopping 기법
 3. 여러 feature 이용해 보기



## Authors
* [Il Gu Yi](https://github.com/ilguyi)
* Heedong Yoon
