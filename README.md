# Data Science Notebooks

Data science Python notebooks—a collection of Jupyter notebooks on machine learning, deep learning, statistical inference, data analysis and visualization.

This repo contains various Python Jupyter notebooks I have created to experiment and learn with the core libraries essential for working with data in Python and work through exercises, assignments, course works, and explore subjects that I find interesting such as machine learning and deep learning. Familiarity with Python as a language is assumed.

The essential core libraries that I will be focusing on for working with data are NumPy, Pandas, Matplotlib, PyTorch, TensorFlow, Keras, Caffe, scikit-learn, spaCy, NLTK, Gensim, and related packages.

## Table of Contents

- [Data Science Notebooks](#data-science-notebooks)
  - [Table of Contents](#table-of-contents)
  - [How to Use this Repo](#how-to-use-this-repo)
  - [About](#about)
  - [Software](#software)
  - [Deep Learning](#deep-learning)
    - [Projects](#projects)
    - [DL Assignments, Exercises or Course Works](#dl-assignments-exercises-or-course-works)
      - [fast.ai's Deep Learning Part 1: Practical Deep Learning for Coders 2018 (v2): Oct - Dec 2017](#fastais-deep-learning-part-1-practical-deep-learning-for-coders-2018-v2-oct---dec-2017)
      - [fast.ai's Deep Learning Part 1: Practical Deep Learning for Coders 2019 (v3): Oct - Dec 2018](#fastais-deep-learning-part-1-practical-deep-learning-for-coders-2019-v3-oct---dec-2018)
      - [fast.ai's Deep Learning Part 2: Cutting Edge Deep Learning for Coders 2017 (v1): Feb - Apr 2017](#fastais-deep-learning-part-2-cutting-edge-deep-learning-for-coders-2017-v1-feb---apr-2017)
      - [fast.ai's Deep Learning Part 2: Cutting Edge Deep Learning for Coders 2018 (v2): Mar - May 2018](#fastais-deep-learning-part-2-cutting-edge-deep-learning-for-coders-2018-v2-mar---may-2018)
  - [Machine Learning](#machine-learning)
    - [ML Assignments, Exercises or Course Works](#ml-assignments-exercises-or-course-works)
      - [Andrew Ng's "Machine Learning" class on Coursera](#andrew-ngs-%22machine-learning%22-class-on-coursera)
      - [fast.ai's machine learning course](#fastais-machine-learning-course)
  - [Libraries or Frameworks](#libraries-or-frameworks)
    - [NumPy](#numpy)
    - [PyTorch](#pytorch)
    - [TensorFlow](#tensorflow)
    - [Keras](#keras)
    - [Pandas](#pandas)
    - [Matplotlib](#matplotlib)
  - [Kaggle Competitions](#kaggle-competitions)
  - [License](#license)
    - [Code](#code)
    - [Text](#text)

## How to Use this Repo

- Run the code using the Jupyter notebooks available in this repository's [notebooks](/notebooks) directory.
- Launch a live notebook server with these notebooks using [binder](https://beta.mybinder.org/): [![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/cedrickchee/data-science-notebooks/blob/master/notebooks/index.ipynb/master)

## About

The notebooks were written and tested with Python 3.6, though other Python versions (including Python 3.x) should work in nearly all cases.

See [index.ipynb](/notebooks/index.ipynb) for an index of the notebooks available.

## Software

The code in the notebook was tested with Python 3.6, though most (but not all) will also work correctly with Python 3.x.

The packages I used to run the code in the notebook are listed in [requirements.txt](requirements.txt) (Note that some of these exact version numbers may not be available on your platform: you may have to tweak them for your own use). To install the requirements using conda, run the following at the command-line:

```bash
$ conda install --file requirements.txt
```

To create a stand-alone environment named DSN with Python 3.6 and all the required package versions, run the following:

```bash
$ conda create -n DSN python=3.5 --file requirements.txt
```

You can read more about using conda environments in the [Managing Environments](http://conda.pydata.org/docs/using/envs.html) section of the conda documentation.

## Deep Learning

### Projects

|Notebook|Description|
| --- | --- |
| [Deep Painterly Harmonization](/notebooks/deep_learning/deep_painterly_harmonization/harmonization.ipynb) | Implement [Deep Painterly Harmonization paper](https://arxiv.org/abs/1804.03189) in PyTorch |
| [Language modelling in Malay language for downstream NLP tasks](/notebooks/deep_learning/ULMFiT/malay_language_model.ipynb) | Implement [Universal Language Model Fine-tuning for Text Classification (ULMFiT)](http://nlp.fast.ai/classification/2018/05/15/introducting-ulmfit.html) in PyTorch |
| [Not Hotdog AI Camera mobile app](notebooks/deep_learning/fastai_mobile/README.md) | Asia virtual study group project for fast.ai deep learning part 1, v3 course. Ship a convolutional neural network on Android/iOS with PyTorch and Android Studio/Xcode |

### Language Models

Notebooks for trying out transformer and large language models.

|Notebook|Description|
| --- | --- |
| [Flan-UL2 20B](/notebooks/deep_learning/deep_painterly_harmonization/harmonization.ipynb) | Flan 20B with UL2 code walkthrough. This shows how you can get it running on 1x A100 40GB GPU with the HuggingFace library and using 8-bit inference. Using CoT, zeroshot (logical reasoning, story writing, common sense reasoning, speech writing). Testing large (2048) token input. |

### DL Assignments, Exercises or Course Works

#### fast.ai's Deep Learning Part 1: Practical Deep Learning for Coders 2018 (v2): Oct - Dec 2017

| Notebook | Description |
| --- | --- |
| [lesson1](https://nbviewer.jupyter.org/github/cedrickchee/fastai/blob/master/courses/dl1/lesson1.ipynb), <br /> [lesson1-vgg](https://nbviewer.jupyter.org/github/cedrickchee/fastai/blob/master/courses/dl1/lesson1-vgg.ipynb), <br /> [lesson1-rxt50](https://nbviewer.jupyter.org/github/cedrickchee/fastai/blob/master/courses/dl1/lesson1-rxt50.ipynb), <br /> [keras_lesson1](https://nbviewer.jupyter.org/github/cedrickchee/fastai/blob/master/courses/dl1/keras_lesson1.ipynb) | Lesson 1 - Recognizing Cats and Dogs |
| [lesson2-image_models](https://nbviewer.jupyter.org/github/cedrickchee/fastai/blob/master/courses/dl1/lesson2-image_models.ipynb) | Lesson 2 - Improving Your Image Classifier |
| [lesson3-rossman](https://nbviewer.jupyter.org/github/cedrickchee/fastai/blob/master/courses/dl1/lesson3-rossman.ipynb) | Lesson 3 - Understanding Convolutions |
| [lesson4-imdb](https://nbviewer.jupyter.org/github/cedrickchee/fastai/blob/master/courses/dl1/lesson4-imdb.ipynb) | Lesson 4 - Structured Time Series and Language Models |
| [lesson5-movielens](https://nbviewer.jupyter.org/github/cedrickchee/fastai/blob/master/courses/dl1/lesson5-movielens.ipynb) | Lesson 5 - Collaborative Filtering; Inside the Training Loop |
| [lesson6-rnn](https://nbviewer.jupyter.org/github/cedrickchee/fastai/blob/master/courses/dl1/lesson6-rnn.ipynb), <br /> [lesson6-sgd](https://nbviewer.jupyter.org/github/cedrickchee/fastai/blob/master/courses/dl1/lesson6-sgd.ipynb) | Lesson 6 - Interpreting Embeddings; RNNs from Scratch |
| [lesson7-cifar10](https://nbviewer.jupyter.org/github/cedrickchee/fastai/blob/master/courses/dl1/lesson7-cifar10.ipynb), <br /> [lesson7-CAM](https://nbviewer.jupyter.org/github/cedrickchee/fastai/blob/master/courses/dl1/lesson7-CAM.ipynb) | Lesson 7 - ResNets from Scratch |

#### fast.ai's Deep Learning Part 1: Practical Deep Learning for Coders 2019 (v3): Oct - Dec 2018

[Deep Learning Part 1: 2019 Edition](https://github.com/cedrickchee/knowledge/blob/master/courses/fast.ai/deep-learning-part-1/2019-edition/README.md)

| Notebook | Description |
| --- | --- |
| [00_notebook_tutorial.ipynb](https://nbviewer.jupyter.org/github/cedrickchee/fastai-course-v3/blob/master/nbs/dl1/00_notebook_tutorial.ipynb), <br /> [lesson1-pets.ipynb](https://nbviewer.jupyter.org/github/cedrickchee/fastai-course-v3/blob/master/nbs/dl1/lesson1-pets_20181023.ipynb) | [Lesson 1 - Image Recognition](https://github.com/cedrickchee/knowledge/blob/master/courses/fast.ai/deep-learning-part-1/2019-edition/lesson-1-image-recognition.md) |
| [lesson2-download.ipynb](https://nbviewer.jupyter.org/github/cedrickchee/fastai-course-v3/blob/master/nbs/dl1/lesson2-download_20181205.ipynb) <br /> [lesson2-sgd.ipynb](https://nbviewer.jupyter.org/github/cedrickchee/fastai-course-v3/blob/master/nbs/dl1/lesson2-sgd_20181207.ipynb) | [Lesson 2 - Computer Vision: Deeper Applications](https://github.com/cedrickchee/knowledge/blob/master/courses/fast.ai/deep-learning-part-1/2019-edition/lesson-2-deeper-dive-into-cv.md) |
| [lesson3-planet.ipynb](https://nbviewer.jupyter.org/github/cedrickchee/fastai-course-v3/blob/master/nbs/dl1/lesson3-planet_20181210.ipynb) <br /> [lesson3-camvid.ipynb](https://nbviewer.jupyter.org/github/cedrickchee/fastai-course-v3/blob/master/nbs/dl1/lesson3-camvid_20181212.ipynb) <br /> [lesson3-head-pose.ipynb](https://nbviewer.jupyter.org/github/cedrickchee/fastai-course-v3/blob/master/nbs/dl1/lesson3-head-pose-20181226.ipynb) <br /> [lesson3-imdb.ipynb](https://nbviewer.jupyter.org/github/cedrickchee/fastai-course-v3/blob/master/nbs/dl1/lesson3-imdb_20181226.ipynb) | [Lesson 3 - Multi-label, Segmentation, Image Regression, and More](https://github.com/cedrickchee/knowledge/blob/master/courses/fast.ai/deep-learning-part-1/2019-edition/lesson-3-multilabel-segmentation.md) |
| [lesson4-tabular.ipynb](https://nbviewer.jupyter.org/github/cedrickchee/fastai-course-v3/blob/master/nbs/dl1/lesson4-tabular_20181226.ipynb) <br /> [lesson4-collab.ipynb](https://nbviewer.jupyter.org/github/cedrickchee/fastai-course-v3/blob/master/nbs/dl1/lesson4-collab_20181227.ipynb) | [Lesson 4 - NLP, Tabular, and Collaborative Filtering](https://github.com/cedrickchee/knowledge/blob/master/courses/fast.ai/deep-learning-part-1/2019-edition/lesson-4-nlp-tabular-collab.md) |
| [lesson5-sgd-mnist.ipynb](https://nbviewer.jupyter.org/github/cedrickchee/fastai-course-v3/blob/master/nbs/dl1/lesson5-sgd-mnist-20190104.ipynb) | [Lesson 5 - Foundations of Neural Networks](https://github.com/cedrickchee/knowledge/blob/master/courses/fast.ai/deep-learning-part-1/2019-edition/lesson-5-foundations-neural-nets.md) |
| [lesson6-rossmann.ipynb](https://nbviewer.jupyter.org/github/cedrickchee/fastai-course-v3/blob/master/nbs/dl1/lesson6-rossmann_20190107.ipynb) <br /> [rossman_data_clean.ipynb](https://nbviewer.jupyter.org/github/cedrickchee/fastai-course-v3/blob/master/nbs/dl1/rossman_data_clean_20190107.ipynb) <br /> [lesson6-pets-more.ipynb](https://nbviewer.jupyter.org/github/cedrickchee/fastai-course-v3/blob/master/nbs/dl1/lesson6-pets-more_20190105.ipynb) | [Lesson 6 - Foundations of Convolutional Neural Networks](https://github.com/cedrickchee/knowledge/blob/master/courses/fast.ai/deep-learning-part-1/2019-edition/lesson-6-foundations-convolutional-neural-nets.md) |
| [lesson7-resnet-mnist.ipynb](https://nbviewer.jupyter.org/github/cedrickchee/fastai-course-v3/blob/master/nbs/dl1/lesson7-resnet-mnist_20190108.ipynb) <br /> [lesson7-superres-gan.ipynb](https://nbviewer.jupyter.org/github/cedrickchee/fastai-course-v3/blob/master/nbs/dl1/lesson7-superres-gan_20190110.ipynb) <br /> [lesson7-superres-imagenet.ipynb](https://nbviewer.jupyter.org/github/cedrickchee/fastai-course-v3/blob/master/nbs/dl1/lesson7-superres-imagenet.ipynb) <br /> [lesson7-superres.ipynb](https://nbviewer.jupyter.org/github/cedrickchee/fastai-course-v3/blob/master/nbs/dl1/lesson7-superres_20190110.ipynb) <br /> [lesson7-wgan.ipynb](https://nbviewer.jupyter.org/github/cedrickchee/fastai-course-v3/blob/master/nbs/dl1/lesson7-wgan_20190110.ipynb) <br /> [lesson7-human-numbers.ipynb](https://nbviewer.jupyter.org/github/cedrickchee/fastai-course-v3/blob/master/nbs/dl1/lesson7-human-numbers_20190111.ipynb) | [Lesson 7 - ResNets, U-Nets, GANs and RNNs](https://github.com/cedrickchee/knowledge/blob/master/courses/fast.ai/deep-learning-part-1/2019-edition/lesson-7-resnet-unet-gan-rnn.md) |

#### fast.ai's Deep Learning Part 2: Cutting Edge Deep Learning for Coders 2017 (v1): Feb - Apr 2017

[Deep Learning Part 2: 2017 Edition](https://github.com/cedrickchee/knowledge/blob/master/courses/fast.ai/deep-learning-part-2/2017-edition/README.md)

| Notebook | Description |
| --- | --- |
| [neural-style](https://github.com/fastai/courses/blob/master/deeplearning2/neural-style.ipynb) | [Lesson 8 - Artistic Style](https://github.com/cedrickchee/knowledge/blob/master/courses/fast.ai/deep-learning-part-2/2017-edition/lesson-8-artistic-style.md) |
| [imagenet-processing](https://github.com/fastai/courses/blob/master/deeplearning2/imagenet_process.ipynb) | [Lesson 9 - Generative Models](https://github.com/cedrickchee/knowledge/blob/master/courses/fast.ai/deep-learning-part-2/2017-edition/lesson-9-generative-models.md) |
| [neural-sr](https://github.com/fastai/courses/blob/master/deeplearning2/neural-sr.ipynb), <br /> [keras-dcgan](https://github.com/fastai/courses/blob/master/deeplearning2/DCGAN.ipynb), <br /> [pytorch-tutorial](https://github.com/fastai/courses/blob/master/deeplearning2/pytorch-tut.ipynb), <br /> [wgan-pytorch](https://github.com/fastai/courses/blob/master/deeplearning2/wgan-pytorch.ipynb) | [Lesson 10 - Multi-modal & GANs](https://github.com/cedrickchee/knowledge/blob/master/courses/fast.ai/deep-learning-part-2/2017-edition/lesson-10-multi-modal-and-gans.md) |
| [kmeans-clustering](https://github.com/cedrickchee/fastai-dl2-2017/blob/master/kmeans_test.ipynb), <br /> [babi-memory-neural-net](https://github.com/fastai/courses/blob/master/deeplearning2/babi-memnn.ipynb) | [Lesson 11 - Memory Networks](https://github.com/cedrickchee/knowledge/blob/master/courses/fast.ai/deep-learning-part-2/2017-edition/lesson-11-memory-networks.md) |
| [spelling_bee_RNN](https://github.com/cedrickchee/fastai-dl2-2017/blob/master/spelling_bee_RNN.ipynb) | [Lesson 12 - Attentional Models](https://github.com/cedrickchee/knowledge/blob/master/courses/fast.ai/deep-learning-part-2/2017-edition/lesson-12-attentional-models.md) |
| [translate-pytorch](https://github.com/fastai/courses/blob/master/deeplearning2/translate-pytorch.ipynb), <br /> [densenet-keras](https://github.com/fastai/courses/blob/master/deeplearning2/densenet-keras.ipynb) | [Lesson 13 - Neural Translation](https://github.com/cedrickchee/knowledge/blob/master/courses/fast.ai/deep-learning-part-2/2017-edition/lesson-13-neural-translation.md) |
| [rossmann](https://github.com/fastai/courses/blob/master/deeplearning2/rossman.ipynb), <br /> [tiramisu-keras](https://github.com/fastai/courses/blob/master/deeplearning2/tiramisu-keras.ipynb) | [Lesson 14 - Time Series & Segmentation](https://github.com/cedrickchee/knowledge/blob/master/courses/fast.ai/deep-learning-part-2/2017-edition/lesson-14-time-series-and-segmentation.md) |

#### fast.ai's Deep Learning Part 2: Cutting Edge Deep Learning for Coders 2018 (v2): Mar - May 2018

[Deep Learning Part 2: 2018 Edition](https://github.com/cedrickchee/knowledge/blob/master/courses/fast.ai/deep-learning-part-2/2018-edition/README.md)

| Notebook | Description |
| --- | --- |
| [Pascal VOC—Object Detection](https://nbviewer.jupyter.org/github/cedrickchee/fastai/blob/master/courses/dl2/pascal.ipynb) | [Lesson 8 - Object Detection](https://github.com/cedrickchee/knowledge/blob/master/courses/fast.ai/deep-learning-part-2/2018-edition/lesson-8-object-detection.md) |
| [Pascal VOC—Multi Object Detection](https://nbviewer.jupyter.org/github/cedrickchee/fastai/blob/master/courses/dl2/pascal-multi.ipynb) | [Lesson 9 - Single Shot Multibox Detector (SSD)](https://github.com/cedrickchee/knowledge/blob/master/courses/fast.ai/deep-learning-part-2/2018-edition/lesson-9-multi-object-detection.md) |
| [IMDB—Language Model](https://nbviewer.jupyter.org/github/cedrickchee/fastai/blob/master/courses/dl2/imdb.ipynb) | [Lesson 10 - Transfer Learning for NLP and NLP Classification](https://github.com/cedrickchee/knowledge/blob/master/courses/fast.ai/deep-learning-part-2/2018-edition/lesson-10-transfer-learning-nlp.md) |
| [WMT15 Giga French-English—Neural Machine Translation](https://nbviewer.jupyter.org/github/cedrickchee/fastai/blob/master/courses/dl2/translate.ipynb), <br /> [DeViSE (Deep Visual-Semantic Embedding Model)](https://nbviewer.jupyter.org/github/cedrickchee/fastai/blob/master/courses/dl2/devise.ipynb) | [Lesson 11 - Neural Translation; Multi-modal Learning](https://github.com/cedrickchee/knowledge/blob/master/courses/fast.ai/deep-learning-part-2/2018-edition/lesson-11-neural-translation.md) |
| [CIFAR-10 DarkNet](https://nbviewer.jupyter.org/github/cedrickchee/fastai/blob/master/courses/dl2/cifar10-darknet.ipynb), <br /> [CIFAR-10 DAWNBench](https://nbviewer.jupyter.org/github/cedrickchee/fastai/blob/master/courses/dl2/cifar10-dawn.ipynb), <br /> [Wasserstein GAN](https://nbviewer.jupyter.org/github/cedrickchee/fastai/blob/master/courses/dl2/wgan.ipynb), <br /> [CycleGAN](https://nbviewer.jupyter.org/github/cedrickchee/fastai/blob/master/courses/dl2/cyclegan.ipynb) | [Lesson 12 - DarkNet; Generative Adversarial Networks \(GANs\)](https://github.com/cedrickchee/knowledge/blob/master/courses/fast.ai/deep-learning-part-2/2018-edition/lesson-12-gan.md) |
| [TrainingPhase API](https://nbviewer.jupyter.org/github/cedrickchee/fastai/blob/master/courses/dl2/training_phase.ipynb), <br /> [Neural Algorithm of Artistic Style Transfer](https://nbviewer.jupyter.org/github/cedrickchee/fastai/blob/master/courses/dl2/style-transfer.ipynb) | [Lesson 13 - Image Enhancement; Style Transfer; Data Ethics](https://github.com/cedrickchee/knowledge/blob/master/courses/fast.ai/deep-learning-part-2/2018-edition/lesson-13-image-enhancement.md) |
| [Super Resolution](https://nbviewer.jupyter.org/github/cedrickchee/fastai/blob/master/courses/dl2/enhance.ipynb), <br /> [Real-time Style Transfer Neural Net](https://nbviewer.jupyter.org/github/cedrickchee/fastai/blob/master/courses/dl2/style-transfer-net.ipynb), <br /> [Kaggle Carvana Image Masking](https://nbviewer.jupyter.org/github/cedrickchee/fastai/blob/master/courses/dl2/carvana.ipynb), <br /> [Kaggle Carvana Image Masking using U-Net](https://nbviewer.jupyter.org/github/cedrickchee/fastai/blob/master/courses/dl2/carvana-unet.ipynb), <br /> [Kaggle Carvana Image Masking using U-Net Large](https://nbviewer.jupyter.org/github/cedrickchee/fastai/blob/master/courses/dl2/carvana-unet-lrg.ipynb) | [Lesson 14 - Super Resolution; Image Segmentation with U-Net](https://github.com/cedrickchee/knowledge/blob/master/courses/fast.ai/deep-learning-part-2/2018-edition/lesson-14-image-segmentation.md) |

## Machine Learning

### ML Assignments, Exercises or Course Works

#### Andrew Ng's "Machine Learning" class on Coursera
  - [Exercise 1 - Linear Regression](/notebooks/machine_learning/coursera/ml_exercise_1.ipynb)
  - [Exercise 2 - Logistic Regression](/notebooks/machine_learning/coursera/ml_exercise_2.ipynb)

#### fast.ai's machine learning course
  - [Lesson 1 - Random Forest](/notebooks/machine_learning/fastai/lesson1-rf.ipynb)
  - [Lesson 2 - Random Forest Interpretation](/notebooks/machine_learning/fastai/lesson2-rf_interpretation.ipynb)
  - [Lesson 3 - Random Forest Foundations](/notebooks/machine_learning/fastai/lesson3-rf_foundations.ipynb)
  - [Lesson 4 - MNIST SGD](/notebooks/machine_learning/fastai/lesson4-mnist_sgd.ipynb)
  - [Lesson 5 - Natural Language Processing (NLP)](/notebooks/machine_learning/fastai/lesson5-nlp.ipynb)

## Libraries or Frameworks

### [NumPy](/notebooks/numpy/index.ipynb)

|Notebook|Description|
| --- | --- |
| [NumPy in 10 minutes](/notebooks/numpy/crash_course.ipynb) | Introduction to NumPy for deep learning in 10 minutes |

### [PyTorch](/notebooks/pytorch/index.ipynb)

_WIP_

### [TensorFlow](/notebooks/tensorflow/index.ipynb)

|Notebook|Description|
| --- | --- |
| [Guide to TensorFlow Keras on TPUs MNIST](/notebooks/tensorflow/google_cloud_tpu/guide_to_tensorflow_keras_on_tpu_mnist.ipynb) | Guide to TensorFlow + Keras on TPU v2 for free on Google Colab |

### [Keras](/notebooks/keras/index.ipynb)

_WIP_

### [Pandas](/notebooks/pandas/index.ipynb)

_WIP_

### [Matplotlib](/notebooks/matplotlib/index.ipynb)

_WIP_

## Kaggle Competitions

| Notebook | Description |
| --- | --- |
| [planet_cv](https://github.com/cedrickchee/fastai/blob/master/courses/dl1/planet_cv.ipynb) | [Planet: Understanding the Amazon from Space](https://www.kaggle.com/c/planet-understanding-the-amazon-from-space)—use satellite data to track the human footprint in the Amazon rainforest |
| [Rossmann](https://github.com/cedrickchee/fastai/blob/master/courses/dl1/lesson3-rossman.ipynb) | [Rossmann Store Sales](https://www.kaggle.com/c/rossmann-store-sales)—forecast sales using store, promotion, and competitor data |
| [fish](https://github.com/cedrickchee/fastai/blob/master/courses/dl1/fish.ipynb) | [The Nature Conservancy Fisheries Monitoring](https://www.kaggle.com/c/the-nature-conservancy-fisheries-monitoring)—Can you detect and classify species of fish? |

## License

This repository contains a variety of content; some developed by Cedric Chee, and some from third-parties. The third-party content is distributed under the license provided by those parties.

*I am providing code and resources in this repository to you under an open source license.  Because this is my personal repository, the license you receive to my code and resources is from me and not my employer.*

The content developed by Cedric Chee is distributed under the following license:

### Code

The code in this repository, including all code samples in the notebooks listed above, is released under the [MIT license](LICENSE). Read more at the [Open Source Initiative](https://opensource.org/licenses/MIT).

### Text

The text content of the book is released under the CC-BY-NC-ND license. Read more at [Creative Commons](https://creativecommons.org/licenses/by-nc-nd/3.0/us/legalcode).