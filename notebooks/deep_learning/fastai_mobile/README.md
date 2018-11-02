# Fast.ai Mobile Camera

:tada: Check out a working PyTorch-Caffe2 implementation on mobile: :tada:

- [Android Camera app demo (video)](https://youtu.be/TYkoaVNCMos)

**Guide - How I Shipped a Neural Network on Android/iOS Phones with PyTorch and Android Studio/Xcode**

I'll walk you through every step, from problem all the way to building and deploying the Android/iOS app to mobile phones.

- Learn how to ship SqueezeNet from PyTorch to Caffe2 + Android/iOS app. Please take a look at this [notebook](https://nbviewer.jupyter.org/github/cedrickchee/data-science-notebooks/blob/master/notebooks/deep_learning/fastai_mobile/shipping_squeezenet_from_pytorch_to_android.ipynb).
  - Android project for AI Camera app tutorial in this [notebook](https://nbviewer.jupyter.org/github/cedrickchee/data-science-notebooks/blob/master/notebooks/deep_learning/fastai_mobile/shipping_squeezenet_from_pytorch_to_android.ipynb#Fast.ai-Mobile-Camera-Project).
  - iOS project (TBD)
- Get started with an introduction to [Open Neural Network Exchange formst (ONNX)](https://onnx.ai/) in this Jupyter [notebook](https://nbviewer.jupyter.org/github/cedrickchee/data-science-notebooks/blob/master/notebooks/deep_learning/fastai_mobile/onnx_from_pytorch_to_caffe2.ipynb).

---

## Background

This work is part of our project while studying [fast.ai's Deep Learning for Coders, Part 1 version 3 course](https://forums.fast.ai/c/part1-v3). I joined fast.ai Live—the new version of International Fellowships this year, thanks to Jeremy and Rachel.

The idea for this project originated from [Sanyam (init27)](https://forums.fast.ai/u/init_27)'s Not Hotdog mobile app idea while we were looking for possible projects which we can take on together or even individually. At the point during our discussions, we realized that shipping a neural network on iOS with CoreML with PyTorch is really a painful experience. Just take a look at this blog post, ["How I Shipped a Neural Network on iOS with CoreML, PyTorch, and React Native"](https://attardi.org/pytorch-and-coreml) written by Stefano J. Attardi to understand the state and scale of this problem. We are motivated by the size of the problem and the potentials, for example, [developers are offering consulting services to convert neural networks to run on mobile devices](http://machinethink.net/faster-neural-networks/).

The details are available in our [**fast.ai Asia study group** pre-class meeting notes](https://hackmd.io/s/Sk5tydOjQ) on 2018-10-20 19:30 GMT+8.

## The Project

We will port SqueezeNet or a small ConvNet in PyTorch into mobile phone and build a simple Android and iOS app.

- Project Goal
  - **Work on this project together** in our virtual Asia study group and **publish a blog post series**.
- High-level activities (plan):
  - Research and design an efficient Convolutional Neural Networks for on-device/mobile vision.
    - Design a mobile-first computer vision models for PyTorch, designed to effectively maximize accuracy while being mindful of the restricted resources for an on-device application—better than SqueezeNet!
    - Benchmark: SqueezeNet v1.1 vs. MobileNetV2 vs. Shufflenet
  - Build a Not Hotdog classifier using the new mobile-first CV model in PyTorch
    - Train, deploy and run on iOS and Android devices.

## About this Repo

This directory contains relevant files for implementating:
- SqueezeNet v1.1 model
- Jupyter notebooks for replicating the work done:
  - Introduction to ONNX
  - Shipping SqueezeNet from PyTorch to ONNX to Android app
    - What is SqueezeNet
    - Export the PyTorch model as ONNX model
    - ONNX with Caffe2 backend
    - Export the model to run on mobile devices
    - Integrating Caffe2 on mobile
    - Shipping the models into the Android app
    - Android app development using Android Studio
  - Demo

## Mobile-First Computer Vision

### Models and Network Architectures

Quick literature review:

- MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications (submitted on 17 Apr 2017)
  - Separable convolution
  - The important point here is that the bottleneck of the computational cost is now conv1x1!
- ShuffleNet: an extremely efficient convolutional neural network for mobile devices (submitted on 4 Jul 2017 (v1), last revised 7 Dec 2017 (this version, v2))
  - While conv1x1 is already efficient and there seems to be no room for improvement, grouped conv1x1 can be used for this purpose!
- MobileNetV2: Inverted Residuals and Linear Bottlenecks (Submitted on 13 Jan 2018 (v1), last revised 2 Apr 2018 (this version, v3))
  - [MobileNet version 2](http://machinethink.net/blog/mobilenet-v2/)
    - It's faster, uses less memory, and is better at conserving battery power.
    - Another option is SqueezeNet, which uses even fewer parameters than MobileNet, but it’s optimized mostly for low memory situations, not so much for speed. It also has lower accuracy. Recently a new version was announced, SqueezeNext, and I’m interested in comparing this to MobileNet V2, so I might write a future blog post about this.
- [Squeezenet](https://github.com/DeepScale/SqueezeNet)
- Enet
- BinaryConnect: Training Deep Neural Networks with binary weights during propagations
  - You can even squeeze a network down to 1-bit.

## Fast.ai

- Some example of discussion forum threads:
  - [Getting neural networks to work on phone](https://forums.fast.ai/t/getting-neural-networks-to-work-on-phone/2603)

## Version History

### v0.1

- _TBD_

## TODO

- [x] Install ONNX
- [x] Install Caffe2
- [ ] Transfer learning SqueezeNet with new datasets (i.e. hotdog, not hotdog)
- [x] Upgrade Android Studio
- [x] Update Android SDK
- [x] Install and setup Android NDK
- [x] Resolve all issues related to Android Studio
- [x] Android app demo
- [ ] Research on newer mobile-first computer vision models (objective: inference speed above 30 fps)
- [ ] iOS app
- [ ] Research on improving model accuracy

## License

MIT. Copyright 2018 Cedric Chee and contributors.
