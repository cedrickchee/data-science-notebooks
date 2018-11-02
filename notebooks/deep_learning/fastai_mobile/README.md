# Fast.ai Mobile Camera

## Background

This work is part of our project while studying fast.ai's Deep Learning for Coders, Part 1 version 3 course.

The idea for this project originated from Sanyam (init27)'s Not Hotdog mobile app idea while we were looking for possible projects which we can take on together or even individually.

### The Plan

Work on this project together and publish a blog post series.

## The Project

Port SqueezeNet, a small CNN into mobile phone and build a simple Android and iOS app.

- Tasks:
  - Research and design an efficient Convolutional Neural Networks for on-device/mobile vision.
    - Design a mobile-first computer vision models for PyTorch, designed to effectively maximize accuracy while being mindful of the restricted resources for an on-device application—better than SqueezeNet!
    - Benchmark: SqueezeNet v1.1 vs. MobileNetV2 vs. Shufflenet
  - Build a Not Hotdog classifier using the new mobile-first CV model in PyTorch
    - Train, deploy and run on iOS and Android devices.

## Repo

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
