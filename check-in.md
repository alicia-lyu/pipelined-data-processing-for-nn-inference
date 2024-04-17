# Project Check In: Asynchronous End-to-End Audio and Image Processing for Faster Neural Network Inference

Group ZARN: Alicia Lyu, Ruijia Chen, Zach Potter, Nithin Weerasinghe

## What we have done so far

- Familiarize ourselves with the background of the project
- Follow hands-on tutorial to learn about NVIDIA Triton Inference Server
- Setup environment
- Image team: 
  - Set up models (OpenCV and ResNet)
  - Write the initial client program of naive sequential data processing
  - Run experiments on uniformly arriving image data (ongoing)
- Audio team:
  - Set up the model (wav2vec)
  - Write the initial client program of naive sequential data processing (ongoing)
- Write code for asynchronous data processing (can be shared by both workloads)

## Challenges we faced so far

- Resource reservation: As our project needs a GPU, resource reservation has been a big problem. We initially decided to use CloudLab as our primary resource provider. However, multiple experiments fail at startup, despite reserving in advance. We spent significant amount of time reserving, waiting, and then finding out it fails. Eventually we switched to GCP. We scanned almost every zone in the US to find a machine with GPU. Now that we have one, we're never letting it go, despite high cost of keeping it running.
- Lack of toolset for team coordination: This is the same challenge we faced with the assignments. We feel that, despite a few years of Computer Science education, we are never systematically introduced into Linux, bash, and git. We are expected to learn them ourselves. Although we are able to do that to some extent, we have to go through many times of trial and error, and end up with a solution probably not optimal.

## Our updated timeline

- Apr 21: Wrap up our current tasks
- Apr 28: 
  - One person: Run asynchronous experiments for both image and audio
  - One person: Experiment different data arrival patterns (e.g. Poisson distribution)
  - One person: Utilize multiple CPU threads to process data and start inference jobs
- May 2: Wrap up our work and prepare for poster presentation
- May 7: Write the report

We researched into the video model provided by our contact person Minghao. Nexus is a full-edge system for video inferencing, not a model to be deployed on the NVIDIA Triton server. He also mentioned that Video inferencing is a similar workload to image data, because it is just images arriving in differental distributions. We decide not to have a designated section for video data, but simply experiment with different data arrival patterns.