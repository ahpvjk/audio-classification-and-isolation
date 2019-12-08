[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/ahpvjk/audio-classification-and-isolation/blob/master/LICENSE)

# Audio Classification and Isolation: A Deep NeuralNetwork Approach

>__This work is for the [CS230 Fall 2019 Project](https://cs230.stanford.edu/project/)__
>* Co-authors: Pratap Vedula and Muni Venkata Jasantha Konduru

## Introduction
A particular sound of interest is almost always overlapping with other multiple waves from different sources, and if those are at a comparable if not higher amplitude, will lower the ability to interpret effectively when heard. Audio classification and isolation will allow us to focus on specific sounds of interest. Few example scenarios are self-driving cars identifying a police siren, isolating a broadcaster's voice from others in a loud crowd, or recognizing an infant crying in a noisy environment.

## Installation
#### System Packages
```bash
sudo apt-get install python3
sudo apt-get install python3-pip
sudo apt-get install ffmpeg
```

#### Python Packages
```bash
pip3 install -r requirements.txt
```

### Usage
#### Training
```bash
./__init__.py --train 
```

#### Evaluation
```bash
./__init__.py --eval [MODEL_STATE_PT_FILE]
```

#### Prediction
```bash
./__init__.py --predict [MODEL_STATE_PT_FILE] [INPUT_FILE]
```
