
## Introduction
**[Towards Privacy-Supporting Fall Detection via Deep Unsupervised RGB2Depth Adaptation](https://arxiv.org/abs/2308.12049)**

Fall detection is a vital task in health monitoring, as
it allows the system to trigger an alert and therefore enabling
faster interventions when a person experiences a fall. Although
most previous approaches rely on standard RGB video data, such
detailed appearance-aware monitoring poses significant privacy
concerns. Depth sensors, on the other hand, are better at preserving
privacy as they merely capture the distance of objects from the
sensor or camera, omitting color and texture information.
In this paper, we introduce a privacy-supporting solution that
makes the RGB-trained model applicable in depth domain and
utilizes depth data at test time for fall detection. To achieve crossmodal
fall detection, we present an unsupervised RGB to Depth
(RGB2Depth) cross-modal domain adaptation approach that leverages
labelled RGB data and unlabelled depth data during training.
Our proposed pipeline incorporates an intermediate domain
module for feature bridging, modality adversarial loss for modality
discrimination, classification loss for pseudo-labeled depth data
and labeled source data, triplet loss that considers both source
and target domains, and a novel adaptive loss weight adjustment
method for improved coordination among various losses.
Our approach achieves state-of-the-art results in the unsupervised
RGB2Depth domain adaptation task for fall detection.

<div align="center">
  <img src="https://raw.githubusercontent.com/1015206533/privacy_supporting_fall_detection/master/resources/introduction.png" width="800px"/><br>
    <p style="font-size:1.5vw;">Unsupervised Modality Adaptation for Fall Detection (UMA-FD)</p>
</div>

## Installation 
The project is developed based on [MMAction2](https://github.com/open-mmlab/mmaction2). 
First, we need to install the environment required for mmaction to run.
Then, we need to install the packages required by this project, such as tensorboard.
The version of the installation package is detailed in the requirements_new.txt

```shell
conda create -n open-mmlab python=3.8 pytorch=1.10 cudatoolkit=11.3 torchvision -c pytorch -y
conda activate open-mmlab
pip3 install openmim
mim install mmcv-full
mim install mmdet  # optional
mim install mmpose  # optional
git clone https://github.com/open-mmlab/mmaction2.git
cd mmaction2
pip3 install -e .
pip3 install tensorboard
……
```

## Dataset

In the experiments of the paper, we use part data of of [NTU RGB+D](https://rose1.ntu.edu.sg/dataset/actionRecognition/) and [kinetics-700](https://github.com/cvdfoundation/kinetics-dataset). The data label used is under the data folder. According to the data label, to reproduce the results of the article, we need to download the data from the corresponding link. 
For the  kinetics-700 dataset, our generated depth data of rgb video has been uploaded to [Baidu Netdisk](https://pan.baidu.com/s/1nWfesEPo0X3qIiEY7-eAOg), and the extraction code is 2ogm.
Take the file [idm_train_depth_unsim.txt](https://github.com/1015206533/privacy_supporting_fall_detection/blob/master/data/kinetics_700/annotation/idm_train_depth_unsim.txt) as an example to illustrate the organizational form of the data file.
Below is a line from the data file, and each row has four columns, which respectively represent the data file path, frame number, action label(0 means non-fall and 1 means fall), and modal label(0 means depth date and 1 means rgb data). 
The last level of the data file directory is the name of the video, and the penultimate level is the category to which the video belongs.
The images divided into single frame are saved in the corresponding data file directory, and the naming format adopts "img_{:04}.png".format(frame_number), such as the figure below.

```shell
/home/hejun/dataset/uda/kinetics_700/depth_sample/depth_falling_off_chair/0222_HvWtr0qrUao 28 1 0

link：https://pan.baidu.com/s/1nWfesEPo0X3qIiEY7-eAOg 
code：2ogm 
```

![date format](https://raw.githubusercontent.com/1015206533/privacy_supporting_fall_detection/master/resources/data_format.png)

## Get Started

We can use the following commands for model training and inference. The file idm_uda.py is the configuration file for the X3D backbone. We can also use the other backbones, such as I3D and C3D, and the corresponding configuration file are i3d_idm_uda.py and c3d_idm_uda.py. The parameters and training data required for model training can be adjusted in the configuration file. The file test_label.txt is the true label file of the test data, and each line contains the label of a sample.

```shell
./tools/dist_train.sh x3d_uda/configs/idm_uda.py 2 --seed 1024 --validate --test-last --test-best --deterministic

./tools/dist_test.sh ./x3d_uda/configs/idm_uda.py ./checkpoints/best_top1_acc_epoch_239.pth 2 --out results.pkl

python ./own_scipt/evaluation/evaluation.py ./results.pkl ./test_label.txt
```

## FAQ

Please send emails to ask questions.
