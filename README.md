<p align="center">
  <img src="docs/sys.png" width="900", style="border-radius:10%">
  <h1 align="center">One Map to Find Them All: Real-time Open-Vocabulary Mapping for Zero-shot Multi-Object Navigation</h1>
  <h3 align="center">
    <a href="https://www.kth.se/profile/flbusch?l=en">Finn Lukas Busch</a>,
    <a href="https://www.kth.se/profile/timonh">Timon Homberger</a>,
    <a href="https://www.kth.se/profile/jgop">Jesús Ortega-Peimbert</a>,
    <a href="https://www.kth.se/profile/quantao?l=en">Quantao Yang</a>,
    <a href="https://www.kth.se/profile/olovand" style="white-space: nowrap;"> Olov Andersson</a>
  </h3>
  <p align="center">
    <a href="https://www.finnbusch.com/OneMap/">Project Website</a> , <a href="https://arxiv.org/pdf/2409.11764">Paper (arXiv)</a>
  </p>
</p>
<p align="center">
  <a href="https://github.com/KTH-RPL/OneMap/actions/workflows/docker-build.yml">
    <img src="https://github.com/KTH-RPL/OneMap/actions/workflows/docker-build.yml/badge.svg" alt="Docker Build">
  </a>
</p>

This repository contains the code for the paper "One Map to Find Them All: Real-time Open-Vocabulary Mapping for
Zero-shot Multi-Object Navigation". We provide a [dockerized environment](#setup-docker) to run the code or
you can [run it locally](#setup-local-without-docker).

In summary we open-source:
- The OneMap mapping and navigation code
- The evaluation code for single- and multi-object navigation
- The multi-object navigation dataset and benchmark
- The multi-object navigation dataset generation code, such that you can generate your own datasets

## Abstract
The capability to efficiently search for objects in complex environments is fundamental for many real-world robot 
applications. Recent advances in open-vocabulary vision models have resulted in semantically-informed object navigation \
methods that allow a robot to search for an arbitrary object without prior training. However, these 
zero-shot methods have so far treated the environment as unknown for each consecutive query.
In this paper we introduce a new benchmark for zero-shot multi-object navigation, allowing the robot to leverage
information gathered from previous searches to more efficiently find new objects. To address this problem we build a
reusable open-vocabulary feature map tailored for real-time object search. We further propose a probabilistic-semantic
map update that mitigates common sources of errors in semantic feature extraction and leverage this semantic uncertainty
for informed multi-object exploration. We evaluate our method on a set of object navigation tasks in both simulation
as well as with a real robot, running in real-time on a Jetson Orin AGX. We demonstrate that it outperforms existing
state-of-the-art approaches both on single and multi-object navigation tasks.
## Setup (Docker)
### 0. Docker 
You will need to have Docker installed on your system. Follow the [official instructions](https://docs.docker.com/engine/install/ubuntu/) to install.
You will also need to have the [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
installed and configured as docker runtime on your system.

### 1. Clone the repository
```
# https
git clone https://github.com/KTH-RPL/OneMap.git
# or ssh
git clone git@github.com:KTH-RPL/OneMap.git
cd OneMap/
```
### 2. Build the Docker Image
The docker image build process will build habitat-sim and download model weights. You can choose to let the container
download the habitat scenes during build, or if you have them already downloaded, you can set `HM3D=LOCAL` and provide
the absolute `HM3D_PATH` to the `versioned_data` directory on your machine in the `.env` file in the root of the repository. 

If you want the container to download the scenes for you, set `HM3D=FULL` in the `.env` file and provide your
Matterport credentials. You can get access for Matterport for free [here](https://matterport.com/partners/meta).
You will not need to provide a `HM3D_PATH` then.
Having configured the `.env` file, you can build the docker image in the root of the repository with:
```
docker compose build
```
The build will take a while as `habitat-sim` is built from source. You can launch the docker container with:
```
bash run_docker.sh
```
and open a new terminal in the container with:
```
docker exec -it onemap-onemap-1 bash
```
## Setup (Local, without Docker)

### 1. Clone the repository
```
# https
git clone https://github.com/KTH-RPL/OneMap.git
# or ssh
git clone git@github.com:KTH-RPL/OneMap.git
cd OneMap/
```
### 2. Install dependencies
```
python3 -m pip install gdown torch torchvision torchaudio meson
python3 -m pip install -r requirements.txt
```
Manually install newer `timm` version:
```
python3 -m pip install --upgrade timm>=1.0.7
```
YOLOV7:
```
git clone https://github.com/WongKinYiu/yolov7
```
Build planning utilities:
```
python3 -m pip install ./planning_cpp/
```
### 3. Download the model weights
```
mkdir -p weights/
```
SED extracted weights:
```
gdown 1D_RE4lvA-CiwrP75wsL8Iu1a6NrtrP9T -O weights/clip.pth
```
YOLOV7 weights and MobileSAM weights:
```
wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6e.pt -O weights/yolov7-e6e.pt
wget https://github.com/ChaoningZhang/MobileSAM/raw/refs/heads/master/weights/mobile_sam.pt -O weights/mobile_sam.pt
```
### 4. Download the habitat data


## Running the code
### 1. Run the example
You can run the code on an example, visualized in [rerun.io](https://rerun.io/) with:
#### Docker
You will need to have [rerun.io](https://rerun.io/) installed on the host for visualization.
Ensure the docker is running and you are in the container as described in the [Docker setup](#setup-docker). Then launch
the rerun viewer **on the host** (not inside the docker) with:
```
rerun
```
and launch the example in the container with:
``` 
python3 habitat_test.py --config config/mon/base_conf_sim.yaml
```
#### Local
Open the rerun viewer and example from the root of the repository with:
```
rerun
python3 habitat_test.py --config config/mon/base_conf_sim.yaml
```
### 2. Run the evaluation
You can reproduce the evaluation results from the paper for single- and multi-object navigation.
#### Single-object navigation
```
python3 eval_habitat.py --config config/mon/eval_conf.yaml
```
This will run the evaluation and save the results in the `results/` directory. You can read the results with:
```
python3 read_results.py --config config/mon/eval_conf.yaml
```
#### Multi-object navigation
```
python3 eval_habitat_multi.py --config config/mon/eval_multi_conf.yaml
```
This will run the evaluation and save the results in the `results_multi/` directory. You can read the results with:
```
python3 read_results_multi.py --config config/mon/eval_multi_conf.yaml
```
#### Dataset generation
While we provide the generated dataset for the evaluation of multi-object navigation, we also release the code to
generate the datasets with varying parameters. You can generate the dataset with
```
python3 eval/dataset_utils/gen_multiobject_dataset.py
```
and change the parameters such as number of objects per episode in the corresponding file.

## Citation
If you use this code in your research, please cite our paper:
```
@misc{busch2024mapallrealtimeopenvocabulary,
      title={One Map to Find Them All: Real-time Open-Vocabulary Mapping for Zero-shot Multi-Object Navigation}, 
      author={Finn Lukas Busch and Timon Homberger and Jesús Ortega-Peimbert and Quantao Yang and Olov Andersson},
      year={2024},
      eprint={2409.11764},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2409.11764}, 
}
```
<p align="center">
  <img src="docs/sys.png" width="900", style="border-radius:10%">
  <h1 align="center">One Map to Find Them All: Real-time Open-Vocabulary Mapping for Zero-shot Multi-Object Navigation</h1>
  <h3 align="center">
    <a href="https://www.kth.se/profile/flbusch?l=en">Finn Lukas Busch</a>,
    <a href="https://www.kth.se/profile/timonh">Timon Homberger</a>,
    <a href="https://www.kth.se/profile/jgop">Jesús Ortega-Peimbert</a>,
    <a href="https://www.kth.se/profile/quantao?l=en">Quantao Yang</a>,
    <a href="https://www.kth.se/profile/olovand" style="white-space: nowrap;"> Olov Andersson</a>
  </h3>
  <p align="center">
    <a href="https://www.finnbusch.com/OneMap/">Project Website</a> , <a href="https://arxiv.org/pdf/2409.11764">Paper (arXiv)</a>
  </p>
</p>
<p align="center">
  <a href="https://github.com/KTH-RPL/OneMap/actions/workflows/docker-build.yml">
    <img src="https://github.com/KTH-RPL/OneMap/actions/workflows/docker-build.yml/badge.svg" alt="Docker Build">
  </a>
</p>

This repository contains the code for the paper "One Map to Find Them All: Real-time Open-Vocabulary Mapping for
Zero-shot Multi-Object Navigation". We provide a [dockerized environment](#setup-docker) to run the code or
you can [run it locally](#setup-local-without-docker).

In summary we open-source:
- The OneMap mapping and navigation code
- The evaluation code for single- and multi-object navigation
- The multi-object navigation dataset and benchmark
- The multi-object navigation dataset generation code, such that you can generate your own datasets

## Abstract
The capability to efficiently search for objects in complex environments is fundamental for many real-world robot 
applications. Recent advances in open-vocabulary vision models have resulted in semantically-informed object navigation \
methods that allow a robot to search for an arbitrary object without prior training. However, these 
zero-shot methods have so far treated the environment as unknown for each consecutive query.
In this paper we introduce a new benchmark for zero-shot multi-object navigation, allowing the robot to leverage
information gathered from previous searches to more efficiently find new objects. To address this problem we build a
reusable open-vocabulary feature map tailored for real-time object search. We further propose a probabilistic-semantic
map update that mitigates common sources of errors in semantic feature extraction and leverage this semantic uncertainty
for informed multi-object exploration. We evaluate our method on a set of object navigation tasks in both simulation
as well as with a real robot, running in real-time on a Jetson Orin AGX. We demonstrate that it outperforms existing
state-of-the-art approaches both on single and multi-object navigation tasks.
## Setup (Docker)
### 0. Docker 
You will need to have Docker installed on your system. Follow the [official instructions](https://docs.docker.com/engine/install/ubuntu/) to install.
You will also need to have the [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
installed and configured as docker runtime on your system.

### 1. Clone the repository
```
# https
git clone https://github.com/KTH-RPL/OneMap.git
# or ssh
git clone git@github.com:KTH-RPL/OneMap.git
cd OneMap/
```
### 2. Build the Docker Image
The docker image build process will build habitat-sim and download model weights. You can choose to let the container
download the habitat scenes during build, or if you have them already downloaded, you can set `HM3D=LOCAL` and provide
the absolute `HM3D_PATH` to the `versioned_data` directory on your machine in the `.env` file in the root of the repository. 

If you want the container to download the scenes for you, set `HM3D=FULL` in the `.env` file and provide your
Matterport credentials. You can get access for Matterport for free [here](https://matterport.com/partners/meta).
You will not need to provide a `HM3D_PATH` then.
Having configured the `.env` file, you can build the docker image in the root of the repository with:
```
docker compose build
```
The build will take a while as `habitat-sim` is built from source. You can launch the docker container with:
```
bash run_docker.sh
```
and open a new terminal in the container with:
```
docker exec -it onemap-onemap-1 bash
```
## Setup (Local, without Docker)

### 1. Clone the repository
```
# https
git clone https://github.com/KTH-RPL/OneMap.git
# or ssh
git clone git@github.com:KTH-RPL/OneMap.git
cd OneMap/
```
### 2. Install dependencies
```
python3 -m pip install gdown torch torchvision torchaudio meson
python3 -m pip install -r requirements.txt
```
Manually install newer `timm` version:
```
python3 -m pip install --upgrade timm>=1.0.7
```
YOLOV7:
```
git clone https://github.com/WongKinYiu/yolov7
```
Build planning utilities:
```
python3 -m pip install ./planning_cpp/
```
### 3. Download the model weights
```
mkdir -p weights/
```
SED extracted weights:
```
gdown 1D_RE4lvA-CiwrP75wsL8Iu1a6NrtrP9T -O weights/clip.pth
```
YOLOV7 weights and MobileSAM weights:
```
wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6e.pt -O weights/yolov7-e6e.pt
wget https://github.com/ChaoningZhang/MobileSAM/raw/refs/heads/master/weights/mobile_sam.pt -O weights/mobile_sam.pt
```
### 4. Download the habitat data


## Running the code
### 1. Run the example
You can run the code on an example, visualized in [rerun.io](https://rerun.io/) with:
#### Docker
You will need to have [rerun.io](https://rerun.io/) installed on the host for visualization.
Ensure the docker is running and you are in the container as described in the [Docker setup](#setup-docker). Then launch
the rerun viewer **on the host** (not inside the docker) with:
```
rerun
```
and launch the example in the container with:
``` 
python3 habitat_test.py --config config/mon/base_conf_sim.yaml
```
#### Local
Open the rerun viewer and example from the root of the repository with:
```
rerun
python3 habitat_test.py --config config/mon/base_conf_sim.yaml
```
### 2. Run the evaluation
You can reproduce the evaluation results from the paper for single- and multi-object navigation.
#### Single-object navigation
```
python3 eval_habitat.py --config config/mon/eval_conf.yaml
```
This will run the evaluation and save the results in the `results/` directory. You can read the results with:
```
python3 read_results.py --config config/mon/eval_conf.yaml
```
#### Multi-object navigation
```
python3 eval_habitat_multi.py --config config/mon/eval_multi_conf.yaml
```
This will run the evaluation and save the results in the `results_multi/` directory. You can read the results with:
```
python3 read_results_multi.py --config config/mon/eval_multi_conf.yaml
```
#### Dataset generation
While we provide the generated dataset for the evaluation of multi-object navigation, we also release the code to
generate the datasets with varying parameters. You can generate the dataset with
```
python3 eval/dataset_utils/gen_multiobject_dataset.py
```
and change the parameters such as number of objects per episode in the corresponding file.

## Citation
If you use this code in your research, please cite our paper:
```
@misc{busch2024mapallrealtimeopenvocabulary,
      title={One Map to Find Them All: Real-time Open-Vocabulary Mapping for Zero-shot Multi-Object Navigation}, 
      author={Finn Lukas Busch and Timon Homberger and Jesús Ortega-Peimbert and Quantao Yang and Olov Andersson},
      year={2024},
      eprint={2409.11764},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2409.11764}, 
}
```
