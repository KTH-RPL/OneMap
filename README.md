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

## Changes
- [29/07/2026]: Released the CUDA port of the mapping backend as [CuOneMap](https://github.com/finnBsch/CuOneMap), usable from here as an experimental drop-in backend. See [CUDA backend](#cuda-backend-experimental).
- [28/10/2025]: Docker build to CUDA 12.8 for RTX 50 series support. Fixed issues with reading the results for multi-object nav.

## Upcoming Changes
- Change annotation format for multi-object nav to match paper naming, see below.

## CUDA backend (experimental)

The mapping update in `mapping/feature_map.py` is also available as a CUDA implementation,
[CuOneMap](https://github.com/finnBsch/CuOneMap). It is a drop-in replacement that reproduces the same maps to
float tolerance: roughly 3x faster at the map settings shipped here, and up to 40x on large maps with wide blur
kernels. It can keep the feature map in system RAM, so GPU memory stays available for the vision models.

Install it without cloning:

```
pip install git+https://github.com/finnBsch/CuOneMap.git
```

Then enable it in your mapping config:

```yaml
MappingConf:
  use_cuda_backend: True
```

or set `ONEMAP_BACKEND=cuda`. If the module is not installed, OneMap falls back to the PyTorch implementation.

This backend is **experimental**. One preprocessing step differs from the PyTorch version on purpose, so results
are close but not bit-identical — the [CuOneMap README](https://github.com/finnBsch/CuOneMap#differences-from-the-reference)
lists every known difference, along with benchmarks and scripts to verify it against this implementation. We would
love to hear how it works for you: please open an issue with feedback or problems.

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
NOTE: Fix to build habitat-sim:
```
CMAKE_ARGS="-DCMAKE_POLICY_VERSION_MINIMUM=3.5"  python3 -m pip install git+https://github.com/facebookresearch/habitat-sim.git@v0.2.4
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

Create the datasets directory:
```
mkdir -p datasets
```

#### Download HM3D Scene Dataset
You can obtain access to Matterport for free [here](https://matterport.com/partners/meta). Once you have your credentials, download the HM3D scenes:

```
python3 -m habitat_sim.utils.datasets_download \
  --username <MATTERPORT_ID> --password <MATTERPORT_SECRET> \
  --uids hm3d_train_v0.2 \
  --data-path datasets

python3 -m habitat_sim.utils.datasets_download \
  --username <MATTERPORT_ID> --password <MATTERPORT_SECRET> \
  --uids hm3d_val_v0.2 \
  --data-path datasets
```

Create the `hm3d_v0.2` symlink if not already there:
```
ln -s datasets/scene_datasets/hm3d datasets/scene_datasets/hm3d_v0.2
```

#### Download Navigation Episode Datasets

Download the multi-object episodes dataset:
```
gdown 1lBpYxXRjj8mDSUTI66xv0PfNd-vdSbNj -O multiobject_episodes.zip
unzip multiobject_episodes.zip
mv multiobject_episodes datasets/
rm multiobject_episodes.zip
```

Download ObjectNav HM3D v1 dataset:
```
wget https://dl.fbaipublicfiles.com/habitat/data/datasets/objectnav/hm3d/v1/objectnav_hm3d_v1.zip
unzip objectnav_hm3d_v1.zip
mv objectnav_hm3d_v1 datasets/
rm objectnav_hm3d_v1.zip
```

Download ObjectNav HM3D v2 dataset (required for multi-object navigation):
```
wget https://dl.fbaipublicfiles.com/habitat/data/datasets/objectnav/hm3d/v2/objectnav_hm3d_v2.zip
unzip objectnav_hm3d_v2.zip
mv objectnav_hm3d_v2 datasets/
rm objectnav_hm3d_v2.zip
```

Your `datasets/` directory should now contain:
```
datasets/
├── scene_datasets/
│   ├── hm3d/
│   └── hm3d_v0.2/
├── versioned_data/hm3d-0.2/hm3d/
├── multiobject_episodes/
├── objectnav_hm3d_v1/
└── objectnav_hm3d_v2/
```

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

**Note** that to reproduce the paper results for single-object navigation, it is advised to use the [eval/s_eval](https://github.com/KTH-RPL/OneMap/tree/eval/s_eval) branch.
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

**Note that the resulting table will report multiple metrics which correspond to the following names in the paper (table->paper name): SPL->PPL, Progress->PR, s->SR, s_spl->SPL.**

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
@INPROCEEDINGS{11128393,
      author={Busch, Finn Lukas and Homberger, Timon and Ortega-Peimbert, Jesús and Yang, Quantao and Andersson, Olov},
      booktitle={2025 IEEE International Conference on Robotics and Automation (ICRA)}, 
      title={One Map to Find Them All: Real-time Open-Vocabulary Mapping for Zero-shot Multi-Object Navigation}, 
      year={2025},
      volume={},
      number={},
      pages={14835-14842},
      keywords={Training;Three-dimensional displays;Uncertainty;Navigation;Semantics;Benchmark testing;Search problems;Probabilistic logic;Real-time systems;Videos},
      doi={10.1109/ICRA55743.2025.11128393},
}
```
