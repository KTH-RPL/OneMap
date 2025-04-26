FROM dsalvat1/cudagl:12.3.1-runtime-ubuntu22.04

# Declare the HM3D argument
ARG HM3D

ARG HM3D_PATH

ARG MATTERPORT_ID
ARG MATTERPORT_SECRET

# Validate HM3D argument and show usage
RUN if [ -z "${HM3D}" ]; then \
        echo -e "\e[31mERROR: HM3D argument is required to be set in the .env file. Please specify one of:\e[0m"; \
        echo -e "\e[36m  - LOCAL: Requires additional HM3D_PATH argument pointing to local HM3D dataset\e[0m"; \
        echo -e "\e[36m  - MINI:  Downloads the HM3D mini_val dataset\e[0m"; \
        echo -e "\e[36m  - FULL:  Downloads the complete HM3D dataset\e[0m"; \
        exit 1; \
    elif [ "${HM3D}" = "LOCAL" ]; then \
        if [ -z "${HM3D_PATH}" ]; then \
            echo -e "\e[31mERROR: When using HM3D=LOCAL, you must provide HM3D_PATH argument\e[0m"; \
            echo -e "\e[36mUsage: docker build --build-arg HM3D=LOCAL --build-arg HM3D_PATH=/path/to/hm3d ...\e[0m"; \
            exit 1; \
        fi; \
        echo -e "\e[32mUsing local HM3D dataset from: ${HM3D_PATH}\e[0m"; \
    elif [ "${HM3D}" = "MINI" ]; then \
        echo -e "\e[32mWill download HM3D mini_val dataset\e[0m"; \
    elif [ "${HM3D}" = "FULL" ]; then \
        echo -e "\e[32mWill download complete HM3D dataset\e[0m"; \
    else \
        echo -e "\e[31mERROR: Invalid HM3D value '${HM3D}'. Must be one of: LOCAL, MINI, FULL\e[0m"; \
        exit 1; \
    fi

    
WORKDIR /onemap


RUN apt update && apt install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    vim \
    ca-certificates \
    pkg-config \
    wget \
    zip \
    libeigen3-dev \
    ninja-build \
    python3.10-dev \
    python3-pip \
    python-is-python3\
    libjpeg-dev libglm-dev libgl1-mesa-glx libegl1-mesa-dev mesa-utils xorg-dev freeglut3-dev \
    unzip &&\
    apt-get clean all &&\
    rm -rf /var/lib/apt/lists/*

# #
COPY requirements.txt .
RUN python3 -m pip install gdown torch torchvision torchaudio meson
RUN python3 -m pip install -r requirements.txt
RUN python3 -m pip install --upgrade timm>=1.0.7
RUN CMAKE_ARGS="-DCMAKE_POLICY_VERSION_MINIMUM=3.5"  python3 -m pip install git+https://github.com/facebookresearch/habitat-sim.git@v0.2.4
# #
# #
RUN git clone https://github.com/WongKinYiu/yolov7
RUN mkdir -p weights
RUN gdown 1D_RE4lvA-CiwrP75wsL8Iu1a6NrtrP9T -O weights/clip.pth
RUN wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6e.pt -O weights/yolov7-e6e.pt
RUN wget https://github.com/ChaoningZhang/MobileSAM/raw/refs/heads/master/weights/mobile_sam.pt -O weights/mobile_sam.pt
# #
# # # Copy the project code
COPY ./config ./config
COPY ./eval ./eval
COPY ./mapping ./mapping
COPY ./planning ./planning
COPY ./planning_cpp ./planning_cpp
COPY ./spot_utils ./spot_utils
COPY ./vision_models ./vision_models
COPY ./eval_habitat.py .
COPY ./eval_habitat_multi.py .
COPY ./onemap_utils ./onemap_utils
COPY ./habitat_test.py ./
RUN python3 -m pip install ./planning_cpp/
RUN mkdir datasets

ENV PYTHONPATH="${PYTHONPATH}:/onemap/"

RUN  if [ "$HM3D" = "MINI" ] ; then python3 -m habitat_sim.utils.datasets_download \
  --username $MATTERPORT_ID --password $MATTERPORT_SECRET \
  --uids hm3d_minival_v0.2 \
  --data-path datasets ; fi
# 
RUN  if [ "$HM3D" = "FULL" ] ; then python3 -m habitat_sim.utils.datasets_download \
  --username $MATTERPORT_ID --password $MATTERPORT_SECRET \
  --uids hm3d_train_v0.2 \
  --data-path datasets && \
python3 -m habitat_sim.utils.datasets_download \
  --username $MATTERPORT_ID --password $MATTERPORT_SECRET \
  --uids hm3d_val_v0.2 \
  --data-path datasets ;  fi

# If we use the local dataset, symlinks get broken, we need to recreate a "local" symlink
RUN if [ "$HM3D" = "LOCAL" ] ; then mkdir -p /onemap/datasets/scene_datasets/hm3d && \
ln -s /onemap/datasets/versioned_data/hm3d-0.2/hm3d/train/hm3d_basis.scene_dataset_config.json /onemap/datasets/scene_datasets/hm3d/hm3d_basis.scene_dataset_config.json ; fi

RUN gdown 1lBpYxXRjj8mDSUTI66xv0PfNd-vdSbNj -O multiobject_episodes.zip
RUN unzip multiobject_episodes.zip
RUN mv multiobject_episodes datasets/ && rm multiobject_episodes.zip
RUN wget https://dl.fbaipublicfiles.com/habitat/data/datasets/objectnav/hm3d/v1/objectnav_hm3d_v1.zip
RUN unzip objectnav_hm3d_v1.zip
RUN mv objectnav_hm3d_v1 datasets/ && rm objectnav_hm3d_v1.zip


ENTRYPOINT ["sh", "-c", "if [ \"$HM3D\" = \"LOCAL\" ]; then ln -s $(ls /onemap/datasets/versioned_data/hm3d-0.2/hm3d | grep -v \"hm3d_basis.scene_dataset_config.json\" | sed \"s#^#/onemap/datasets/versioned_data/hm3d-0.2/hm3d/#\") /onemap/datasets/scene_datasets/hm3d/; fi && exec \"$@\"", "--"]

CMD ["/bin/bash"]
