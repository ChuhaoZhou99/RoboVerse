# 🐳 Docker for NVIDIA RTX50 Series GPUs

## ⚙️ Prerequisites
This document provides a step-by-step Docker configuration tutorial for RTX50 series GPUs. For RTX50 series GPUs, the following environments are required.

### Environment Requirements

| Component      | Version   | Notes                          |
|----------------|-------------------|--------------------------------|
| 🐧 OS           | Ubuntu ≥ 22.04     | Required by IsaacLab        |
| 🐍 Python       | python == 3.10     | Required by multiple simulators        |
| 🔥 PyTorch      | torch ≥ 2.7.1      | Required by RTX50 series GPUs        |
| 🚀 CUDA         | CUDA ≥ 12.8        | Required by RTX50 series GPUs |


## 🛠️ Installation
### 1️⃣ Step 1: Install Docker
Please make sure that you have install `docker` in the officially recommended way. Otherwise, please refer to the [official guide](https://docs.docker.com/engine/install/ubuntu/).

### 2️⃣ Step 2: Pull the official NVIDIA image
To make sure the docker environment supports RTX50 series GPUs and cuda 12.8. Please pull the official Ubuntu 22.04 base image that supports cuda 12.8 from NVIDIA by running the following commands:
```bash
docker pull nvidia/cuda:12.8.0-base-ubuntu22.04
```

### 3️⃣ Step 3: Setup GPU toolkit and GUI in host
To call the GPUs via docker, please install the [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-container-toolkit) in your host follwoing the [official guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). 

Besides, please run the following commands to make sure that the docker can access the host GUI:
```bash
xhost +local:docker

sudo systemctl restart docker
```

### 4️⃣ Step 4: Setup docker environments
Please run the base image with GPU supporting and install necessary development tools (build-essential, CMake, git, etc.).

```bash
docker run --gpus all -it nvidia/cuda:12.8.0-base-ubuntu22.04

apt-get update && apt-get install -y --no-install-recommends build-essential cmake git curl wget ca-certificates pkg-config software-properties-common unzip nano sudo
```

Please make sure that you have installed `Anaconda3` in the officially recommended way. Otherwise, please refer to the [official guide](https://www.anaconda.com/docs/getting-started/anaconda/install#linux-installer).

Then, setup the conda environment with `python==3.10` for RoboVerse:
```bash
conda create -n roboverse python=3.10
```

### 5️⃣ Step 5: Setup RoboVerse-IsaacLab environments
Please pull the RoboVerse official code repository:
```bash
git clone https://github.com/RoboVerseOrg/RoboVerse.git

cd RoboVerse
```

The environment in the pyproject.toml is currently not compatible for NVIDIA RTX50 series GPUs. Please use `pip` to install isaacsim manually.
```bash
pip install protobuf
pip install pyglet
pip install isaacsim==4.2.0.2
pip install isaacsim-extscache-physics==4.2.0.2
pip install isaacsim-extscache-kit==4.2.0.2
pip install isaacsim-extscache-kit-sdk==4.2.0.2
```

Please install the IsaacLab dependencies by running following commands:
```bash
cd third_party

wget https://codeload.github.com/isaac-sim/IsaacLab/zip/refs/tags/v1.4.1 -O IsaacLab-1.4.1.zip && unzip IsaacLab-1.4.1.zip

cd IsaacLab-1.4.1

sed -i '/^EXTRAS_REQUIRE = {$/,/^}$/c\EXTRAS_REQUIRE = {\n    "sb3": [],\n    "skrl": [],\n    "rl-games": [],\n    "rsl-rl": [],\n    "robomimic": [],\n}' source/extensions/omni.isaac.lab_tasks/setup.py

./isaaclab.sh -i
```

After installing the IsaacLab, the torch will be modified to 2.4.0, reinstall the torch to 2.7.1. The `torch==2.4.0` will not be compatible with NVIDIA RTX50 series GPUs.
```bash
pip install --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Finally, please install the necessary libraries required by IsaacLab.
```bash
pip install rootutils
pip install tyro
pip install loguru
pip install open3d
```

### 6️⃣ Step 6: Setup RoboVerse-Mujoco environments
After setting up issaclab, mujoco can be easily installed with the following command:
```bash
pip install mujoco 
pip install dm-control
```

### 7️⃣ Step 7: Setup RoboVerse Reinforcement Learning environments
RoboVerse provides two reinforcement learning demos: [PPO Reaching](https://roboverse.wiki/metasim/get_started/advanced/rl_example/0_ppo_reaching#ppo-reaching) and [FastTD3 Humanoid](https://roboverse.wiki/metasim/get_started/advanced/rl_example/1_fttd3_humanoid). To run these two demos, please follow the steps below to setup your environments.

Setup the PPO environments.
```bash
pip install stable-baselines3
```

Setup the FastTD3 environments.
```bash
pip install mujoco-mjx
pip install dm-control
pip install jax[cuda12]
pip install wandb
pip install tensordict
```

## ⚠️ Trouble Shooting
The following issues may arise in various modules during Docker configuration. 	Corresponding solutions are provided for each case.

### RoboVerse-IsaacLab
### 1. "\[omni.gpu_foudation_factory.plugin] Failed to create any GPU devices, including an attempt with compatibility mode."
This problem is due to incorrect startup method of docker images and the docker cannot access the phsical GPUs in the host. 

Save the running docker container to the docker images.
```bash
docker ps -a

docker commit your_container_id your_image_name
```
Rerun the image by following setting, which allows the docker to call the physical GPUs in the host.
```bash
docker run -it --gpus all \
-e DISPLAY=$DISPLAY \
-v /tmp/.X11-unix:/tmp/.X11-unix \
-v /dev/dri:/dev/dri \
-v /usr/share/vulkan/icd.d:/usr/share/vulkan/icd.d:ro \
-v /usr/share/vulkan/implicit_layer.d:/usr/share/vulkan/implicit_layer.d:ro \
-e NVIDIA_VISIBLE_DEVICES=all \
-e NVIDIA_DRIVER_CAPABILITIES=all \
-e XDG_RUNTIME_DIR=/run/user/$(id -u) \
your_image_name /bin/bash
```

### 2. Errors about shared libraries not found. For example, "OSError: libGL.so.1: cannot open shared object file: No such file or directory."
This issue arises from the absence of required dynamic libraries.

Please install the necessary dynamic libraries for IsaacLab.
```bash
sudo apt-get install libgl1-mesa-glx libsm6 libxt6 libglu1-mesa
```

### 3. Errors related to numpy version. "Error: partially initialized module 'numpy.core.arrayprint' has no attribute 'array2string'."
The RoboVerse-IsaacLab is sensitive to numpy version, currently it only supports for `numpy < 2`. Please install the `numpy < 2`.
```bash
pip install numpy==1.26.4
```

### 4. "TypeError: array.\_\_init\_\_() got an unexpected keyword argument 'owner'."
This seems a bug in the omni library function.

The corresponding file is:
```
/home/anaconda3/envs/roboverse/lib/python3.10/site-packages/isacsim/extscache/omni.replicator.core-1.11.20+186.1.8.1x64.r.cp310/omni/replicatorcore/scripts/utils/annotator_utils.py
```
Delete the `owner` parameter in corresponding function (line 341) in the annotator_utils.py file.


### RoboVerse-Mujoco
### 1. "ValueError: Image width 1024 > framebuffer width 640. Either reduce the image width or specify a larger offscreen framebuffer in the model XML using clause."
This is a bug from the new version of RoboVerse (After the `/RoboVerse/metasim/sim/mujoco/mujoco.py` updating from 2025.07.11).

A temporary solution is to hard code the `camera.width=640` and `camera.height=480` in `/RoboVerse/metasim/sim/mujoco/mujoco.py`
lines 341.

### RoboVerse-Reinforcement Learning
### 1. "ModuleNotFoundError: No mudule named 'metasim'"
This issue is due to the project path is not correctly setting.

Add project path at the beginning of `/RoboVerse/get_started/rl/fast_td3/1_fttd3_humanoid.py`
```python
roboverse_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

sys.path.append(roboverse_path)
```

### 2. "ImportError: Cannot initialize a headless EGL display."
This issue is due to that the docker could not find the `EGL` engine for rendering.

You can manually set environment variables in docker:
```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl 
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
```

Otherwise, you can add these environment variables in the startup command of the docker image:
```bash
docker run -it --gpus all \
-e DISPLAY=$DISPLAY \
-v /tmp/.X11-unix:/tmp/.X11-unix \
-v /dev/dri:/dev/dri \
-v /usr/share/vulkan/icd.d:/usr/share/vulkan/icd.d:ro \
-v /usr/share/vulkan/implicit_layer.d:/usr/share/vulkan/implicit_layer.d:ro \
-e NVIDIA_VISIBLE_DEVICES=all \
-e NVIDIA_DRIVER_CAPABILITIES=all \
-e XDG_RUNTIME_DIR=/run/user/$(id -u) \
-e MUJOCO_GL=egl \
-e PYOPENGL_PLATFORM=egl \
-e LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
roboverse /bin/bash
```
### 3. "AttributeError: 'NoneType' object has no attribute 'eglQueryString'"
This issue is due to the lack of necessary dynamic libraries for EGL engine. Please install them via the following commands:
```bash
sudo apt-get install libegl1 libgl1-mesa-glx
```

### 4. "jaxlib.\_jax.XlaRuntimeError: FAILED\_PRECONDITION: DNN library initialization failed"
The FastTD3 requires `cudnn >= 9.8.0`. Please install new version of cudnn.
```bash
pip install nvidia-cudnn-cu12==9.10.2.21
```

## ✅ Done
Based on this environment, you can use the simulator `mujoco`, `mujoco-mjx`, `IssacLab v1.4` and run the reinforcement learning algorithm `PPO` and `FastTD3`.

If you encounter any other errors with docker, please let us know by raising an [issue](https://github.com/RoboVerseOrg/RoboVerse/issues), and we will fix it as soon as possible.







