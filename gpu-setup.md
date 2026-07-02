# NVIDIA Container Toolkit Setup

## 1. Fix the sources list

```bash
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
```

## 2. Install

```bash
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
```

## 3. Configure Docker and restart

```bash
sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker
```

## 4. Verify GPU is visible to Docker

```bash
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```
