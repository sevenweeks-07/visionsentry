git clone https://github.com/triton-inference-server/tensorrtllm_backend.git && cd tensorrtllm_backend
apt-get update
apt-get install git-lfs
git lfs install
git submodule update --init --recursive

curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

apt-get update
apt-get install -y nvidia-container-toolkit
systemctl restart docker

cd tensorrtllm_backend/

docker run --rm -ti \
  --net=host \
  --shm-size=16g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v `pwd`:/mnt \
  -w /mnt \
  --gpus all \
  nvcr.io/nvidia/tritonserver:25.05-trtllm-python-py3 bash

