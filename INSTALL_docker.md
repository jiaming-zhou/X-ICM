Clone the X-ICM repo:
```bash
git clone git@github.com:jiaming-zhou/X-ICM.git
```

Download docker Image and create Container:
```bash
### pull the docker Image from dockerhub:
docker pull jiamingzhou2472/agnostos:v1

### initialize the container
sudo docker run -it -v /tmp/.X11-unix/:/tmp/.X11-unix:rw -v /usr/lib/nvidia:/usr/lib/nvidia -e SDL_VIDEO_GL_DRIVER=libGL.so.1.7.0 -e DISPLAY=$DISPLAY -e NVIDIA_VISIBLE_DEVICES=all -e  NVIDIA_DRIVER_CAPABILITIES=all --gpus=all -p 6666:22 -v /path/to/source_files:/path/to/target_files --name agnostos jiamingzhou2472/agnostos:v1 bash
```

After loading, use the "XICM" conda environment:
```bash
conda activate XICM
```

Re-install the packages:
```bash
cd X-ICM
pip install --no-deps -e YARR
pip install -e PyRep
pip install -e RLBench

### you can disable NCCL from using shared memory
export NCCL_SHM_DISABLE=1
```
