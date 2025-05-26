**⭐️⭐️⭐️ Guideline adapted from [RoboPrompt](https://github.com/davidyyd/roboprompt) ⭐️⭐️⭐️**

Important: If you meet problems during below installation, you can check the installation guideline provided by [RVT](https://github.com/NVlabs/RVT).


# Installation

## 1. Initial Setup:
Create a conda environment with Python 3.10 and clone X-ICM repo:
```bash
conda create -n XICM python=3.10
conda activate XICM
pip install pip==24.0 # fixed required for YARR
git clone git@github.com:jiaming-zhou/X-ICM.git
cd X-ICM
```

Please make sure you are using `pip==24.0` when installing the dependencies.

## 2. PyRep and Coppelia Simulator:

Check instructions from the [PyRep](https://github.com/stepjam/PyRep). Paste below for convenience:

PyRep requires version **4.1** of CoppeliaSim. Download: 
- [Ubuntu 16.04](https://downloads.coppeliarobotics.com/V4_1_0/CoppeliaSim_Player_V4_1_0_Ubuntu16_04.tar.xz)
- [Ubuntu 18.04](https://downloads.coppeliarobotics.com/V4_1_0/CoppeliaSim_Player_V4_1_0_Ubuntu18_04.tar.xz)
- [Ubuntu 20.04](https://www.coppeliarobotics.com/previousVersions#)

Once you have downloaded CoppeliaSim, you can unzip the simulator and clone PyRep from git:

```bash
tar -xf <EDIT ME>/PATH/TO/COPPELIASIM.tar.xz
```

Add the following to your *~/.bashrc* file: (__NOTE__: the 'EDIT ME' in the first line)

```bash
export COPPELIASIM_ROOT=<EDIT ME>/PATH/TO/COPPELIASIM/INSTALL/DIR
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
```

Remember to source your bashrc (`source ~/.bashrc`) or 
zshrc (`source ~/.zshrc`) after this.

Install the PyRep package:

```bash
cd PyRep
pip install -r requirements.txt
pip install -e .
```

You should be good to go!
You could try running one of the examples in the *examples/* folder.

If you encounter errors, please use the [PyRep issue tracker](https://github.com/stepjam/PyRep/issues).

## 3. RLBench

Install the RLBench package:
```bash
cd RLBench
pip install -r requirements.txt
pip install -e .
```

## 4. YARR

Install the YARR package:
```bash
cd YARR
pip install -r requirements.txt
pip install -e .
```

## 5. X-ICM

Finally, install the dependencies for X-ICM:
```bash
pip install git+https://github.com/openai/CLIP.git
pip install -r requirements.txt
```

## 6. [Optional] Setup Virtual Display

This is only required if you are running on a remote server without a physical display.

We provide a script to set up the virtual display in Ubuntu 20.04.

```bash
sudo apt update
sudo apt-get reinstall xorg freeglut3-dev libxcb-randr0-dev libxrender-dev libxkbcommon-dev libxkbcommon-x11-0 libavcodec-dev libavformat-dev libswscale-dev
sudo nvidia-xconfig -a --use-display-device=None --virtual=1280x1024

wget https://sourceforge.net/projects/virtualgl/files/2.5.2/virtualgl_2.5.2_amd64.deb/download -O virtualgl_2.5.2_amd64.deb
sudo dpkg -i virtualgl*.deb
rm virtualgl*.deb
nohup sudo X &
```

Any later command using display (e.g., dataset generation and evaluation) should be run with `DISPLAY=:0.0 python ...`.

For more details, please refer to the [PyRep](https://github.com/stepjam/PyRep?tab=readme-ov-file#running-headless).

