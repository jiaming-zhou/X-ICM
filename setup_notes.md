conda create -n xicm_5090 python=3.11 -y
conda activate xicm_5090

# 先装 torch/vllm 5090 新栈
# 然后：
./setup_xicm_env.sh

source env_coppelia.sh
bash scripts/eval_XICM.sh "0" 1 Qwen2.5.7B.instruct 1 0 "random" false

Recommended 5090 environment:
- Python 3.11
- PyTorch 2.10.0+cu128
- CUDA runtime 12.8
- GPU: NVIDIA GeForce RTX 5090, compute capability sm_120

记下来了。你后面在其他 conda 环境里复现，可以按这个 SOP 来。

# 无 sudo 安装 `xvfb-run` 成功流程

## 1. 进入目标 conda 环境

```bash
conda activate <your_env>
```

例如：

```bash
conda activate xicm_5090
```

---

## 2. 不推荐的路线

不要优先用这个：

```bash
conda install -c conda-forge xorg-x11-server-xvfb-cos7-x86_64 -y
```

你之前遇到的问题是它装出来的 `Xvfb` 依赖旧库：

```text
libcrypto.so.10: cannot open shared object file
```

这个路线会很麻烦。

---

## 3. 正确安装方式

用这个：

```bash
conda install -c conda-forge xorg-xserver-xvfb -y
```

安装后检查：

```bash
find $CONDA_PREFIX -name "Xvfb" -o -name "xvfb-run"
```

你成功时看到的是类似：

```bash
$CONDA_PREFIX/bin/Xvfb
$CONDA_PREFIX/x86_64-conda-linux-gnu/sysroot/usr/bin/xvfb-run
```

---

## 4. 加入 PATH

```bash
export PATH=$CONDA_PREFIX/bin:$PATH
```

然后确认：

```bash
which Xvfb
which xvfb-run
```

---

## 5. 测试 xvfb-run

```bash
xvfb-run -a -e /tmp/xvfb_test.log \
-s "-screen 0 1024x768x24 +extension GLX +render -noreset" \
bash -lc 'echo DISPLAY=$DISPLAY; sleep 2'

echo $?
cat /tmp/xvfb_test.log
```

成功标志：

```text
DISPLAY=:99
0
```

---

# CoppeliaSim / RLBench / X-ICM 运行前设置

```bash
export PATH=$CONDA_PREFIX/bin:$PATH
export COPPELIASIM_ROOT=/home/xli990/software/CoppeliaSim
export LD_LIBRARY_PATH=$COPPELIASIM_ROOT:$LD_LIBRARY_PATH
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
unset QT_QPA_PLATFORM
```

重点：

```bash
unset QT_QPA_PLATFORM
```

不要用：

```bash
export QT_QPA_PLATFORM=offscreen
```

否则容易出现：

```text
This plugin does not support createPlatformOpenGLContext!
Error: signal 11
```

---

# 测试 CoppeliaSim

```bash
xvfb-run -a -e /tmp/coppelia_xvfb.log \
-s "-screen 0 1024x768x24 +extension GLX +render -noreset" \
timeout 30s $COPPELIASIM_ROOT/coppeliaSim.sh -h

echo $?
cat /tmp/coppelia_xvfb.log
```

如果看到：

```text
simulator launched
OpenGL3Renderer: load succeeded
Vision: load succeeded
```

说明基本成功。

如果最后是：

```text
124
The X11 connection broke
```

这是因为 `timeout 30s` 到时间强制结束，不是核心错误。

---

# 正式跑 X-ICM / RLBench

把原命令外面包一层：

```bash
xvfb-run -a -e /tmp/xicm_xvfb.log \
-s "-screen 0 1024x768x24 +extension GLX +render -noreset" \
python main.py ...
```

如果原来是脚本：

```bash
bash scripts/eval_XICM.sh ...
```

就改成：

```bash
xvfb-run -a -e /tmp/xicm_xvfb.log \
-s "-screen 0 1024x768x24 +extension GLX +render -noreset" \
bash scripts/eval_XICM.sh ...
```

---

# 可选：写入 conda 自动激活脚本

```bash
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
mkdir -p $CONDA_PREFIX/etc/conda/deactivate.d
```

```bash
cat > $CONDA_PREFIX/etc/conda/activate.d/xvfb_coppeliasim.sh <<'EOF'
export OLD_PATH_FOR_XVFB=$PATH
export OLD_LD_LIBRARY_PATH_FOR_XVFB=$LD_LIBRARY_PATH
export OLD_QT_QPA_PLATFORM_PLUGIN_PATH_FOR_XVFB=$QT_QPA_PLATFORM_PLUGIN_PATH
export OLD_QT_QPA_PLATFORM_FOR_XVFB=$QT_QPA_PLATFORM

export PATH=$CONDA_PREFIX/bin:$PATH
export COPPELIASIM_ROOT=/home/xli990/software/CoppeliaSim
export LD_LIBRARY_PATH=$COPPELIASIM_ROOT:$LD_LIBRARY_PATH
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
unset QT_QPA_PLATFORM
EOF
```

```bash
cat > $CONDA_PREFIX/etc/conda/deactivate.d/xvfb_coppeliasim.sh <<'EOF'
export PATH=$OLD_PATH_FOR_XVFB
export LD_LIBRARY_PATH=$OLD_LD_LIBRARY_PATH_FOR_XVFB
export QT_QPA_PLATFORM_PLUGIN_PATH=$OLD_QT_QPA_PLATFORM_PLUGIN_PATH_FOR_XVFB
export QT_QPA_PLATFORM=$OLD_QT_QPA_PLATFORM_FOR_XVFB

unset COPPELIASIM_ROOT
unset OLD_PATH_FOR_XVFB
unset OLD_LD_LIBRARY_PATH_FOR_XVFB
unset OLD_QT_QPA_PLATFORM_PLUGIN_PATH_FOR_XVFB
unset OLD_QT_QPA_PLATFORM_FOR_XVFB
EOF
```

之后重新激活环境：

```bash
conda deactivate
conda activate <your_env>
```

再测试：

```bash
xvfb-run -a bash -lc 'echo DISPLAY=$DISPLAY'
```

成功输出 `DISPLAY=:99` 就可以。
