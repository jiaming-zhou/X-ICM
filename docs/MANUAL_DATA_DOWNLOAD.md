# 📊 AGNOSTOS Benchmark Data Manual Download

The benchmark consists of two parts (all data are available at [huggingface](https://huggingface.co/datasets/Jiaming2472/AGNOSTOS)):
- 📚 18 seen tasks for training (140G in total, split into five files), links:

    [[seen_tasks.part_aa]](https://huggingface.co/datasets/Jiaming2472/AGNOSTOS/resolve/main/seen_tasks.part_aa?download=true) | [[seen_tasks.part_ab]](https://huggingface.co/datasets/Jiaming2472/AGNOSTOS/resolve/main/seen_tasks.part_ab?download=true) | [[seen_tasks.part_ac]](https://huggingface.co/datasets/Jiaming2472/AGNOSTOS/resolve/main/seen_tasks.part_ac?download=true) | [[seen_tasks.part_ad]](https://huggingface.co/datasets/Jiaming2472/AGNOSTOS/resolve/main/seen_tasks.part_ad?download=true) | [[seen_tasks.part_ae]](https://huggingface.co/datasets/Jiaming2472/AGNOSTOS/resolve/main/seen_tasks.part_ae?download=true)
- 🔍 23 unseen tasks for cross-task testing (20.2GB, one single file), link:

    [[unseen_tasks.tar]](https://huggingface.co/datasets/Jiaming2472/AGNOSTOS/resolve/main/unseen_tasks.tar)

After downloading, process the files:

```bash
### for seen task data, combine all five files
cat seen_tasks.part_* > seen_tasks.tar
### check the file, it should be "8217d78779acfd2873d0f55849c8efcc"
md5sum seen_tasks.tar

tar -xvf seen_tasks.tar
tar -xvf unseen_tasks.tar
```

Creating symbolic links to the sub-folder `data`:
```bash
cd X-ICM
mkdir data
ln -s /path/to/seen_tasks data/
ln -s /path/to/unseen_tasks data/
```

# 🤖 X-ICM Model Manual Download

Download our pre-trained dynamics diffusion model from [[dynamics_diffusion.tar]](https://huggingface.co/Jiaming2472/X-ICM/resolve/main/dynamics_diffusion.tar?download=true) for cross-task in-context sample selection.

After downloading, extract and create a symbolic link to the sub-folder `data`.
```bash
tar -xvf dynamics_diffusion.tar

cd X-ICM
ln -s /path/to/dynamics_diffusion data/
```
