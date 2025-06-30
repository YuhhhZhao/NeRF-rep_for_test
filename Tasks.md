# Tasks need to complete
1. [x] 阅读NeRF论文和代码 做出注释 与框架代码进行比对 ✅ 2025-06-28
2. [x] 完成Datasets class`src/datasets/nerf/blender.py` ✅ 2025-06-28
	- `__init__`: This function is responsible for loading the specified format file from disk, calculating and storing it in a specific form.
	- `__getitem__`: This function is responsible for providing the input required for training and the ground truth output to the model at runtime. For example, for NeRF, it provides 1024 rays and 1024 RGB values.
	- `__len__`: This function returns the number of training or testing samples. The index value obtained from `__getitem__` is usually in the range [0, len-1].
	 debug
3. *阅读Model代码* 
4. Renderer `src/models/nerf/renderer/volume_renderer.py`
5. Trainer and Evaluator `src/train/trainers/nerf.py` `src/evaluators/nerf.py`

- 学习tensorboard将结果可视化