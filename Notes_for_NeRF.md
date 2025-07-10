# 代码框架


# 实现细节
## 视角向量
>*输入*：坐标位置x 的位置编码以及视角向量d的位置编码  
  *输出*：在坐标位置x的体密度比$\sigma$ 以及视线d方向观察时x处的RGB值C

- 而神经网络必须考虑视线角度的原因是需要考虑反光特性
- 虽然加入视线角度信息会导致模型参数量增大，但是增强了模型的性能
### 视角向量对高频和低频的影响
没有位置编码 生成图像中的高频几何和纹理细节都丢失 体现在图像上就是纹理丰富的地方看起来是模糊的，图像变化剧烈的地方就会有一种虚化的感觉

>**高频和低频**
- 图像：低频是图像的纹理细节
	- 高频：简单地说，图像信号中的高频分量，指的就是图像信号强度（亮度/灰度）变化剧烈的地方，也就是我们常说的边缘（轮廓）。
	- 低频：图像信号中的低频分量，指的就是图像强度（亮度/灰度）变换平缓的地方，也就是大片色块，变化不那么明显的地方。
- 三维场景：低频代表体密度的剧烈变化（高低频可由梯度大小来进行划分）

>**产生问题的原因**：首先我们假设物体上有两个空间上接近的点 $A$ 和 $B$，当其投影到图像上产生的颜色分别是 $C_A$ 和 $C_B$，一般来说，这两个点的位置差异不大的情况下，投影得到的颜色差异也不会很大，这样就导致在某些高频信息或者变化较大的地方会产生信息丢失的情况，如下图最右侧的图像，变化被模糊了或者说平滑了，高频信息变成了低频信息，所以就必须使得神经网络学会去放大这种信息，也就是要同时具备学习低频和高频信息的能力，而传统神经网络只能学习到低频信息，这就需要位置编码来辅助。

通过如下图的方式来进行位置编码![[Pasted image 20250626202400.png]]可以放大两点间位置上的差距 从而学习到高频信息

## coarse & fine
> 采用层级采样策略 

>原因： 由于每条射线都需要在近点和远点之间采样大量的点进行评估，导致计算量大
>先验：射线上大部分区域都是空的，或者是被遮挡，对最终的颜色没有贡献

### 做法
首先采样 $N_C$ 个点，然后进行一种类似softmax的操作，考虑其色彩权重，这样就得到了一些权重大的点，这样就会着重考虑这些点，然后对这些重要的位置进行重新采样（更为密集），然后把两次采样的结果统一送入网络进行训练，然后当反向传播的时候就会着重更新这些重要的点的参数，这样网络就可以学会采样重要的点进行生成

#### coarse 网络
1. 层级采样$N_c$个位置，计算$\hat{W_i}$ 权重；
	![[Pasted image 20250626211208.png]]
2. 依据权重$\hat{W_i}$ 采样出$N_f$个位置
3. 单条光线总采样数为$N_c+N_f$
## 损失函数
![[Pasted image 20250627144612.png]]
c和f代表coarse和fine


## 代码实现
### train（）

parser（）进行输入的解码
样例：
```python
import argparse 

if __name__ == "__main__" : 
    parser = argparse.ArgumentParser(description='parser example')
    parser.add_argument('--seed', default=1, type=int, help='Random seed')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--datapath', default='../../dataset/', type=str, help='dataset path')
    parser.add_argument('--resume', action='store_true', help='resume from checkpoint')

    args = parser.parse_args()
    print('1: ', args.lr)
    print('2: ', args.resume)
    print('3: ', args)

```

## dataloader
`imp.load_source(module, path).Dataset(**args)` 动态加载数据
读入data/nerf_synthetic文件夹里面的文件