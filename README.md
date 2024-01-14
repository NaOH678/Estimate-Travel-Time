# 基于路径的城市旅行时间预测


## 实验目的

- 查阅相关文献，了解现有ETA相关技术；
- 根据相关文献中的不足之处提出一种可行的方法；
- 了解小型科研的基本流程。



## 实验环境

- macOS 13.0

- Python 3.10.1

- PyTorch 



## 问题背景

到达时间预估(Estimate Travel Time，ETA )

### 目标

从某一个时刻出发，预测到达目的地的时间。

 ![image](https://github.com/NaOH678/Estimate-Travel-Time/assets/112929756/87ca29d2-6653-40b3-91e6-41d8283641af)


### 交通工具

出租车、公交车、货车、电动车、自行车。根据开题答辩中老师和学长学姐们的建议，我决定做城市旅行时间预测，交通工具是出租车。

### 下游应用

- 路径规划
- 智能定价
- 车辆调度

ETA有着广泛的下游应用，与我们的生活密切相关。所以如果能准确的预测到达时间，那么将会对我们的生活提供更多的便利。





## 问题定义与描述

### 轨迹

给定轨迹
$$
T = \{p_1, p_2, \cdots, p_{|T|}\}，
$$
其中|T|是该点中包含轨迹点的数量。每一个轨迹点$p_i$包括三个部分组成,分别是**经度，纬度，时间戳**$p_i.{lat}，p_i.{lng}，p_i.{ts}$。



### 定义2 外部因素

对于每段轨迹，我们记录一些外部因子：

- 开始时间：timeID（0～1439）
- 星期：weekID（0～6）
- 司机：driverID（str）
- 路程：dist
- 日期：dateID（0～30）
- 行驶状态：status（1 or 0）

具体每个因子的含义将在数据集介绍部分说明。



### 任务

输入轨迹T和外部因素，输出旅行时间time。



## 相关工作与现存问题



<img width="923" alt="image" src="https://github.com/NaOH678/Estimate-Travel-Time/assets/112929756/3fc4e09d-3725-4fdd-9444-e5b10007cb90">

- AVG

  这种方法是ETA相关问题的baseline方法。基于历史数据，对于计算相同时间段出发的行程的平均速度。然后根据我们需要预测轨迹的出发时间找到相应的平均速度，根据路程计算旅行时间。可以看出这种方法是十分粗略的。没有考虑轨迹点之间的联系，以及诸多外部因素。

- GBDT

  根据实验三中方法，提取出相关特征，使用XGBoost进行预测。这种方法属于传统机器学习方法的代表。依赖人工提取特征，难以拟合该更加复杂的交通时空关系。

- 深度学习

  从2018年开始，逐渐出现了基于深度学习的预测方法。本次课程设计主要参考的是2018年的DeepTTE。

  RNN：捕捉轨迹数据的时间依赖；

  CNN：捕捉相邻轨迹点之间的空间依赖。



## 数据集介绍

**数据集**：成都市2014年8月3日至8月23日出租车轨迹数据

- 轨迹点；
- 时间间隔：轨迹点相对应，起始点的时间为0；
- 距离间隔 ：与轨迹点相对应，起始点距离为0；
- 开始时间：timeID ，将一天按照每分钟为单位进行划分，取值为**0~1439**;
- 星期：weekID，表示当天是星期几，取值为**0~6**;
- 司机：driverID，一个数字字符串。用来区别不同驾驶员；
- 路程：dists；
- 日期：dateID，表示当天的日期，取值为**0~30**;
- 行驶状态：status，取值为**1（载客）**或**0（空载）**



共18000条数据，10800条作为训练集，3600条作为验证集，最后3600条作为测试集。轨迹点的平均采样时间为60s。数据集是按照时间顺序进行划分的。

数据集示例如下图所示：

<img width="271" alt="image" src="https://github.com/NaOH678/Estimate-Travel-Time/assets/112929756/71333865-3e3c-4083-a230-d3d520465b2e">


## 项目框架

本项目的整体示意图如下：

<img width="898" alt="image" src="https://github.com/NaOH678/Estimate-Travel-Time/assets/112929756/3fd6f484-5302-471f-88a7-54c30066ceca">


- 首先将轨迹数据输入搭建好的**地理卷积层**中，获取**空间依赖信息**；
- 使用**特征提取层**将**(行驶状态、路段长度、司机ID、开始时间、星期、日期)**进行Embedding；、
- 然后将上述两者进行拼接，输入**LSTM**中，利用**Attention机制**进行特征融合，学习**时间依赖特征**；
- 将LSTM的结果输入**多任务学习层**，计算整体损失和分段损失的**加权求和损失函数**作为模型的损失函数，进行反向传播。
- 最后输出预测时间。



## 挑战与解决方案

### 挑战一

- 挑战：变量提取。如何提取出有效的变量输入模型？
- 对策：查阅相关文献，实验前进行**案例分析、回归分析**。提取相关特征，构建一个**Attribute Layer**。

#### 案例分析

1. 验证**路径对旅行时间的影响**，选取两条路程都为3.21KM左右的轨迹，出发时间相似，距离市中心的路程也比较相似。但是他们的旅行时间差异比较大。黄色路径的行程用时1087s；橙色行程用时840s。

​                         <img width="204" alt="image" src="https://github.com/NaOH678/Estimate-Travel-Time/assets/112929756/3c0c05e9-2d3a-4b38-9b9c-3a8122f00442">  <img width="275" alt="image" src="https://github.com/NaOH678/Estimate-Travel-Time/assets/112929756/8fdb338d-f7e8-4e10-b9e9-90c76c551f03">  <img width="292" alt="image" src="https://github.com/NaOH678/Estimate-Travel-Time/assets/112929756/f6cb8d28-1521-4398-b13a-3cffbcdef2d3">


说明了路径对旅行时间的影响是很大的，我们需要学习不同路径的特征进行时间预测。

2. 验证**出发时间对旅行时间的影响**：选取了两条路程均为3KM的轨迹。蓝色旅行时间短484s， 出发时间是16:06分；黄色时间更长 1087s，出发时间是18:11分. 同时黄色线路更靠近市中心，交通拥堵的情况也会更多。

   ​                                                           <img width="279" alt="image" src="https://github.com/NaOH678/Estimate-Travel-Time/assets/112929756/864c1849-df47-4566-8243-062ac86c129b">


   

**考虑将出发时间作为一个变量输入模型**

#### 回归分析

##### 模型构建

**案例分析主要是以探索为主，过程存在一些主观性。并且没能很好的控制变量。所以考虑构建一个关于旅行时间的回归方程。**

现有样本矩阵$X \in \R^{n\times p}$，回归系数$\beta \in \R^{p}$，响应变量为旅行时间$Y$:
$$
Y = X\beta + \epsilon
$$

$$
\begin{align}
\begin{array}{c}
\beta = (\beta_1, \beta_2, \cdots, \beta_p)'
\end{array}\quad X = \left(\begin{array}{c c}

x_{11} &x_{12} &\cdots &x_{1p} \\
x_{21} &x_{22} &\cdots &x_{2p} \\
\vdots &\vdots &\vdots &\vdots \\
x_{np} &x_{n2} &\cdots &x_{np}

\end{array}
\right) \tag{}


\end{align}
$$



利用最小二乘回归求的回归系数的估计:
$$
\hat{\beta} = (X'X)^{-1}X'Y
$$



因为变量中存在分类变量，不能直接参与回归。要将分类变量转化为哑变量。

- 将start_time(6:00~24:00)划分为9段，每段两个小时，这里有9个分类变量, 每个变量取值为0,1。下同。
- weekID代表星期(0～6)，这里有7个分类变量；
- dateID代表每一天，数据集涉及到了19天，共19个分类变量;
- 行程距离，连续变量;
- 载客状态占比，连续变量;

为了避免完全共线性对模型的影响，分别将start_time，weekID，dateID中删除一个分类变量。所以最后的变量个数为9+7+18+2-3 = 33。



##### 结果分析

根据p值和置信区间对结果进行分析，如果p值小于0.05或置信区间不包含0，则表示该变量是显著的。

<img width="274" alt="image" src="https://github.com/NaOH678/Estimate-Travel-Time/assets/112929756/33c7c054-3cea-4580-9927-dd131e0ed285">       <img width="353" alt="image" src="https://github.com/NaOH678/Estimate-Travel-Time/assets/112929756/42dbf99e-238a-444a-a685-977eeac5d2d6">
 



可以发现，**载客状态比例**，**行程距离**，**起始时间**，**weekID**基本都通过了显著性检验吗，但是**dateID**有许多没有通过显著性检验。中期答辩时没有理解这是为什么。==但是在结题答辩的时候，助教指出了问题所在，因为数据时一个月的，所以没有横向的比较。所以并没有太大的作用。==



经过回归分析，可以初步确定提取出上述变量加入模型。**Attribute Layer**结构如下：

<img width="278" alt="image" src="https://github.com/NaOH678/Estimate-Travel-Time/assets/112929756/8d57d16a-c777-4602-b9ed-1fe753716e31">

将上述几个变量先进行embedding。然后进行拼接，输出结果。

**因为代码比较长，所以代码的相关含义直接写在了注释中，方便助教查阅～	**

```py
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
import numpy as np
from torch.autograd import Variable

class Net(nn.Module):
  	# 定义了一个包含嵌入层维度信息的列表。每个元组表示一个嵌入层，包括输入维度、输出维度，以及该				嵌入层的名称。
    embed_dims = [('driverID', 24000, 16), ('weekID', 7, 3), ('timeID', 1440, 8)]

    def __init__(self):
        super(Net, self).__init__()
        # whether to add the two ends of the path into Attribute Component
        self.build()
        
    def build(self):
        # build 方法通过遍历嵌入层信息列表，为每个嵌入层添加一个 PyTorch 的 Embedding 模						块。
        for name, dim_in, dim_out in Net.embed_dims:
            self.add_module(name + '_em', nn.Embedding(dim_in, dim_out))

    def out_size(self):
        sz = 0
        for name, dim_in, dim_out in Net.embed_dims:
            sz += dim_out
        # 增加一个维度，用于最后distance表征
        return sz + 1

    #forward 方法定义了模型的前向传播过程。它遍历每个嵌入层，将输入属性经过相应的 Embedding 		层，并将结果拼接到一个列表中。最后，通过 torch.cat 将所有嵌入层的结果拼接在一起，得到最终		的特征向量作为模型的输出。其中，总距离通过 utils.normalize 进行规范化后添加到特征向量中。
    def forward(self, attr):
        em_list = []
        for name, dim_in, dim_out in Net.embed_dims:
            embed = getattr(self, name + '_em')
            attr_t = attr[name].view(-1, 1)

            attr_t = torch.squeeze(embed(attr_t))

            em_list.append(attr_t)

        dist = utils.normalize(attr['dist'], 'dist')
        em_list.append(dist.view(-1, 1))

        return torch.cat(em_list, dim = 1)

```



### 挑战二

- 挑战：**时空关系的挖掘**。轨迹数据包含了丰富的时空关系，如何将他们提取出来进行时间预测。这也是传统方法没有考虑的因素。
- 对策：**构建一个Geo-Con模块**。构建Geo-Conv Layer提取时空关系；构建循环神经网络LSTM，提取时间依赖。

<img width="414" alt="image" src="https://github.com/NaOH678/Estimate-Travel-Time/assets/112929756/a839c236-68c6-4128-9b7c-607f88f84982">
                                    <img width="320" alt="image" src="https://github.com/NaOH678/Estimate-Travel-Time/assets/112929756/03c31d9e-c260-4e89-85ff-38550eefdb37">
        

根据左边的图可以看到，现将轨迹点映射到成为一个更高维度的向量，然后在这个更高维度的向量上进行卷积，最后添加的绿色的就是在上一个环节中`out_size(self)`有个`sz+1`的代码，目的就在于此。所以最后提取出的信息包括轨迹点之间的信息和距离信息作为右图中蓝色的模块（作为空间关系）与红色的（即上一环节中的输出）进行拼接。然后在输入到LSTM中去捕获时间依赖。



该模块主要代码如下	

```py
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils

class Net(nn.Module):
    # 初始化函数接受两个参数 kernel_size 和 num_filter，并保存它们为模型的属性。然后调用 				build 方法构建模型。
    def __init__(self, kernel_size, num_filter):
        super(Net, self).__init__()

        self.kernel_size = kernel_size
        self.num_filter = num_filter

        self.build()

    '''
    		build 方法创建了模型的各个层：
    		state_em: Embedding 层，用于处理轨迹的状态信息。
				process_coords: 全连接层，将轨迹的经纬度坐标和状态信息映射到一个16维的向量。
				conv: 1维卷积层，用于对映射后的向量进行卷积操作。
    ''' 
    def build(self):
        self.state_em = nn.Embedding(2, 2)
        self.process_coords = nn.Linear(4, 16)
        # 删除地理卷积层做的修改
        # self.process_coords = nn.Linear(4, 32)
        self.conv = nn.Conv1d(16, self.num_filter, self.kernel_size)

    def forward(self, traj, config):
      """
      forward 方法定义了模型的前向传播过程。它接受轨迹数据 traj 和配置信息 config 作为输入,			并经过一系列操作得到输出 conv_locs。具体步骤包括：
      
			将经度 (lngs) 和纬度 (lats) 扩展为第三个维度。
			通过 Embedding 层处理轨迹的状态信息 (states)。
			将经纬度和状态信息拼接成一个输入张量 locs。
			将 locs 映射到16维向量并进行维度转置。
			对映射后的向量进行卷积操作，使用 ELU 作为激活函数。
			计算本地路径的距离信息并拼接到卷积结果中。
      """
        lngs = torch.unsqueeze(traj['lngs'], dim=2)
        lats = torch.unsqueeze(traj['lats'], dim=2)
        states = self.state_em(traj['states'].long())
        locs = torch.cat((lngs, lats, states), dim=2)

        # 将坐标映射成为16维向量
        locs = F.tanh(self.process_coords(locs))
        locs = locs.permute(0, 2, 1)
        
        # 删除地理卷积层（把没注释的注释，将注释的解除注释）
        # conv_locs = locs.permute(0, 2, 1)
        conv_locs = F.elu(self.conv(locs)).permute(0, 2, 1)

        # calculate the dist for local paths
        local_dist = utils.get_local_seq(traj['dist_gap'], self.kernel_size, 						config['dist_gap_mean'],  config['dist_gap_std'])
        local_dist = torch.unsqueeze(local_dist, dim=2)
        conv_locs = torch.cat((conv_locs, local_dist), dim=2)

        return conv_locs

```

### 挑战三

- 挑战：**整体预测与分段预测**。常用的方法是**整个路径预测时间**或者是将**路径分成子路径预测时间最后求和**。但是两者各有千秋。整体预测会忽略道路之间的连接、红绿灯、转向问题；但是分段预测能较好的估计每一段的时间，但是会忽略分段之间的联系。
- 对策：**加权损失函数**。构建一个损失函数，计算损失的方法是是对**分段预测时间的损失**与**整体预测时间的损失**的加权求和。

<img width="754" alt="image" src="https://github.com/NaOH678/Estimate-Travel-Time/assets/112929756/f64a836e-b0ea-46a8-ac6a-f12e9ad0056c">

蓝色和红色时前两个模块得到的特征，紫色模块时输出的时间。`R-k`表示第$k$段输出的时间，分别与对应的每一段真实时间做差求得损失，这部分的损失称为**分段损失**；`r-end`表示的是整条轨迹的损失，称为总体损失。定义模型的损失为
$$
Loss = \alpha\times subLoss + (1-\alpha)\times entireLoss \label{4}
$$
然后进行梯度回传，学习参数。

相关代码

```py
class EntireEstimator(nn.Module):
  # 整体预测
    def __init__(self, input_size, num_final_fcs, hidden_size=128):
        super(EntireEstimator, self).__init__()

        self.input2hid = nn.Linear(input_size, hidden_size)

        self.residuals = nn.ModuleList()
        for i in range(num_final_fcs):
            self.residuals.append(nn.Linear(hidden_size, hidden_size))

        self.hid2out = nn.Linear(hidden_size, 1)

    def forward(self, attr_t, sptm_t):
        inputs = torch.cat((attr_t, sptm_t), dim=1)

        hidden = F.leaky_relu(self.input2hid(inputs))

        for i in range(len(self.residuals)):
            residual = F.leaky_relu(self.residuals[i](hidden))
            hidden = hidden + residual

        out = self.hid2out(hidden)

        return out
```



```py
class LocalEstimator(nn.Module):
  # 分段预测
    def __init__(self, input_size):
        super(LocalEstimator, self).__init__()

        self.input2hid = nn.Linear(input_size, 64)
        self.hid2hid = nn.Linear(64, 32)
        self.hid2out = nn.Linear(32, 1)

    def forward(self, sptm_s):
        hidden = F.leaky_relu(self.input2hid(sptm_s))

        hidden = F.leaky_relu(self.hid2hid(hidden))

        out = self.hid2out(hidden)

        return out
```

```py


    def eval_on_batch(self, attr, traj, config):
    		# 两者结合
        local_out = None
        local_length = None
        if self.training:
            entire_out, (local_out, local_length) = self(attr, traj, config)
        else:
            entire_out = self(attr, traj, config)

        # 整条轨迹进行预测，计算损失
        pred_dict, entire_loss = eval_on_batch_entire(entire_out, attr['time'], config['time_mean'], config['time_std'])

        if self.training:
            # get the mean/std of each local path
            mean, std = (self.kernel_size - 1) * config['time_gap_mean'], (self.kernel_size - 1) * config[
                'time_gap_std']

            # get ground truth of each local path
            local_label = utils.get_local_seq(traj['time_gap'], self.kernel_size, mean, std)

            # 分段轨迹进行预测，计算损失
            local_loss = eval_on_batch_local(local_out, local_length, local_label, mean, std)

            return pred_dict, (1 - self.alpha) * entire_loss + self.alpha * local_loss
        else:
            return pred_dict, entire_loss
```





## 实验

### 评估指标

**平均绝对百分比误差（MAPE）**
$$
MAPE = \frac{1}{n}\sum_{i=1}^n\frac{|t_i - \hat{t_i}|}{|t_i|}
$$


### 对比实验

与传统方法对比：

- AVG：用所以用的数据集的训练集计算了每个时间段的平均速度，然后在验证集和测试集上预测时间。

  MAPE：0.36

- GBDT：采用XGBoost方法，类似于实验三，预测时间。

  MAPE：0.27

### 实验结果

- 训练集MAPE：0.092；
- 验证集MAPE：0.1991；
- 测试集MAPE：0.2163.

### 消融分析

将**地理卷积层(Geo-Conv)**删除

- 训练集MAPE：0.1069;
- 验证集MAPE：0.2204;
- 测试集MAPE：0.2467.

可以看到，删除了地理卷积层后预测准确率明显下降了。说明了地理卷积层发挥了不错的作用。



## 项目总结

### 项目创新点

- 引入了地理卷积层，相比于传统方法，更好的捕捉了空间关系，提升了模型的预测准确率；

- 采用多任务学习，将整个路段的损失和每一条子路段的损失相结合，提高了预测的精准度。

  再次回到公式$\refeq{4}$

  <img width="292" alt="image" src="https://github.com/NaOH678/Estimate-Travel-Time/assets/112929756/e18a5031-8b6e-4f1a-8dc0-16f2da674620">


不断调整$\alpha$的取值来探究加权损失函数的作用。

- 当$\alpha=0$时，模型的损失仅仅为完全损失，可见测试集的MAPE偏高；
- 当$\alpha=1$时，模型的损失仅仅为局部损失求和，测试集的MAPE也偏高。

说明了加权损失函数在一定程度上缓解了两种方法的不足之处。后续可以调整$\alpha $的大小寻找到最佳参数。



### 不足之处

- 模型改进

  近几年的论文中提到了利用图神经网络捕捉道路的空间依赖和transformer架构进行预测。可以考虑构建这样的模型。

- 数据集扩充

  本次实验由于数据可得性和算力的问题使用的是一个小数据集，原数据集有非常大。

- 特征补充

  有的论文中提到了天气、速度的因素，但是本次实验的数据集中无法提取相关特征。

- 缺少对司机行为模式的挖掘

  本实验中只将”driver ID”作为特征加入了模型。但是无法得到更具体的驾驶行为模式，比如这个司机是比较保守的，还是激进的等等。

## 收获

### 模型解释与Idea提出

在结题答辩的最后，助教问了我一个问题：“CNN和LSTM”在这里有什么区别？好像都提取了时空依赖。”当时我并没有回答出这个问题。**另外一个学长解释道：“把轨迹点映射到高维空间之后，在上面进行卷积，就像是在一张图片上进行卷积，CNN学习到相邻像素点之间的联系。回到这个项目中，这时候CNN会学习到相邻轨迹点之间的关系，但是并没有时间顺序。所以这时候要把CNN提取出来的信息输入到LSTM中获取时间上的依赖。”**我觉得这位学长说的十分有道理，有一种醍醐灌顶的感觉。这也说明了在平常学习中忽视了模型的解释性。这是我静候学习需要关注的。

另外助教还提到，不是因为这个模型在最后实验中取得了好的效果而使用这种方法，而是要去分析现有的方法存在什么问题没有解决，所以我们提出一种方法尝试解决这种问题，最后进行实验来验证我们提出的方法是有效的。我觉得这是今后在科研和学习中非常重要的思想。

按照上面的思路，我使用的模型并没有考虑到轨迹之间的相互影响。为了去学习轨迹之间影响，人们提出了使用图神经网络进行学习。这也正是因为轨迹之间的交错纵横形成了一张图，所以才会有2020年开始使用图神经网络捕获时空关系的论文。

### 深度学习项目基本框架

之前的AI相关课程设计理念的代码比较简单，基本在一个ipynb文件种就可以解决。但本次课程设计让我体验了一次完整的深度学习项目的代码框架。

1. 在真正的项目中，代码都是各个py文件各司其职的，有的文件被嵌套使用。阅读源码的过程还是比较痛苦的。刚开始来呢数据是如何经过处理，如何输入模型都搞不清楚，花了很长的时间才摸清楚了门道。
2. 以及不同模型之间是如何连接的，也花费了不少时间阅读。
3. 在一个完整的深度学习项目中，还有日志功能，保存了每一段时间训练的参数，防止模型在训练过程中崩溃而导致参数毫无保留。
4. 最后在参考了论文作者开源的代码仓库的基础上完成了本次项目的代码。作者仓库感觉是被污染了，每一行都有bug（捂脸），最后在他的框架了进行了许多修改，完成了本次课程设计。



总体而言，本次课程设计收获颇丰。不仅了解了旅行时间预测的相关知识，也学习到了一些科研的基本思路。同时体验了深度学习完整的代码框架。

最后感谢老师、助教和评委学长提供的宝贵建议。特别感谢杨苗苗学姐的帮助！



代码存放在了我的仓库中

