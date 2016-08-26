
[TOC]


# 1.network3.py模块

## 1.1 network3.py模块结构

- 该模块包含一个网络类Nework和三个网络layer类（分别为ConvPoolLayer类/FulllyConnectedLayer类/SoftmaxLayer类）
- 该模块包含5个函数
	- linear
	- RelU
	- size
	- laod_data_shared
	- dropout_layer
- 同时，该模块还包含一个变量：GPU，取值为False或者True




# 2. 代码使用举例
## 2.1 创建一个Network类的对象
下面语句创建一个Nerwork类的对象net，或者说是：net是一个指向Nerwork类的对象

```
net = Network([
        FullyConnectedLayer(n_in=784, n_out=100),
        SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)
```

## 2.2 Network类需要的输入
该类需要两个参数：
- 第一个参数是一个list数据，它描述了网络的结构，该list的每个参数分别某一层的类的对象，例如，上面代码中
	- list[0]为FullyConnectedLayer类的对象
        - list[0]为SoftmaxLayer类的对象
- 第二个参数为一个常数，用来指定mini_batch_size的大学

## 2.3 Network类的__init__函数（构造函数）
- self：指向该类的一个实例化对象，对本例而言，self其实就是指向net对象
- 为self的相关参数赋值
	- self.layers：存放的是网络的结构，为一个list
	- mini_batch_size：mini_batch的尺寸
	- params：网络各层的参数，是一个list，其中，每个元素还是一个list（存放该层的weights和bias）
	- x：theano的矩阵占位符
	- y：theano的向量占位符
	- output：网络初始输出（无dropout）
	- output_dropout：网络初始输出（有dropout）


## 2.4 FullyConnectedLayer类和SoftmaxLayer类

这两个类都定义了一个单层网络，它们都具有三个方法
- __init__：网络的构造函数 
- set_input：设置网络输入，并计算网路的输出
- accuracy：计算网络的精度

### FullyConnectedLayer的set_inpt方法
分别计算网络在无dropout和有dropout下的输出
- self.output：无dropout下的该layer的activations
- self.y_out：无dropout下的该layer的输出labels
- self.inpt_dropout：有dropout下的该layer的输入（即对不需要的输入neurons直接赋值为0）
- self.output_dropout：有dropout下的该layer的activations
