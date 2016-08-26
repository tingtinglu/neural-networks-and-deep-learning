
[TOC]


# network3.pyģ��

## network3.pyģ��ṹ

- ��ģ�����һ��������Nework����������layer�ࣨ�ֱ�ΪConvPoolLayer��/FulllyConnectedLayer��/SoftmaxLayer�ࣩ
- ��ģ�����5������
	- linear
	- RelU
	- size
	- laod_data_shared
	- dropout_layer
- ͬʱ����ģ�黹����һ��������GPU��ȡֵΪFalse����True




# ����ʹ�þ���
## ����һ��Network��Ķ���
������䴴��һ��Nerwork��Ķ���net������˵�ǣ�net��һ��ָ��Nerwork��Ķ���

```
net = Network([
        FullyConnectedLayer(n_in=784, n_out=100),
        SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)
```

## Network�������Ҫ������
������Ҫ����������
- ��һ��������һ��list���ݣ�������������Ľṹ����list��ÿ�������ֱ�ĳһ�����Ķ������磬���������
	- list[0]ΪFullyConnectedLayer��Ķ���
        - list[0]ΪSoftmaxLayer��Ķ���
- �ڶ�������Ϊһ������������ָ��mini_batch_size�Ĵ�ѧ

## Network���__init__���������캯����
- self��ָ������һ��ʵ�������󣬶Ա������ԣ�self��ʵ����ָ��net����
- Ϊself����ز�����ֵ
	- self.layers����ŵ�������Ľṹ��Ϊһ��list
	- mini_batch_size��mini_batch�ĳߴ�
	- params���������Ĳ�������һ��list�����У�ÿ��Ԫ�ػ���һ��list����Ÿò��weights��bias��
	- x��theano�ľ���ռλ��
	- y��theano������ռλ��
	- output�������ʼ�������dropout��
	- output_dropout�������ʼ�������dropout��


## FullyConnectedLayer���SoftmaxLayer��

�������඼������һ���������磬���Ƕ�������������
- __init__������Ĺ��캯�� 
- set_input�������������룬��������·�����
- accuracy����������ľ���

### FullyConnectedLayer��set_inpt����
�ֱ������������dropout����dropout�µ����
- self.output����dropout�µĸ�layer��activations
- self.y_out����dropout�µĸ�layer�����labels
- self.inpt_dropout����dropout�µĸ�layer�����루���Բ���Ҫ������neuronsֱ�Ӹ�ֵΪ0��
- self.output_dropout����dropout�µĸ�layer��activations