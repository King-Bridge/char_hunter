基本上需要修改的部分都在train.py里。其余几个文件可以看看，了解一下sampler、loss、metric是什么原理就行。


目前跑一次大概要一个小时左右。


1，然后主要调一下train里面loss的参数，49-53行，alpha和gamma，我代码里写了说明。
2，也可以试一下对于二分类器的sample比例（正负样本目前是1：1，可以再改一改，比如正例又放回地采样）
3，对于20个二分类器，由于输出的是概率，所以threshold要改一下，位于metric.py文件的第55行。


依赖包信息
torch                     2.2.2+cu121
torchaudio                2.2.2+cu121
torchvision               0.17.2+cu121


