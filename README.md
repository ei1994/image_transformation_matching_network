## image_transformation_matching_network   
**专利：基于图像转换的异源图像块匹配方法（申请号：201810276986.3）**     
## 概述：
异源图像（可见光和SAR、可见光和近红外）包含大量互补信息，这对获取图像完整信息十分有利。但异源图像之间的特征差异很大，特别是SAR图像中存在的相干斑噪声。传统的基于手工提取特征的方法无法得到他们之间具有一致性和相关性的特征信息，所以很容易导致错误的匹配结果。近年来研究较热的生成对抗网络，可以将**异源图像转换为同源图像，再进行匹配，降低匹配难度（SAR图像的相干斑噪声多，相当于去噪网络）** 。另外，使用网络对转换后的同源图像进行匹配检测，相比于传统的基于距离的匹配预测方法，**能提取图像中更高层的特征语义信息，也不需要考虑距离度量问题** ，有利于提高匹配的精度和鲁棒性。   
## 创新点：	
（1）异源转同源思想，降低图像匹配难度，相当于用**GAN去除SAR的噪声** ，得到具有一致性的图像特征用于匹配预测；   
（2）**双分支参数不共享MatchNet网络** ，特征提取网络最后得到的特征图，直接在深度通道concate后输入匹配网络，再加一层卷积处理增加非线性（相比reshape为一个向量后再预测的方法，保留更多的空间特征信息），之后连接全连接层，得到图像块对预测的匹配概率。  



