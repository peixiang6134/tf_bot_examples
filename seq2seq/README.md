

### Seq2Seq简介

Seq2Seq可以完成两个不定长序列的映射，它由`Encoder`与`Decoder`两部分组成，`Encoder`负责把序列编码成一个固定长度的向量，这个向量作为输入传给`Deocder`，`Decoder`将根据自身的状态以及`Encoder`的输出生成目标序列。下图就是一个简单的Seq2Seq模型示意图：

![1581593306961](imgs/1581593306961.png)



Seq2Seq模型在机器翻译、文本摘要等许多文本生成领域都有应用。如果想更深入地了解Seq2Seq，可参考以下论文：

 [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/pdf/1409.3215.pdf)

 [Learning Phrase Representation using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/pdf/1406.1078.pdf)



下面将展示如何使用 [神经网络配置系统](https://openai.weixin.qq.com/chat )配置一个Seq2Seq模型。



### 配置模型

Seq2Seq模型的配置主要分为三个部分：

* 设置数据输入的参数
* 配置Encoder
* 配置Decoder



#### 设置数据输入的参数



进入系统[首页](https://openai.weixin.qq.com/chat)，点击配置神经网络：

![1581590959415](imgs/1581590959415.png)



提供数据的特征文件和标注文件：

![1581590975393](imgs/1581590975393.png)







![1581590990402](imgs/1581590990402.png)



配置数据输入的参数：

![1581591002111](imgs/1581591002111.png)



配置将标注数值化的方式：

![1581591023256](imgs/1581591023256.png)



引入`Word Embedding`作为`Encoder`的`Embedding`层：

![1581591065074](imgs/1581591065074.png)



设置`Word Embedding`的输入，提交之后选择使用`sequence to sequence`架构：

![1581591081299](imgs/1581591081299.png)



#### 设置Encoder

选择配置encoder:

![1581591109376](imgs/1581591109376.png)



在embedding层的基础上增加`RNN`:

![1581591144920](imgs/1581591144920.png)



![1581591192168](imgs/1581591192168.png)

Encoder的RNN只需要输出状态即可：

![1581591182436](imgs/1581591182436.png)

![1581591219786](imgs/1581591219786.png)

点击`设置encoder输出`，将`encoder`的状态设置为输出，因为我们需要将它传给`decoder`：

![1581591295860](imgs/1581591295860.png)

接下来就可以配置`decoder`了。



#### 设置decoder

选择配置decoder:

![1581591442721](imgs/1581591442721.png)



利用`增加网络联接`复用`encoder`部分的`embedding`，输入选标注`I0_shifted`：

![1581591497196](imgs/1581591497196.png)

然后增加一层`RNN`，注意需要将`RNN`设置中的输入`initial state`设置为是（需要使用`encoder`的输出作为`initial state`），并且将`RNN`设置为单向（`decoder`是从左至右生成单词的）：

![1581591602433](imgs/1581591602433.png)

![1581591623027](imgs/1581591623027.png)

`decoder rnn`的输入是`decoder embedding`层的输出和`encoder`的状态：

![1581591767493](imgs/1581591767493.png)



最后，在`decoder`输出的基础上增加一个全连接层，将`decoder`的隐状态映射到词典空间才能生成单词：

![1581591891476](imgs/1581591891476.png)

![1581591902883](imgs/1581591902883.png)



![1581591918226](imgs/1581591918226.png)

这样，模型的基本架构就配置完毕了。接下来设置一下模型的输出，全连接层输出的词典上的分布将作为输出：

![1581591969269](imgs/1581591969269.png)

然后是配置模型训练的参数，利用`decoder`的输出和标注计算损失：

![1581592026050](imgs/1581592026050.png)

设置了损失后就可以配置模型训练的优化器了：

![1581592094321](imgs/1581592094321.png)

使用`Adam`作为优化器：

![1581592114067](imgs/1581592114067.png)

![1581592126014](imgs/1581592126014.png)

至此，一个简单但完整的`Seq2Seq`模型就配置完毕了。你可以下载代码，利用本项目[data目录](data/)下提供的数据，测试一下是否有效！
