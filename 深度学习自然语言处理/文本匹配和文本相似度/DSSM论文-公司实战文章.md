DSSM

https://posenhuang.github.io/papers/cikm2013_DSSM_fullversion.pdf

# 架构图

架构图很简单，也有点老了

![image-20201223164854016](https://picsfordablog.oss-cn-beijing.aliyuncs.com/2021-01-14-081908.jpg)

核心细节点有两个：一个是使用了cosine做了查询和文档的相似度量

![image-20201223165620044](https://picsfordablog.oss-cn-beijing.aliyuncs.com/2021-01-14-81910.jpg)

第二个就是，softmax

![image-20201223165655726](https://picsfordablog.oss-cn-beijing.aliyuncs.com/2021-01-14-081909.jpg)

第三个是损失函数，使用最大似然估计，只计算了正样本：

![image-20201223165905179](https://picsfordablog.oss-cn-beijing.aliyuncs.com/2021-01-14-081906.jpg)



对于DSSM，主要是想提几个小细节，也是我自己的思考，不准确的地方，欢迎拍砖。

首先，为什么采用(Query,D+,D-1,D-2,D-3)的方式作为输入，而不是采用(Query,D+)；(Query,D-1)；(Query,D-2)；(Query,D-3)；作为单独的pair样本对作为输入；

这个问题，其实还可以换个问法，为什么DSSM的损失函数，使用的是一个正样本多个负样本归一化之后对正样本求交叉熵，而不是单个pair对作为输入，去求二分类的交叉熵；

我的理解是，这个其实适合业务场景相关的一个问题；参考下面这个回答的答案：

DSSM 为什么以一个正样本几个负样本softmax归一化然后正样本交叉熵的方式算loss? - xSeeker的回答 - 知乎 https://www.zhihu.com/question/425436660/answer/1522163398

我直接截图过来：

![image-20210114153620063](https://picsfordablog.oss-cn-beijing.aliyuncs.com/2021-01-14-081907.jpg)



本质上，还是在学习一种顺序关系，正样本排在负样本之前



# DSSM在各大公司的实战



实践DSSM召回模型 - 王多鱼的文章 - 知乎 https://zhuanlan.zhihu.com/p/136253355

深度语义模型以及在淘宝搜索中的应用:https://developer.aliyun.com/article/422338  写的很好

百度NLP | 神经网络语义匹配技术：https://www.jiqizhixin.com/articles/2017-06-15-5  

语义匹配 - 乐沐阳的文章 - 知乎 https://zhuanlan.zhihu.com/p/57550660



# 损失函数

DSSM通过推导公式，可以得到最大化似然估计和交叉熵损失函数是一致的。

【辩难】DSSM 损失函数是 Pointwise Loss 吗？ - xSeeker的文章 - 知乎 https://zhuanlan.zhihu.com/p/322065156

交叉熵损失函数原理详解：

https://blog.csdn.net/b1055077005/article/details/100152102

