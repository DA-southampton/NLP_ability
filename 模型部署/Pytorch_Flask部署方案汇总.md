TF自身带有部署方案，Pytorch就比较难，搜索了一些关于关于Pytorch部署的文章，以及一些细节讲解。

1. Flask+nginx + uwsgi

其中一种部署方案就是使用Flask提供部署接口，向外提供http接口。

不过这种存在一种问题，就是高并发的问题。如果并发量不大，使用flask自带的server就没有问题，但是如果高并发，用在生产环境，这样的性能是跟不上。

这种时候，Flask+nginx + uwsgi 就是一个比较好的方案。

简单来说，不一定准确，nginx做负载均衡，uwsgi处理多进程，Flask处理数据。

在这个过程中，有一些细节需要注意，都是属于计算机的基础知识，我把我查过的一些资料放在下面：

为什么nginx可以直接部署，还要uWSGI，gunicorn等中间件？ - lip lee的回答 - 知乎
https://www.zhihu.com/question/342967945/answer/804493384
这个答案比较清晰的讲出了为什么需要uWSGI 和nginx，以及部分的作用


完全理解同步/异步与阻塞/非阻塞 - Maples7的文章 - 知乎
https://zhuanlan.zhihu.com/p/22707398
这个文章的例子比较清晰的清楚了同步异步阻塞非阻塞的区别，很形象。一般来说，Flask默认是单进程单线程阻塞的方式。

机器学习web服务化实战：一次吐血的服务化之路
https://cloud.tencent.com/developer/article/1563415
使用Flask进行多进程的时候，因为线程正常之间不会出现数据的通信，会出现每个进程都会load一次模型，如果模型太大，会出现占用太大内存的问题。这个文章给出一个方法可以做到多个进程初始化一次模型。

Flask使用自带的多线程的时候，会对线程进行隔离：参考这里：
Flask中的线程隔离：
https://blog.csdn.net/weixin_43870742/article/details/95604665?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-1.add_param_isCf&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-1.add_param_isCf


有些时候上述的框架也可以代替为：

nginx + uwsgi + flask
nginx + gunicorn + flask
nginx + gunicorn + gevent + flask

这里面需要注意的就是 gunicorn 和 uwsgi起到的是同一个作用，只不过他的方式可以使用gthread 或者 gevent，gunicorn的work类型不同，在处理读写密集型和cpu密集型时效率是不一样的

具体可以参考这里：https://zhuanlan.zhihu.com/p/81801240

关于IO密集型、计算密集型，多线程、多进程：

IO密集型是指频繁的输入输出数据，比如请求网页，读写文件这些简单任务。

CPU密集型任务指的是CPU计算占据主要任务，大量CPU时间在不停的在进行矩阵运算或者视频编码这些需要计算的东西。

简单来讲，CPU密集任务我们一般使用多进程，进程数设定为CPU的核数比较好。IO密集型任务一般使用多线程。

“CPU密集型也叫计算密集型，指的是系统的硬盘、内存性能相对CPU要好很多，此时，系统运作大部分的状况是CPU Loading 100%，CPU要读/写I/O(硬盘/内存)，I/O在很短的时间就可以完成，而CPU还有许多运算要处理，CPU Loading很高。”

关于这个可以参考这里：
https://www.cnblogs.com/aspirant/p/11441353.html


简单的代码验证可以参考这里：

https://zhuanlan.zhihu.com/p/24283040

平常的时候，部署完毕，我们需要对接口进行压测，这个部分直接百度就可以，有很多。

（先提一点，关于黑马头条B站这里有一个关于FLask的讲解系列视频，可以看一下）


2. 关于Pytorch其他部署方案汇总

https://github.com/DA-southampton/Deep-Learning-in-Production

直接去这个网站看，有很多