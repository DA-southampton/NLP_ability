## 46全排列
设计到一个回溯方法
这里有一个回溯方法总结
【LeetCode】回溯法总结 - 鱼枕的文章 - 知乎
https://zhuanlan.zhihu.com/p/63252392

## 48 旋转图像
题目位置：https://leetcode-cn.com/problems/rotate-image/

题目解析：https://blog.csdn.net/fuxuemingzhu/article/details/79451733

矩阵旋转90度，我觉得需要注意的是不可能按照九十度来做，而是看90度对应的是什么规律
发现，90度对应的是先上下翻转，再按照左上到右下的对角线进行翻转

## 55 跳跃游戏
题目位置：https://leetcode-cn.com/problems/jump-game/
题目解析： https://blog.csdn.net/fuxuemingzhu/article/details/83504437
方法使用的是贪心
这道题的核心在于实时维持一个可以达到的最大位置，如果此时的索引大于这个值，说明根本到不了这里



## 215 数组中的最大第K个元素
题目：https://leetcode-cn.com/problems/kth-largest-element-in-an-array/
题目解析：https://blog.csdn.net/fuxuemingzhu/article/details/79264797

循环，每次去除一个最大值

以后看看有咩有更好的方法



leetcode中动态规划总结 知乎搜搜
