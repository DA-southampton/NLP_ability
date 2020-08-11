leetcode62-不同路径-找规律

题目地址：https://leetcode-cn.com/problems/unique-paths/

题目解析参考地址：https://zhuanlan.zhihu.com/p/43358393
题目大意：
一个机器人位于一个 m x n 网格的左上角 （起始点在下图中标记为“Start” ）。

机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为“Finish”）。

问总共有多少条不同的路径？

题目解析地址

题目解析：

这个道题的本质上是一个规律问题：到达边上的点的路径总是1，中间格到达的路径是左边和上面存储路径的和。

一个简单的推导公式为：A[x][x]=A[x-1]+A[y-1]

```python
## 代码已经通过letcdoe进行测试
class Solution:
    def uniquePaths(self, m, n):
        """
        :type m: int
        :type n: int
        :rtype: int
        """
        # 初始化一个值全为1的矩阵，这样可以不用再去给边界赋值
        path_martrix = [[1 for i in range(m)] for j in range(n)]
        for line in range(1, n):  # 从第二行第二列开始遍历矩阵
            for col in range(1, m):
                path_martrix[line][col] = path_martrix[line - 1][col] + path_martrix[line][col - 1]  # 推导式
        return path_martrix[n - 1][m - 1]  # 返回矩阵最右下角的值
```


