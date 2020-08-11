leetcode64-最小路径和-找规律

题目地址：https://leetcode-cn.com/problems/minimum-path-sum/

题目大意：

给定一个包含非负整数的 m x n 网格，请找出一条从左上角到右下角的路径，使得路径上的数字总和为最小。

说明：每次只能向下或者向右移动一步。

例子：
输入:
[
  [1,3,1],
  [1,5,1],
  [4,2,1]
]
输出: 7
解释: 因为路径 1→3→1→1→1 的总和最小。

参考链接：https://blog.csdn.net/weixin_38278334/article/details/89930610

题目解析：这道题目和leetcdoe62本质上是相似的，本质上也是一个找规律的问题。

a b

c d

到达d这个位置的路径和应该是(b,c)之间最小值加上d原本的值，然后不停的向右下角那个位置递归（不知道说递归是不是准确）。

还有一个细节点需要注意，62那道题目边上的点都是1，这里应该是不停的相加过去。

直接看代码

```python
##代码已经通过leetcode代码进行测试
class Solution:
    def minPathSum(self, grid: List[List[int]]) -> int:
        
        m=len(grid)
        n=len(grid[0])
        for i in range(1,m): 
            grid[i][0]= grid [i][0] + grid[i-1][0]
         
        for j in range(1,n):
            grid[0][j] = grid [0][j-1] + grid[0][j]
             
        for i in range(1,m):
            for j in range(1,n):     
                grid[i][j]=grid[i][j]+min(grid[i-1][j],grid[i][j-1])
        return grid[m-1][n-1]  
```