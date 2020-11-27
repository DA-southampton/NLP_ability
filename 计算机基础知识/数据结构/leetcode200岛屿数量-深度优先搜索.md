岛屿数量

题目地址：https://leetcode-cn.com/problems/number-of-islands/

解析地址：https://leetcode-cn.com/problems/number-of-islands/solution/dao-yu-shu-liang-by-leetcode/
这个是官方解析地址，我在第一遍看视频讲解的时候，没太懂，官方这个人员讲的真实一言难尽。

随后我在B站视频找到了这个讲解：
https://www.bilibili.com/video/BV1Mt4y127V4?from=search&seid=6257308462943402854

程序员吴师兄的这个图解还是很清楚的，这个时候我再回过去看官方的讲解，里面很多东西就都清楚了。


这道题核心要点在：递归+深度优先搜索，大白话总结就是对每个非0（也就是为1的岛屿点进行深度优先搜索）

我自己理解这个深度优先是这样的，假如当前这个点是岛屿点，然后向四周蔓延，找到下一个领接的岛屿点，然后把向外蔓延的工作交给这个点，以此类推。

代码实现
```python
##代码已经通过测试
class Solution:
    def dfs(self, grid, r, c):
        grid[r][c] = 0
        nr, nc = len(grid), len(grid[0])
        for x, y in [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]:
            if 0 <= x < nr and 0 <= y < nc and grid[x][y] == "1":
                self.dfs(grid, x, y)

    def numIslands(self, grid):
        nr = len(grid)
        if nr == 0:
            return 0
        nc = len(grid[0])

        num_islands = 0
        for r in range(nr):
            for c in range(nc):
                if grid[r][c] == "1":
                    num_islands += 1
                    self.dfs(grid, r, c)
        
        return num_islands
"""
作者：LeetCode
链接：https://leetcode-cn.com/problems/number-of-islands/solution/dao-yu-shu-liang-by-leetcode/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
"""

"""
复杂度分析：
时间复杂度：O(MN)O(MN)，其中 MM 和 NN 分别为行数和列数。

空间复杂度：O(MN)O(MN)，在最坏情况下，整个网格均为陆地，深度优先搜索的深度达到 M NMN

"""
```