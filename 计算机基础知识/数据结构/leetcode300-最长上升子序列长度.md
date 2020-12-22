最长上升子序列长度

题目地址：https://leetcode-cn.com/problems/longest-increasing-subsequence/

题目解析地址：

1. https://leetcode-cn.com/problems/longest-increasing-subsequence/solution/shi-pin-tu-jie-zui-chang-shang-sheng-zi-xu-lie-by-/
这个人解释的非常的清楚

2. https://leetcode-cn.com/problems/longest-increasing-subsequence/solution/zui-chang-shang-sheng-zi-xu-lie-by-leetcode-soluti/
这个里面有Python代码

我的注解：

核心要点就是，立足当下这个点，找出截止到当前这个点的最大上升子序列长度。

当前最大子序列长度如何找到：就是遍历之前点，看他们到我当前这个点是不是上升的，如果是就长度就1，如果不是，不操作。


代码如下：

```python  
## 代码已经测试通过
class Solution:
    def lengthOfLIS(self, nums):
        if not nums:
            return 0
        dp = []
        for i in range(len(nums)):
            dp.append(1)# 在i这个位置，最小是1
            for j in range(i):
                if nums[i] > nums[j]:
                    dp[i] = max(dp[i], dp[j] + 1)
        return max(dp)
```
