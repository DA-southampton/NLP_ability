Leetcode31-下一个排列-数组翻转

题目链接：https://leetcode-cn.com/problems/next-permutation/

题目解析：https://blog.csdn.net/fuxuemingzhu/article/details/82113409


题目大意：
实现获取下一个排列的函数，算法需要将给定数字序列重新排列成字典序中下一个更大的排列。必须原地修改，只允许使用额外常数空间。


对于这个题目，我觉得需要从实际例子出发。

比如，数组为：7，4，3，2，1 下一个排列没有比它更大的了，所以必然是 1,2,3,4,7

比如 1,2,7,4,3,1， 对于这个数组，他的下一个排列是什么？？这一点是最重要的。

首先，我们发现 基于前面数字为2的情况下，7,4,3,1已经是降序排列，所以比它大的基本没有。所以我们就需要调整2这个数字，也就是调整基础。调整为多少呢？

调整为后面数组中比2大的这个数字3，同时还需要对后面的数组进行升序排列，因为我们还需要进行下一个排列。

不太清楚有没有讲清楚，直接看例子。

首先，调整为 1,3,7,4,2,1

随后调整为，1,3,1,2,4,7

简单来说，找到完全降序的部分，然后调整它的前面一个数字，然后再reverse

直接看代码
```python
## 代码已经通过了leetcode测试
class Solution(object):
    def nextPermutation(self, nums):
        """
        :type nums: List[int]
        :rtype: void Do not return anything, modify nums in-place instead.
        """
        n = len(nums)
        i = n - 1
        while i > 0 and nums[i] <= nums[i - 1]:## 如果到了第一个值，就是第一个值和最后一个值比较，如果整个数组都是降序的，然后翻转整个数组就可以了
            i -= 1
        self.reverse(nums, i, n - 1)
        if i > 0: ## 等于0的时候，我们就直接翻转数组就可以，执行上一个步骤就够了
            for j in range(i, n):
                if nums[j] > nums[i-1]:
                    self.swap(nums, i-1, j)
                    break
        
    def reverse(self, nums, i, j):
        """
        contains i and j.
        """
        for k in range(i, (i + j) / 2 + 1):
            self.swap(nums, k, i + j - k)

        
    def swap(self, nums, i, j):
        """
        contains i and j.
        """
        nums[i], nums[j] = nums[j], nums[i]

```
