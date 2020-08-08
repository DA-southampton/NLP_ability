Leetcode-33-搜索旋转排序数组-二分查找

题目地址：
https://leetcode-cn.com/problems/search-in-rotated-sorted-array/

题目解析：
https://blog.csdn.net/fuxuemingzhu/article/details/79534213

我的理解是这样的，是一个二分查找的问题 首先判断哪边是有序的，然后判断我这个值应该出现在有序的还是无序的那边，从而移动左右指针，继续进一步二分查找

```python
## 代码已经在leetcode测试通过
class Solution(object):
    def search(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        if not nums: return -1
        left, right = 0, len(nums) - 1
        while left <= right:
            mid = (left + right) / 2
            if nums[mid] == target:
                return mid
            if nums[mid] < nums[right]:
                if target > nums[mid] and target <= nums[right]:
                    left = mid + 1
                else:
                    right = mid - 1
            else:
                if target < nums[mid] and target >= nums[left]:
                    right = mid - 1
                else:
                    left = mid + 1
        return -1            
```