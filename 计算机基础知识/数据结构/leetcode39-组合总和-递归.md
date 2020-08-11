leetcode39-组合总和-递归

题目链接： https://leetcode-cn.com/problems/combination-sum/

题目大意：
给定一个无重复元素的数组 candidates 和一个目标数 target ，找出 candidates 中所有可以使数字和为 target 的组合。candidates 中的数字可以无限制重复被选取。

说明：
所有数字（包括 target）都是正整数。
解集不能包含重复的组合。 

题目解析参考地址：https://blog.csdn.net/fuxuemingzhu/article/details/79322462


解析：

拿到这个题目，要去想，最笨的方法是可以解决的。首先，我们有一个数组，然后有一个target，比如说[2,3,4,5],target:8.

解题流程应该是这样的，首先拿掉一个2，也就是8-2=6，然后看剩下的这个6在这个数组加和的情况，比如我还拿掉2，那么剩下的是4，仍然是判断这个4对数组的加和情况。

所以，上述过程是一个递归的过程，就不是不停的在做同一个操作。

看代码：

```python
## 代码已经通过leetcode的测试
class Solution(object):
    def combinationSum(self, candidates, target):
        """
        :type candidates: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        res = []
        candidates.sort()
        self.dfs(candidates, target, 0, res, [])
        return res
    
    
    def dfs(self, nums, target, index, res, path):
        if target < 0:
            return
        elif target == 0:
            res.append(path)
            return
        for i in xrange(index, len(nums)):
            if nums[index] > target:
                return
            self.dfs(nums, target - nums[i], i, res, path + [nums[i]])
```

关于上述这个递归的过程，我在官方的解析中找到一个非常的图：
参考这里：https://leetcode-cn.com/problems/combination-sum/solution/hui-su-suan-fa-jian-zhi-python-dai-ma-java-dai-m-2/

在这个代码中，我当时觉得比较难以理解的地方在于比如上面我减去第一个2，得到6之后，递归的时候如何还是从2开始。

仔细看上面这个代码：
```python
for i in xrange(index, len(nums)):
    ......
    self.dfs(nums, target - nums[i], i, res, path + [nums[i]]) ##仔细理解这两行代码，从而符合官方解析的图，举上面那个例子，去琢磨，不好说出来
```
