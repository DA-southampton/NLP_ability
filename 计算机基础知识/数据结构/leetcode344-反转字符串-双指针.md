leetcode344-反转字符串-双指针

题目地址：
https://leetcode-cn.com/problems/reverse-string/

题目解析：

这道题目很简单，首先别调用函数，没意义，使用双指针，头尾给一个，然后对应位置交换就可以了。直接看代码：

```python
## 本代码leetcode测试已经通过
class Solution(object):
    def reverseString(self, s):
        """
        :type s: List[str]
        :rtype: None Do not return anything, modify s in-place instead.
        """
        left , right = 0 , len(s)-1
        while left<right:
            s[left],s[right] = s[right],s[left]
            left += 1
            right -= 1

```

