leetcode69-求平方根-二分法

题目地址：https://leetcode-cn.com/problems/sqrtx/

题目解析地址：https://leetcode-cn.com/problems/sqrtx/solution/x-de-ping-fang-gen-by-leetcode-solution/

题目大意：
实现 int sqrt(int x) 函数。

计算并返回 x 的平方根，其中 x 是非负整数。

由于返回类型是整数，结果只保留整数的部分，小数部分将被舍去。


题目解析：

分析这个问题，首先很简单我们知道4的平方根是2， 那么5的平方根是多少？也是2.

想一下是怎么算的，从1开始平方，看一下是不是等于5，如果等于，就是了，如果小于5，那么往后走一个到2，看一下2的平方是不是等于5，不是等于5，而是小于5，
所以继续走，3的平方根是9，大于5，说明就是2.

上面这个是最简单的一个方法，复杂度是O(n),也就是目标的增大，我的复杂度也是在增大的。


代码如下，一定要注意边界条件有个0，
```python
## 本代码由于复杂度的原因并没有通过leetcode的测试，但是基础思想和代码是没有问题的，只是在处理大数的时候内存爆掉了
class Solution:
    def mySqrt(self, x):
        temp=''
        for index in range(0,x+1):
            result = index * index
            if result == x:
                return index
            elif result<x:
                pass
            else:
                return index -1 
```

这么写是会内存报错的，测试用例输入的是：2147395599，失败了

说明我们需要优化.

想一下怎么优化：

其实很简单，上面这个过程本质上是在做一个 K^2<=target的流程，所以我们要做的就是对k做二分，判断中间那个值平方和目标值的关系。

我先是自己写了一个二分法，没成功，好久没刷题，全忘了，我先把正确答案写上来，然后放上我的垃圾二分法方便自己总结经验

```python 
## 本代码通过了Leetcode的测试
class Solution:
    def mySqrt(self, x) :
        l, r, ans = 0, x, -1
        while l <= r:
            mid = (l + r) // 2
            if mid * mid <= x:
                ans = mid
                l = mid + 1
            else:
                r = mid - 1
        return ans
```


对比最下面我写的垃圾失败的二分法，经验如下：

1. 首先确定边界值，一般是最左和最右，然后边界值会随着结果而调整

2. 然后需要注意的是中间值是：mid = (l + r) // 2  是加法，而不是减法：k = (right-left)//2，我脑袋秀逗了

3. 需要注意跳出环境 while l <= r， 我的如果没有整平方根的结果根本不能跳出：while:

4. 还需要注意的一点就是二分法移动的时候，不是最左边的往后移动一位，而是最左边的边界值变成中间值往后移动一位，想一下是不是这个道理。

5. 还需要注意的一点就是开始没有想明白的正确代码中二分查找不是一个标准的二分查找。我的意思就是，它不是最终等于某个值的时候输出，类似这种

```python
if mid * mid == x:
    ans = mid
elif mid * mid < x:
    l = mid + 1
else:
    r = mid - 1
```
所以在判断条件的时候，有个很奇怪的代码就是：
```
if mid * mid <= x:
    ans = mid
    l = mid + 1
else:
    r = mid - 1
```

怎么理解这个代码呢，很简单，不管这个边界怎么移动，我只要找到这个边界移动之后最后一个小于x的左边界就可以。


```python 
## 我写的垃圾失败二分法，失败了，为了吸取经验，就没删除，方便之后查找
class Solution:
    def mySqrt(self, x):
        left = 0
        right = x
        k=x//2
        result = k*k
        while:
            if result == x:
                return k
            elif result < x:
                left += 1
                k = (right-left)//2
                result = k*k
                
            else:
                right-=1
                k = (right-left)//2
                result = k*k

```












