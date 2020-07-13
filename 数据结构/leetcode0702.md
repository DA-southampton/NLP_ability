## 盛最多的水  11
https://blog.csdn.net/fuxuemingzhu/article/details/82822939
https://zhuanlan.zhihu.com/p/40616691
我们需要得到的是长方形面积（宽是两条板之间的距离，高比较短的的那个板子的高度），这个代表我们的蓄水面积。

所以重点在这里，蓄水面积包含两个部分：距离+较短板子高度

我们的思路是这样的，首先我确保一个部分达到最大，然后再去慢慢优化板子的高度。

也就是说，首先，我两个指针，一个在首位，一个在末位，这个时候我们的宽度是最大的，计算一下这个时候的面积大小。

然后，我们选择两个板子里面的较短的那一个进行移动，也就是在此基础之上，我想要优化这个板子的高度，所以抛弃掉那个短板子，移动下一个板子。

这个时候，我们想一下，如果移动后的这个板子比初始的这个板子小，肯定面积是减小了，因为宽度小了，高度也小了。

如果大呢？那就不确定了，因为宽度小了，虽然板子高度大了，但是面积就不确定了

## 3sum  15

遍历 O(n3)

优化：
https://blog.csdn.net/fuxuemingzhu/article/details/83115850
https://zhuanlan.zhihu.com/p/53519899
https://zhuanlan.zhihu.com/p/104330759
class Solution(object):
    def threeSum(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        N = len(nums)
        nums.sort()
        res = []
        for t in range(N - 2):
            if t > 0 and nums[t] == nums[t - 1]:## 去掉因为第一个元素存在而出现的相同结果，当然可能第一个元素相同，但是没有结果这种情况
                    continue ## 也就是说避免重复找同一个数字开端的结果，不管这个因为这个数字有没有结果。
            i, j = t + 1, N - 1
            while i < j:
                _sum = nums[t] + nums[i] + nums[j]
                if _sum == 0:
                    res.append([nums[t], nums[i], nums[j]])
                    i += 1
                    j -= 1
                    while i < j and nums[i] == nums[i - 1]: ## 去掉因为i相同而帅选出来的相同结果
                        i += 1
                    while i < j and nums[j] == nums[j + 1]: ## 去掉因为j相同而筛选出来的相同结果
                        j -= 1
                elif _sum < 0:
                    i += 1
                else:
                    j -= 1
        return res


## 电话号码  17
https://blog.csdn.net/fuxuemingzhu/article/details/79363119

class Solution(object):
    def letterCombinations(self, digits):
        """
        :type digits: str
        :rtype: List[str]
        """
        if digits == "": return []
        d = {'2' : "abc", '3' : "def", '4' : "ghi", '5' : "jkl", '6' : "mno", '7' : "pqrs", '8' : "tuv", '9' : "wxyz"}
        res = ['']
        for e in digits:
            res = [w + c for c in d[e] for w in res]
        return res
这个过程就是比如进来是三个数字，先把前两个数字的各种组合组合起来，然后这个结果再和第三个组合起来


https://zhuanlan.zhihu.com/p/53219687
class Solution:
    def letterCombinations(self, digits):
        """
        :type digits: str
        :rtype: List[str]
        """

        # 创建字母对应的字符列表的字典
        dic = {2: ['a', 'b', 'c'],
               3: ['d', 'e', 'f'],
               4: ['g', 'h', 'i'],
               5: ['j', 'k', 'l'],
               6: ['m', 'n', 'o'],
               7: ['p', 'q', 'r', 's'],
               8: ['t', 'u', 'v'],
               9: ['w', 'x', 'y', 'z'],
               }
        # 存储结果的数组
        ret_str = []
        if len(digits) == 0: return []

        # 递归出口，当递归到最后一个数的时候result拿到结果进行for循环遍历
        if len(digits) == 1:
            return dic[int(digits[0])]
        # 递归调用
        result = self.letterCombinations(digits[1:])
        # result是一个数组列表，遍历后字符串操作，加入列表
        for r in result:
            for j in dic[int(digits[0])]:
                ret_str.append(j + r)
        return ret_str


if __name__ == '__main__':
    s = Solution()
    print(s.letterCombinations('23'))

这使用的是递归的思想，他的本质我大概想了一下，应该是比如输入的是2345，先把45的遍历一遍，然后慢慢往回倒，一层层的走出来。

递归的思想，是这样的，首先的一步是不停的网深处走，到最后一层返回一个结果，然后在一层层的往回走

递归本质是在不停的使用自己这个函数最后完成一个结果，比如说遇到最后一层了，然后return一个结果，这个结果返回上一层，然后进行下一步计算。

举个例子， 输入2345 ，debug的话或者自己想，会发现，最后输入的是45，这个时候
result = self.letterCombinations(digits[1:])
就变成了，result = self.letterCombinations(‘5’)
把这个5 带进去就发现了，
if len(digits) == 1:
            return dic[int(digits[0])]

遇到这个函数返回过来，这个返回的结果，就是我们result = self.letterCombinations(‘5’) 的result

这个时候，我们才会继续进行下个步骤
for r in result:
            for j in dic[int(digits[0])]:
                ret_str.append(j + r)
        return ret_str

然后这个返回结构，在作为result = self.letterCombinations(digits[1:])  result = self.letterCombinations(‘45’)
也就是 输入为345 的结果