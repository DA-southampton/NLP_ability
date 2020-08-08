## 电话号码  17

https://blog.csdn.net/fuxuemingzhu/article/details/79363119

```python
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
```

这个过程就是比如进来是三个数字，先把前两个数字的各种组合组合起来，然后这个结果再和第三个组合起来


https://zhuanlan.zhihu.com/p/53219687
```python
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
```

这使用的是递归的思想，他的本质我大概想了一下，应该是比如输入的是2345，先把45的遍历一遍，然后慢慢往回倒，一层层的走出来。

递归的思想，是这样的，首先的一步是不停的网深处走，到最后一层返回一个结果，然后在一层层的往回走

递归本质是在不停的使用自己这个函数最后完成一个结果，比如说遇到最后一层了，然后return一个结果，这个结果返回上一层，然后进行下一步计算。

举个例子， 输入2345 ，debug的话或者自己想，会发现，最后输入的是45，这个时候

result = self.letterCombinations(digits[1:])

就变成了，

result = self.letterCombinations(‘5’)

把这个5 带进去就发现了，

if len(digits) == 1:
            return dic[int(digits[0])]

遇到这个函数返回过来，这个返回的结果，就是我们result = self.letterCombinations(‘5’) 的result

这个时候，我们才会继续进行下个步骤

```python
for r in result:
            for j in dic[int(digits[0])]:
                ret_str.append(j + r)
        return ret_str
```

然后这个返回结构，在作为result = self.letterCombinations(digits[1:])  result = self.letterCombinations(‘45’)
也就是 输入为345 的结果
