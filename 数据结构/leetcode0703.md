## 19  删除链表的倒数第N个节点
题目链接： https://leetcode-cn.com/problems/remove-nth-node-from-end-of-list/
解析链接： https://blog.csdn.net/fuxuemingzhu/article/details/80786149
使用双指针，快慢指针
看代码我的解析：

class Solution(object):
    def removeNthFromEnd(self, head, n):
        """
        :type head: ListNode
        :type n: int
        :rtype: ListNode
        """
        root = ListNode(0)
        root.next = head
        fast, slow, pre = root, root, root
        while n - 1: ## 比如说我要是倒数第二个节点，那么快慢指针的距离应该是多少？想一下
        ## 比如给定一个链表: 1->2->3->4->5, 和 n = 2. 那么快指针应该是在5，慢指针应该是在4，两者相差的是1，所以这里走n-1 个
            fast = fast.next
            n -= 1
        while fast.next:
            fast = fast.next
            pre = slow
            slow = slow.next
        pre.next = slow.next
        return root.next

## 31 下一个排列
题目
https://leetcode-cn.com/problems/next-permutation/

LeetCode 题解 | 31. 下一个排列 - 力扣（LeetCode）的文章 - 知乎
https://zhuanlan.zhihu.com/p/45007701
才发现现在是有官方题目解析的

题目解析： 
https://blog.csdn.net/fuxuemingzhu/article/details/82113409

核心思想，我的理解是这样的
1　　2　　7　　4　　3　　1
7431 是一个完全降序的排列，对于一个完全降序的排列，你只能变成一个最小值才是他的下一个序列

也就是 1 2 1 3 4 7
还需要注意一点，这个针对的是 7 4 3 1 
我们好需要看到 2 ，应该让序列增大一个，也就是把大于2的这个值和2调换顺序

## 搜索旋转排序数组
题目在这里： https://leetcode-cn.com/problems/search-in-rotated-sorted-array/
题目解析在这里 https://blog.csdn.net/fuxuemingzhu/article/details/79534213

我的理解是这样的，是一个二分查找的问题
首先判断哪边是有序的，然后判断我这个值应该出现在有序的还是无序的那边，从而移动左右指针，继续进一步二分查找


## 39 组合综合
给定一个无重复元素的数组 candidates 和一个目标数 target ，找出 candidates 中所有可以使数字和为 target 的组合。


题目链接：
https://leetcode-cn.com/problems/combination-sum/
题目解析：
https://blog.csdn.net/fuxuemingzhu/article/details/79322462


本质是个递归，依次遍历数组元素，当前元素和target与之前元素的差值的大小比较，直至为0

## 46全排列
设计到一个回溯方法
这里有一个回溯方法总结
【LeetCode】回溯法总结 - 鱼枕的文章 - 知乎
https://zhuanlan.zhihu.com/p/63252392

## 48 旋转图像
题目位置：https://leetcode-cn.com/problems/rotate-image/

题目解析：https://blog.csdn.net/fuxuemingzhu/article/details/79451733

矩阵旋转90度，我觉得需要注意的是不可能按照九十度来做，而是看90度对应的是什么规律
发现，90度对应的是先上下翻转，再按照左上到右下的对角线进行翻转

## 55 跳跃游戏
题目位置：https://leetcode-cn.com/problems/jump-game/
题目解析： https://blog.csdn.net/fuxuemingzhu/article/details/83504437
方法使用的是贪心
这道题的核心在于实时维持一个可以达到的最大位置，如果此时的索引大于这个值，说明根本到不了这里

## 56 合并区间
题目位置：https://leetcode-cn.com/problems/merge-intervals/
解析：https://blog.csdn.net/fuxuemingzhu/article/details/69078468

核心点在 首先按照区间首位排序，随后merge

## 62 不同路径
题目位置：https://leetcode-cn.com/problems/unique-paths/
题目解析：https://zhuanlan.zhihu.com/p/43358393

本质是一个规律问题

## 64 最小路径和
题目位置：https://leetcode-cn.com/problems/minimum-path-sum/
解析：https://blog.csdn.net/weixin_38278334/article/details/89930610

这个其实和62很类似，我觉得本质的思想在于：当前点的值不是来自上方和我自身相加，就是左边的值和我自身相加，那么我选择最小的那个
确保我自身是最小的就可以。

## 215 数组中的最大第K个元素
题目：https://leetcode-cn.com/problems/kth-largest-element-in-an-array/
题目解析：https://blog.csdn.net/fuxuemingzhu/article/details/79264797

循环，每次去除一个最大值

以后看看有咩有更好的方法



leetcode中动态规划总结 知乎搜搜