剑指offer24-反转链表-规律

题目链接：
https://leetcode-cn.com/problems/fan-zhuan-lian-biao-lcof/

题目解析：

整个流程其实清晰，对于链表来说，分为两个部分，值和指向下一个节点的指针。

如果要做到翻转链表，最直观的想法改变节点指针指向的方向，由指向下一个节点改为指向上一个节点。

这个操作是顺序实行的，就是一个节点一个节点的过，走流程，改变的指针指向。

但是存在最大的问题，当前把指针指向改为了上一个节点，那么我怎么知道在改变之后下一个节点是什么？

所以我需要另一个节点，帮助我存储当前节点的下一个节点。所以核心代码就是如下：

fair_node = node.next ## node是我当前节点，首先我使用第三方的一个节点存储我下一个节点，防止我改指针方向之后找不到下一个节点
node.next = pre ## 这个时候把我当前节点指针指向改为指向上一个节点，初始上一个节点也就是首节点的上一个节点应该是None
pre =node ## 我要考虑后面代码的执行情况，也就是存储我当前节点为上一个节点
node = fair_node ## 往后面移动，进行下一个节点的执行

直接看代码

```python
## 代码已经通过leetcode的测试
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def reverseList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        node = head
        pre=None
        while node:
            fair_node = node.next
            node.next = pre
            pre =node
            node = fair_node 
        return pre
```
        





# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def reverseList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        pre=ListNode()
        
        node = head

        next = node.next

