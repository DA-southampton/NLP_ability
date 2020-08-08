剑指Offer27-二叉树的镜像-递归

题目链接：https://leetcode-cn.com/problems/er-cha-shu-de-jing-xiang-lcof/

题目解析：
B站UP题目解析地址：https://www.bilibili.com/video/BV1K4411o7KP?p=33

题目大意：
输入一个二叉树，该函数输出它的镜像。


很简单，直接递归就是，首先处理根节点，然后处理左右子树

```python
##以下代码在leetcode测试通过
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def mirrorTree(self, root):
        # write code here
        if root == None:
            return None
        #处理根节点
        root.left,root.right = root.right,root.left
        self.mirrorTree(root.left)
        self.mirrorTree(root.right)
        return root
```