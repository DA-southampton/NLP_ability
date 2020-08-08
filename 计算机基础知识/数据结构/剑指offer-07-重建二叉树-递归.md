剑指offer-07-重建二叉树-递归

题目地址：
https://leetcode-cn.com/problems/zhong-jian-er-cha-shu-lcof/

题目解析：
B站博主讲解视频地址： https://www.bilibili.com/video/BV1K4411o7KP?p=32


题目大意：

给出二叉树的前序遍历和中序遍历结果，请重建二叉树，假设前序遍历和中序遍历都不含重复数字。

例子，给出：
前序遍历 preorder = [1,2,4,7,3,5,6,8]
中序遍历 inorder = [4,7,2,1,5,3,8,6]


解析：

首先，我们需要理解前序遍历和中序遍历的概念。前序中序后序都是相对根节点来说的，比如说前序遍历，首先打印根节点，其次依次打印左节点和右节点。当我们的视线跳到左节点的时候，
这个时候又是一个子树，所以还是依次根节点/左节点/右节点。这是一个递归的过程。

我们看前序遍历中，首先1这个数字是整个二叉树的根节点，这点没有问题。然后2这个点，按道理应该是左子树，但是如果没有左子树，那就是右子树的节点。这个时候就出现分歧了。这个时候，
我们就需要借助中序结构，发现2是在1这个节点左侧的，所以它对应的就是左子树上的数字。

也就是，我们按照前序遍历来看，辅助中序遍历信息。

解题的过程是一个递归的过程，为什么这么说呢？首先，我们找到整个二叉树的根节点，其次我们可以发现一个特点，就是在中序遍历根节点的左边，是根节点的左子树，
在这里就是[4,7,2]，对应到前序遍历就是根节点之后对应长度[2,,4,7]。得到这个之后，不就又是我们相同的题目吗，给一个前序遍历和中序遍历构建左子树的二叉树

同理可得，右子树[5,3,8,6]，和[3,5,6,8]，同样是构建右子树的二叉树。

所以这是个递归的问题。

直接看代码

```python 
## 代码测试在leetcode已经通过
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def buildTree(self, preorder, inorder):
        """
        :type preorder: List[int]
        :type inorder: List[int]
        :rtype: TreeNode
        """
        # write code here
        if not preorder or not inorder:
            return None
        if len(preorder) != len(inorder):
            return None
        # 取出pre 的第一个值  就是根节点
        root = preorder[0]
        rootNode = TreeNode(root)
        # 找到在 tin  中序遍历中的根节点 所在的索引位置
        pos = inorder.index(root)
        # 中序遍历的 列表的左右节点 分开 切片 成两个列表
        tinLeft = inorder[0:pos]
        tinRight = inorder[pos + 1:]
        # 前序遍历的 列表的左右节点 分开 切片 成两个列表
        preLeft = preorder[1:pos + 1]
        preRight = preorder[pos + 1:]

        leftNode = self.buildTree(preLeft, tinLeft)
        rightNode = self.buildTree(preRight, tinRight)

        rootNode.left = leftNode
        rootNode.right = rightNode
        return rootNode
```