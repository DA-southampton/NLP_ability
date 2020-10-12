leetcode701-二叉搜索树插入某个节点

题目链接：https://leetcode-cn.com/problems/insert-into-a-binary-search-tree/

在插入某个节点之后，二叉搜索树应该还保持二叉搜索树的特质，也就是左子树仍然比根节点小，右子树仍然比根节点大。

插入的过程在本质上和做二叉树的搜索很类似。

```python
## 代码已经测试通过
class Solution:
    def insertIntoBST(self, root, val):
        if not root:
            return TreeNode(val)
        
        pos = root
        while pos:
            if val < pos.val:
                if not pos.left:
                    pos.left = TreeNode(val)
                    break
                else:
                    pos = pos.left
            else:
                if not pos.right:
                    pos.right = TreeNode(val)
                    break
                else:
                    pos = pos.right
        
        return root

```
