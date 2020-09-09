剑指offer40-最小的K个数字

题目地址：https://leetcode-cn.com/problems/zui-xiao-de-kge-shu-lcof/

一般可以使用快排和堆排序解决这个问题。

## 堆排序

对于堆排序来讲，如果求最小的k个数字，就使用最大堆，如果求最大的k个数字，就使用最小堆。

这里我们最大堆，这个过程，我把它总结为两个步骤：首先，是把前k个数字构建成最大堆，在这个过程中，我们使用的是一个数组，注意index的变化。

第二个步骤是我们流式插入数字，每个数字都会和堆的顶点去进行比较，因为我是在求最小的数字，所以只有当小于当前堆的顶点数字的时候才需要进行更新最大堆。

在更新最大堆的时候，我们需要做的第一点就是我需要挑一个当前节点下左右子节点比较大那个节点来和我进行更换。

明白上面这些，写出一个堆排序就没有什么问题。
```python
## 堆排序 
class Solution(object):
    def getLeastNumbers(self, arr, k):
        """
        :type arr: List[int]
        :type k: int
        :rtype: List[int]
        """
        tinput=arr
        # 创建最大堆
        def createMaxHeap(num):
            maxHeap.append(num)
            currentIndex = len(maxHeap) - 1
            while currentIndex != 0:
                parentIndex = (currentIndex - 1) >> 1
                if maxHeap[parentIndex] < maxHeap[currentIndex] :
                    maxHeap[parentIndex] , maxHeap[currentIndex] = maxHeap[currentIndex],maxHeap[parentIndex] 
                    currentIndex = parentIndex
                else:
                    break 
                

        # 调整最大堆，也就是K个数字进来之后，后面的数字再进来进行的操作

        def adjustMaxHeap(num):
            if num < maxHeap[0]:
                maxHeap[0] = num

            maxHeapLen = len(maxHeap)
            index = 0
            while index < maxHeapLen :
                leftIndex = index * 2 +1
                rightIndex = index *2 +2
                largerIndex = 0
                if rightIndex < maxHeapLen: ## 我需要挑一个当前节点下左右子节点比较大那个节点来和我进行更换
                    if maxHeap[rightIndex] < maxHeap[leftIndex]:
                        largerIndex = leftIndex
                    else:
                        largerIndex = rightIndex
                elif leftIndex < maxHeapLen :
                    largerIndex = leftIndex
                else:
                    break
            

                if maxHeap[index] < maxHeap[largerIndex]:
                    maxHeap[index],maxHeap[largerIndex] = maxHeap[largerIndex],maxHeap[index]
                index = largerIndex


        maxHeap = []

        if len(tinput) < k or k <= 0:
            return []

        for i in range(len(tinput)):
            if i < k:
                createMaxHeap(tinput[i])
            else:
                adjustMaxHeap(tinput[i])
                
        maxHeap.sort()
        return maxHeap
```

## 快速排序