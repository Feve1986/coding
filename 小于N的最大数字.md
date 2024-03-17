from typing import List
"""
arr:可使用数字的数组
nums:目标数字对应的整数数组
preEq:之前是否每一位都使用了相等的值
index:当前考虑到了第几位，从0开始
函数语义：返回第index位后可组成的最大的数字，包括第index位
"""
def dfs(arr, nums,  preEq,  index):
    #如果考虑完了最后一位，那么能组成的最大值为0
    if index==len(nums):
        return 0
 
    if preEq:#如果前面使用的都是相等的数字
        for i in range(len(arr)-1,-1,-1):#从最大的往下排
            if index==len(nums)-1:
                if arr[i]<nums[index]:#在前面使用的都是相同的数字，那么最后一位的数字不能是相等的，这个看具体面试官的要求
                    temp = dfs(arr, nums, arr[i] == nums[index], index + 1)
                    if temp==-1:#向下深搜发现没有满足要求的值，则选用更小的数字
                        continue
                    # 向下深搜发现了满足要求的值，那说明我们找到结果了，把这一层的结果加进去，之后层层返回即可
                    return arr[i] * pow(10,len(nums)-index-1) + temp
            else:
                if arr[i]<=nums[index]:#非最后一位的数字是可以相等的
                    temp = dfs(arr, nums, arr[i] == nums[index], index + 1)
                    if temp == -1:
                        continue
                    return arr[i] * pow(10, len(nums) - index - 1) + temp
        #如果上面的for循环走完都没有return的话，那就说明这一层所有的数字都不满足要求
        #如果是第一个数字都无法满足的话，那就直接从第二位开始，将preEq置为False递归
        if index==0:
            return dfs(arr, nums, False, index + 1)
        # 如果上面的for循环走完都没有return的话，而且也不是第一位，那就返回-1，回退递归
        return -1
    else:
        #如果前面不是都选用了相等的数字，那就直接每次挑最大的数字就行了。
        return arr[len(arr)-1] * pow(10, len(nums) - index - 1) + dfs(arr, nums, False, index + 1)
 
 
def find(N,arr: List[int]) -> int:
    arr.sort()
    nums = [int(i) for i in str(N)]
    #递归的初始条件，preEq=True，index=0
    return dfs(arr,nums,True,0)
 
 
 
arrInputList = [
[2,3,4,5],
[2,3,4,5],
[2,3,4,5],
[2,3,4,5]
]
NList =   [1234,2234,2231,2134]
ansList = [555, 2233,2225,555]
for i in range(len(NList)):
    ans = find(NList[i], arrInputList[i])
    print(NList[i], arrInputList[i],ans)
