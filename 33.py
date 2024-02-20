# 搜索旋转排序数组
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        n = len(nums)
        def find(nums, target):
            # nums[left-1]>=target
            # nums[right+1]<target
            left, right = 0, n-1
            while left<=right:
                mid = (left+right)//2
                if nums[mid] >= target:
                    left = mid+1
                else:
                    right = mid-1
            return left
        def find2(nums, target):
            left, right = 0, n-1
            # nums[left-1]<target
            # nums[right+1]>=target
            while left<=right:
                mid = (left+right)//2
                if nums[mid]<target:
                    left = mid+1
                else:
                    right = mid-1
            return left

        start = find(nums, nums[0])
        index = start%n
        new_nums = nums[index:n] + nums[:index]
        ans = find2(new_nums, target)
        if ans==n or new_nums[ans]!=target: 
            return -1
        else: 
            return (ans+index)%n
