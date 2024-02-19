# 括号生成，回溯法
class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        
        def dfs(path, left, right):
            if right>left: return
            if left>n or right>n: return
            if left==n and right==n: 
                res.append(path)
                return
            dfs(path+'(', left+1, right)
            dfs(path+')', left, right+1)
            
        res = []
        dfs('', 0, 0)
        return res
