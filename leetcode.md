###### 一维动态规划：
* 打家劫舍：你是一个专业的小偷，计划偷窃沿街的房屋。每间房内都藏有一定的现金，影响你偷窃的唯一制约因素就是相邻的房屋装有相互连通的防盗系统，
  如果两间相邻的房屋在同一晚上被小偷闯入，系统会自动报警。给定一个代表每个房屋存放金额的非负整数数组，计算你 不触动警报装置的情况下 ，一夜之内能够偷窃到的最高金额。
* 零钱兑换：给你一个整数数组 coins ，表示不同面额的硬币；以及一个整数 amount ，表示总金额。
  计算并返回可以凑成总金额所需的 最少的硬币个数 。如果没有任何一种硬币组合能组成总金额，返回 -1 。
  你可以认为每种硬币的数量是无限的。
  > dp = [0] + [float('inf')] * amount, dp[i] = min(dp[i],dp[i-coin] + 1)
* 单词拆分：给你一个字符串 s 和一个字符串列表 wordDict 作为字典。如果可以利用字典中出现的一个或多个单词拼接出 s 则返回 true。
  注意：不要求字典中出现的单词全部都使用，并且字典中的单词可以重复使用。
  > dp = [True] + [False] * n, if dp[i] and s[i: j] in wordDict: dp[j] = True
* 最长递增子序列：给你一个整数数组 nums ，找到其中最长严格递增子序列的长度。
  > dp = len(nums) * [1], if nums[i]>nums[j]: dp[i]=max(dp[i], dp[j]+1)
* 乘积最大子数组: 给你一个整数数组 nums ，请你找出数组中乘积最大的非空连续子数组
  （该子数组中至少包含一个数字），并返回该子数组所对应的乘积。测试用例的答案是一个 32-位 整数。
  > dp_max = [0] * len(nums), dp_min = [0] * len(nums)
  > dp_max[i] = max([dp_max[i-1] * ele, dp_min[i-1] * ele, ele]), dp_min[i] = min([dp_max[i-1] * ele, dp_min[i-1] * ele, ele])
* 分割等和子集：给你一个 只包含正整数 的 非空 数组 nums 。请你判断是否可以将这个数组分割成两个子集，使得两个子集的元素和相等。
  > 其实就是看是否存在若干元素和为total_sum // 2，和零钱兑换问题相似  
  > dp[i] = dp[i] or dp[i - num]
* 买卖股票的最佳时机：给定一个数组 prices ，它的第 i 个元素 prices[i] 表示一支给定股票第 i 天的价格。
  你只能选择 某一天 买入这只股票，并选择在 未来的某一个不同的日子 卖出该股票。设计一个算法来计算你所能获取的最大利润。
  返回你可以从这笔交易中获取的最大利润。如果你不能获取任何利润，返回 0 。
  > dp[i] = max(dp[i-1] + diff, 0)
* 最大子数组和：给你一个整数数组 nums ，请你找出一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。
  > dp[i] = max(dp[i-1]+nums[i], nums[i])
* 分割回文串：给你一个字符串 s，请你将 s 分割成一些子串，使每个子串都是回文串。返回 s 所有可能的分割方案。
  > dp[-1].extend([l+[temp] for l in dp[j]])

###### 多维动态规划：dp = [[0]*(n+1) for _ in range(m+1)]
* 编辑距离：给你两个单词 word1 和 word2， 请返回将 word1 转换成 word2 所使用的最少操作数。
  你可以对一个单词进行如下三种操作：插入一个字符，删除一个字符，替换一个字符
> if word1[i-1] == word2[j-1]:
    dp[i][j] = dp[i-1][j-1]
  else:
    dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])+1
* 不同路径：一个机器人位于一个 m x n 网格的左上角 （起始点在下图中标记为 “Start” ）。
  机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为 “Finish” ）。问总共有多少条不同的路径？
  > dp = [[1]*m for _ in range(n)], dp[j][i] = dp[j][i-1] + dp[j-1][i]
* 最小路径和：给定一个包含非负整数的 m x n 网格 grid ，请找出一条从左上角到右下角的路径，使得路径上的数字总和为最小。说明：每次只能向下或者向右移动一步。
  > dp[i][j] = min(dp[i-1][j], dp[i][j-1]) + grid[i][j]
* 最长公共子序列：给定两个字符串 text1 和 text2，返回这两个字符串的最长公共子序列的长度。如果不存在公共子序列 ，返回 0 。
  一个字符串的 子序列 是指这样一个新的字符串：它是由原字符串在不改变字符的相对顺序的情况下删除某些字符（也可以不删除任何字符）后组成的新字符串。
  例如，"ace" 是 "abcde" 的子序列，但 "aec" 不是 "abcde" 的子序列。两个字符串的 公共子序列 是这两个字符串所共同拥有的子序列。
  > if text1[i-1]==text2[j-1]:
      dp[i][j] = dp[i-1][j-1]+1
  else:
      dp[i][j] = max(dp[i-1][j], dp[i][j-1])
  
###### 链表
* 两两交换链表中的节点（递归）：给你一个链表，两两交换其中相邻的节点，并返回交换后链表的头节点。你必须在不修改节点内部的值的情况下完成本题（即，只能进行节点交换）。
  > head1.next = self.swapPairs(head3)
* K 个一组翻转链表（迭代）：给你链表的头节点 head ，每 k 个节点一组进行翻转，请你返回修改后的链表。
  k 是一个正整数，它的值小于或等于链表的长度。如果节点总数不是 k 的整数倍，那么请将最后剩余的节点保持原有顺序。
  你不能只是单纯的改变节点内部的值，而是需要实际进行节点交换。
* 两数相加（迭代）：给你两个 非空 的链表，表示两个非负的整数。它们每位数字都是按照 逆序 的方式存储的，并且每个节点只能存储 一位 数字。
  请你将两个数相加，并以相同形式返回一个表示和的链表。你可以假设除了数字 0 之外，这两个数都不会以 0 开头。
* 删除链表的倒数第 N 个结点（快慢指针）：给你一个链表，删除链表的倒数第 n 个结点，并且返回链表的头结点。
* 随机链表的复制：给你一个长度为 n 的链表，每个节点包含一个额外增加的随机指针 random ，该指针可以指向链表中的任何节点或空节点。
  构造这个链表的 深拷贝。 深拷贝应该正好由 n 个 全新 节点组成，其中每个新节点的值都设为其对应的原节点的值。
  新节点的 next 指针和 random 指针也都应指向复制链表中的新节点，并使原链表和复制链表中的这些指针能够表示相同的链表状态。复制链表中的指针都不应指向原链表中的节点 。
  例如，如果原链表中有 X 和 Y 两个节点，其中 X.random --> Y 。那么在复制链表中对应的两个节点 x 和 y ，同样有 x.random --> y 。
  返回复制链表的头节点。
> cur.next.random = cur.random.next
* 排序链表：给你链表的头结点 head ，请将其按 升序 排列并返回 排序后的链表 。先把链表元素放到数组中进行排序。
* 合并 K 个升序链表（最小堆）：给你一个链表数组，每个链表都已经按升序排列。请你将所有链表合并到一个升序链表中，返回合并后的链表。
  > 可以通过增加比较规则使得ListNode能够heapify：ListNode.__lt__ = lambda a, b: a.val < b.val。也可以用下面的方式重构为元组的形式。  
  ```python
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        
        dummy=curr=ListNode(0)
        heap=[]
        for idx, head in enumerate(lists):
            if head:
                heappush(heap, (head.val, idx))
                lists[idx]=lists[idx].next

        while heap:
            val,idx=heappop(heap)
            curr.next=ListNode(val)
            curr=curr.next
            if lists[idx]:
                heappush(heap, (lists[idx].val, idx))
                lists[idx]=lists[idx].next
        
        return dummy.next
  ```
* LRU 缓存（双向链表）：请你设计并实现一个满足  LRU (最近最少使用) 缓存 约束的数据结构。
  实现 LRUCache 类：
  LRUCache(int capacity) 以 正整数 作为容量 capacity 初始化 LRU 缓存
  int get(int key) 如果关键字 key 存在于缓存中，则返回关键字的值，否则返回 -1 。
  void put(int key, int value) 如果关键字 key 已经存在，则变更其数据值 value ；如果不存在，则向缓存中插入该组 key-value 。如果插入操作导致关键字数量超过 capacity ，则应该 逐出 最久未使用的关键字。
  > 双向链表  
  ```
  class DLinkedNode:  
    def __init__(self, key=0, value=0):  
        self.key = key  
        self.value = value  
        self.prev = None  
        self.next = None
  ```  
  > self.cache = collections.OrderedDict()  
  > self.cache.move_to_end(key)  
  > self.cache.popitem(last=False)  
* 环形链表 II（快慢指针）：给定一个链表的头节点  head ，返回链表开始入环的第一个节点。 如果链表无环，则返回 null。
  > ptr = head
  while ptr != slow:

###### 二叉树（dfs）
* 二叉树的层序遍历（递归）：给你二叉树的根节点 root ，返回其节点值的 层序遍历 。 （即逐层地，从左到右访问所有节点）。
  ```python
  def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
      if not root: return []
      queue=collections.deque()
      queue.append(root)
      res=[]
      while queue:
          n=len(queue)
          h=[]
          for _ in range(n):
              node=queue.popleft()
              h.append(node.val)
              if node.left:
                  queue.append(node.left)
              if node.right:
                  queue.append(node.right)
          res.append(h)
      return res
  ```
* 验证二叉搜索树（递归）：给你一个二叉树的根节点 root ，判断其是否是一个有效的二叉搜索树。有效 二叉搜索树定义如下：
  节点的左子树只包含 小于 当前节点的数。节点的右子树只包含 大于 当前节点的数。所有左子树和右子树自身必须也是二叉搜索树。
* 二叉搜索树中第K小的元素（递归）：给定一个二叉搜索树的根节点 root ，和一个整数 k ，请你设计一个算法查找其中第 k 个最小元素（从 1 开始计数）。
* 二叉树的右视图（递归）：给定一个二叉树的 根节点 root，想象自己站在它的右侧，按照从顶部到底部的顺序，返回从右侧所能看到的节点值。
* 二叉树的最近公共祖先（递归）：给定一个二叉树, 找到该树中两个指定节点的最近公共祖先。
* 路径总和 III：给定一个二叉树的根节点 root ，和一个整数 targetSum ，求该二叉树里节点值之和等于 targetSum 的 路径 的数目。
  路径 不需要从根节点开始，也不需要在叶子节点结束，但是路径方向必须是向下的（只能从父节点到子节点）。
  ```python
  def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
      if not root or root==q or root==p: return root
      left=self.lowestCommonAncestor(root.left, p, q)
      right=self.lowestCommonAncestor(root.right, p, q)
      if left and right: return root
      if left: return left
      if right: return right
  ```
  > 前缀和
* 二叉树展开为链表：给你二叉树的根结点 root ，请你将它展开为一个单链表：
  展开后的单链表应该同样使用 TreeNode ，其中 right 子指针指向链表中下一个结点，而左子指针始终为 null 。
  展开后的单链表应该与二叉树 先序遍历 顺序相同。
  ```python
  def flatten(self, root: Optional[TreeNode]) -> None:
      """
      Do not return anything, modify root in-place instead.
      """
      def dfs(root):
          if not root: return 
          dfs(root.left)
          dfs(root.right)
          l=root.left
          if l:
              end=l
              while end.right:
                  end=end.right
              end.right=root.right
              root.right=l
              root.left=None
      dfs(root)
  ```
* 对称二叉树：给你一个二叉树的根节点 root ， 检查它是否轴对称。
  ```python
  def isSymmetric(self, root: Optional[TreeNode]) -> bool:
      def dfs(left, right):
          if not (left or right):
              return True
          if not (left and right):
              return False
          if left.val!=right.val:
              return False
          return dfs(left.left, right.right) and dfs(left.right, right.left)

      return dfs(root.left, root.right)
  ```
* 将有序数组转换为二叉搜索树：给你一个整数数组 nums ，其中元素已经按 升序 排列，请你将其转换为一棵 
平衡二叉搜索树。
  ```python 
  def sortedArrayToBST(self, nums: List[int]) -> Optional[TreeNode]:
      def dfs(nums):
          if not nums: return None
          l,r=0,len(nums)
          if l==r: return TreeNode(nums[l])
          mid=(l+r)//2
          root=TreeNode(nums[mid])
          root.left=dfs(nums[l:mid])
          root.right=dfs(nums[mid+1:r])
          return root

      return dfs(nums)
  ```

###### 递归
* 完全平方数（任何正整数都可以由最多四个数的完全平方数表示）：给你一个整数 n ，返回 和为 n 的完全平方数的最少数量 。
  完全平方数 是一个整数，其值等于另一个整数的平方；换句话说，其值等于一个整数自乘的积。例如，1、4、9 和 16 都是完全平方数，而 3 和 11 不是。

###### 回溯（一类特殊的递归，backtrack）
> 回溯：采用试错的思想，它尝试分步的去解决一个问题。在分步解决问题的过程中，当它通过尝试发现现有的分步答案不能得到有效的正确的解答的时候，它将取消上一步甚至是上几步的计算，再通过其它的可能的分步解答再次尝试寻找问题的答案。回溯法通常用最简单的递归方法来实现，在反复重复上述的步骤后可能出现两种情况：
找到一个可能存在的正确的答案；
在尝试了所有可能的分步方法后宣告该问题没有答案。  
> 深度优先搜索算法（英语：Depth-First-Search，DFS）是一种用于遍历或搜索树或图的算法。这个算法会 尽可能深 的搜索树的分支。当结点 v 的所在边都己被探寻过，搜索将 回溯 到发现结点 v 的那条边的起始结点。这一过程一直进行到已发现从源结点可达的所有结点为止。如果还存在未被发现的结点，则选择其中一个作为源结点并重复以上过程，整个进程反复进行直到所有结点都被访问为止。
* 全排列：给定一个不含重复数字的数组 nums ，返回其 所有可能的全排列 。你可以 按任意顺序 返回答案。
  > ans.append(curr[:])
* 子集：给你一个整数数组 nums ，数组中的元素 互不相同 。返回该数组所有可能的子集（幂集）。
  解集 不能 包含重复的子集。你可以按 任意顺序 返回解集。（记得pop）
* 电话号码的字母组合：给定一个仅包含数字 2-9 的字符串，返回所有它能表示的字母组合。答案可以按 任意顺序 返回。
  给出数字到字母的映射如下（与电话按键相同）。注意 1 不对应任何字母。
  > 在数组中直接覆盖可以替换pop操作
* 单词搜索：给定一个 m x n 二维字符网格 board 和一个字符串单词 word 。如果 word 存在于网格中，返回 true ；否则，返回 false 。
  单词必须按照字母顺序，通过相邻的单元格内的字母构成，其中“相邻”单元格是那些水平相邻或垂直相邻的单元格。同一个单元格内的字母不允许被重复使用。
  
###### 贪心算法
* 跳跃游戏：给你一个非负整数数组 nums ，你最初位于数组的 第一个下标 。数组中的每个元素代表你在该位置可以跳跃的最大长度。
  判断你是否能够到达最后一个下标，如果可以，返回 true ；否则，返回 false 。
* 跳跃游戏 II：给定一个长度为 n 的 0 索引整数数组 nums。初始位置为 nums[0]。
  每个元素 nums[i] 表示从索引 i 向前跳转的最大长度。换句话说，如果你在 nums[i] 处，你可以跳转到任意 nums[i + j] 处:0 <= j <= nums[i] i + j < n
  返回到达 nums[n - 1] 的最小跳跃次数。生成的测试用例可以到达 nums[n - 1]。
  > 记录最远的距离，以及达到最远距离的次数。
* 划分字母区间：给你一个字符串 s 。我们要把这个字符串划分为尽可能多的片段，同一字母最多出现在一个片段中。
  > 先记录每个字母在s中的最远位置
* 组合总和：给你一个 无重复元素 的整数数组 candidates 和一个目标整数 target ，找出 candidates 中可以使数字和为目标数 target 的 所有 不同组合 ，并以列表形式返回。你   可以按 任意顺序 返回这些组合。
  candidates 中的 同一个 数字可以 无限制重复被选取 。如果至少一个数字的被选数量不同，则两种组合是不同的。 
  对于给定的输入，保证和为 target 的不同组合数少于 150 个。
* 括号生成：数字 n 代表生成括号的对数，请你设计一个函数，用于能够生成所有可能的并且 有效的 括号组合。
  
###### 数组
* 最长回文子串：给你一个字符串 s，找到 s 中最长的回文子串。如果字符串的反序与原始字符串相同，则该字符串称为回文字符串。
  > temp == temp[::-1], start = max(i-len(res)-1,0) 要考察len(res)+2的情况，因为下一个更长的回文串可能是+2的。
* 只出现一次的数字：给你一个 非空 整数数组 nums ，除了某个元素只出现一次以外，其余每个元素均出现两次。找出那个只出现了一次的元素。
  你必须设计并实现线性时间复杂度的算法来解决此问题，且该算法只使用常量额外空间。
* 多数元素：给定一个大小为 n 的数组 nums ，返回其中的多数元素。多数元素是指在数组中出现次数 大于 ⌊ n/2 ⌋ 的元素。
  你可以假设数组是非空的，并且给定的数组总是存在多数元素。（摩尔投票）
* 颜色分类：给定一个包含红色、白色和蓝色、共 n 个元素的数组 nums ，原地对它们进行排序，使得相同颜色的元素相邻，并按照红色、白色、蓝色顺序排列。
  我们使用整数 0、 1 和 2 分别表示红色、白色和蓝色。必须在不使用库内置的 sort 函数的情况下解决这个问题。
* 下一个排列：整数数组的一个 排列  就是将其所有成员以序列或线性顺序排列。
  整数数组的 下一个排列 是指其整数的下一个字典序更大的排列。更正式地，如果数组的所有排列根据其字典顺序从小到大排列在一个容器中，
  那么数组的 下一个排列 就是在这个有序容器中排在它后面的那个排列。如果不存在下一个更大的排列，那么这个数组必须重排为字典序最小的排列（即，其元素按升序排列）。
  必须 原地 修改，只允许使用额外常数空间。
* 合并区间：以数组 intervals 表示若干个区间的集合，其中单个区间为 intervals[i] = [starti, endi] 。请你合并所有重叠的区间，并返回 一个不重叠的区间数组，该数组需恰好覆盖输入中的所有区间 。
  > intervals.sort(key=lambda x:x[0])
  ```python
  def merge(self, intervals: List[List[int]]) -> List[List[int]]:
      intervals.sort(key=lambda x: x[0])
      res=[]
      for interval in intervals:
          if res and res[-1][1]>=interval[0]:
              res[-1][1]=max(res[-1][1], interval[1])
          else:
              res.append(interval)
      return res
  ```
* 轮转数组：给定一个整数数组 nums，将数组中的元素向右轮转 k 个位置，其中 k 是非负数。
  > nums[:] = nums[::-1]是原地对列表进行反转，nums[:] 表示选取原列表的全部元素；而nums = nums[::-1]是使nums指向一个新的列表对象。
* 除自身以外数组的乘积：给你一个整数数组 nums，返回 数组 answer ，其中 answer[i] 等于 nums 中除 nums[i] 之外其余各元素的乘积 。  
  题目数据 保证 数组 nums之中任意元素的全部前缀元素和后缀的乘积都在  32 位 整数范围内。  
  请 不要使用除法，且在 O(n) 时间复杂度内完成此题。
  ```python
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        n=len(nums)
        res=[1]*n
        for i in range(1,n):
            res[i]=res[i-1]*nums[i-1]
        R=1
        for i in range(n-1,-1,-1):
            res[i]*=R
            R*=nums[i]
        return res
  ```
  > 左右乘
* 缺失的第一个正数：给你一个未排序的整数数组 nums ，请你找出其中没有出现的最小的正整数。  
  请你实现时间复杂度为 O(n) 并且只使用常数级别额外空间的解决方案。
  > 最小的正整数一定<=len(nums)+1, 只需要将<=len(nums)的数对应位置数变为负数即可。  
  > 如果不要求常数级别的额外空间只需要循环即可。

###### 子串
* 和为 K 的子数组：给你一个整数数组 nums 和一个整数 k ，请你统计并返回 该数组中和为 k 的子数组的个数 。  
  子数组是数组中元素的连续非空序列。
  > 前缀和

###### 滑动窗口
* 找到字符串中所有字母异位词：给定两个字符串 s 和 p，找到 s 中所有 p 的 异位词 的子串，返回这些子串的起始索引。不考虑答案输出的顺序。
  异位词 指由相同字母重排列形成的字符串（包括相同的字符串）。
  ```python
  for i in range(m):
      curr[ord(p[i])-ord('a')]+=1
      curr[ord(s[i])-ord('a')]-=1
  res=[]
  if curr==[0]*26: res.append(0)
  for i in range(n-m):
      curr[ord(s[i])-ord('a')]+=1
      curr[ord(s[i+m])-ord('a')]-=1 
      if curr==[0]*26: res.append(i+1)
  return res
  ```
* 无重复字符的最长子串：给定一个字符串 s ，请你找出其中不含有重复字符的 最长子串的长度。
  > 双指针
  ```python
  for i in range(n):
    while j<n and s[j] not in curr:
        curr.add(s[j])
    res=max(res, j-i)
    curr.remove(s[i])
  ```
* 滑动窗口最大值：给你一个整数数组 nums，有一个大小为 k 的滑动窗口从数组的最左侧移动到数组的最右侧。你只可以看到在滑动窗口内的 k 个数字。滑动窗口每次只向右移动一位。
  返回 滑动窗口中的最大值 。
  > 两种做法：一种是用最大堆来排序q = [(-nums[i], i) for i in range(k)], heapq.heapify(q);
  > 另一种是用队列存储：  
  ```python
  res.append(queue[0])
  for i in range(k, n):
      if queue[0]==nums[i-k]: queue.popleft()
      while queue and queue[-1]<nums[i]:
          queue.pop()
      queue.append(nums[i])
      res.append(queue[0])
  return res
  ```
* 最小覆盖子串：给你一个字符串 s 、一个字符串 t 。返回 s 中涵盖 t 所有字符的最小子串。如果 s 中不存在涵盖 t 所有字符的子串，则返回空字符串 "" 。  
  注意：对于 t 中重复字符，我们寻找的子字符串中该字符数量必须不少于 t 中该字符数量。如果 s 中存在这样的子串，我们保证它是唯一的答案。
  > 用needMap和needSum来追踪实时的字符变化，用双指针来检索。
  ```python
    for l in t:
        curr[l]+=1
    i,j=0,0
    while j<n:
        if s[j] in curr:
            if curr[s[j]]>0: neednum-=1
            curr[s[j]]-=1
        while neednum==0:
            if not res or len(res)>j-i+1: res=s[i:j+1]
            if s[i] in curr:
                if curr[s[i]]==0: neednum+=1
                curr[s[i]]+=1
            i+=1
        j+=1
    return res
  ```

###### 双指针
* 接雨水：给定 n 个非负整数表示每个宽度为 1 的柱子的高度图，计算按此排列的柱子，下雨之后能接多少雨水。
  ```python
  while i<j:
    if height[i]<height[j]:
        leftmax=max(leftmax,height[i])
        res+=leftmax-height[i]
        i+=1
    else:
        rightmax=max(rightmax,height[j])
        res+=rightmax-height[j]
        j-=1
  ```
* 三数之和：给你一个整数数组 nums ，判断是否存在三元组 [nums[i], nums[j], nums[k]] 满足 i != j、i != k 且 j != k ，同时还满足 nums[i] + nums[j] + nums[k] == 0 。请你返回所有和为 0 且不重复的三元组。
  ```python
  for i in range(n):
    l,r = i+1,n-1
    if i and nums[i]==nums[i-1]: continue
  ```

###### 矩阵
* 搜索二维矩阵 II：编写一个高效的算法来搜索 m x n 矩阵 matrix 中的一个目标值 target 。该矩阵具有以下特性：  
  每行的元素从左到右升序排列。  
  每列的元素从上到下升序排列。
  > 从右上角开始搜索
* 矩阵置零：给定一个 m x n 的矩阵，如果一个元素为 0 ，则将其所在行和列的所有元素都设为 0 。请使用 原地 算法。
* 螺旋矩阵：给你一个 m 行 n 列的矩阵 matrix ，请按照 顺时针螺旋顺序 ，返回矩阵中的所有元素。
  > 两个判定条件：while left<=right and top<=bottom: if left<right and top<bottom:
  ```python
  def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
      l,r,u,d=0,len(matrix[0])-1,0,len(matrix)-1
      res=[]
      while l<=r and u<=d:
          for i in range(l,r+1):
              res.append(matrix[u][i])
          for j in range(u+1,d+1):
              res.append(matrix[j][r])
          if l<r and u<d:
              for i in range(r-1,l-1,-1):
                  res.append(matrix[d][i])
              for j in range(d-1,u,-1):
                  res.append(matrix[j][l])
          l+=1
          r-=1
          u+=1
          d-=1
      return res
  ```
* 旋转图像：给定一个 n × n 的二维矩阵 matrix 表示一个图像。请你将图像顺时针旋转 90 度。
  你必须在 原地 旋转图像，这意味着你需要直接修改输入的二维矩阵。请不要 使用另一个矩阵来旋转图像。
  > 转置+水平翻转
  ```python
  def rotate(self, matrix: List[List[int]]) -> None:
      """
      Do not return anything, modify matrix in-place instead.
      """
      n=len(matrix)
      for i in range(n):
          for j in range(i+1,n):
              matrix[i][j],matrix[j][i]=matrix[j][i],matrix[i][j]
      
      for i in range(n):
          for j in range(n//2):
              matrix[i][j],matrix[i][n-j-1]=matrix[i][n-j-1],matrix[i][j]

  ```
###### 堆
* 数组中的第K个最大元素：给定整数数组 nums 和整数 k，请返回数组中第 k 个最大的元素。
  请注意，你需要找的是数组排序后的第 k 个最大的元素，而不是第 k 个不同的元素。
  你必须设计并实现时间复杂度为 O(n) 的算法解决此问题。
  > (快速选择，pivot)
* 前K个高频元素：给你一个整数数组 nums 和一个整数 k ，请你返回其中出现频率前 k 高的元素。你可以按 任意顺序 返回答案。
  > num = collections.Counter(nums), num = [(-n[1], n[0]) for n in num.items()]
* 数据流的中位数：中位数是有序整数列表中的中间值。如果列表的大小是偶数，则没有中间值，中位数是两个中间值的平均值。  
  实现 MedianFinder 类:  
  MedianFinder() 初始化 MedianFinder 对象。  
  void addNum(int num) 将数据流中的整数 num 添加到数据结构中。  
  double findMedian() 返回到目前为止所有元素的中位数。与实际答案相差 10-5 以内的答案将被接受。
  > 使用大顶堆和小顶堆，分别来存储较小的一半数和较大的一半数。

###### 栈
* 字符串解码：给定一个经过编码的字符串，返回它解码后的字符串。
  编码规则为: k[encoded_string]，表示其中方括号内部的 encoded_string 正好重复 k 次。注意 k 保证为正整数。
  你可以认为输入字符串总是有效的；输入字符串中没有额外的空格，且输入的方括号总是符合格式要求的。
  此外，你可以认为原始数据不包含数字，所有的数字只表示重复的次数 k ，例如不会出现像 3a 或 2[4] 的输入。
* 每日温度：给定一个整数数组 temperatures ，表示每天的温度，返回一个数组 answer ，其中 answer[i] 是指对于第 i 天，下一个更高温度出现在几天后。如果气温在这之后都不会升高，请在该位置用 0 来代替。
* 有效的括号：
* 最小栈：设计一个支持 push ，pop ，top 操作，并能在常数时间内检索到最小元素的栈。
  实现 MinStack 类:
  MinStack() 初始化堆栈对象。
  void push(int val) 将元素val推入堆栈。
  void pop() 删除堆栈顶部的元素。
  int top() 获取堆栈顶部的元素。
  int getMin() 获取堆栈中的最小元素。
  > 使用两个栈，一个用于存储数，一个用于存储最小数。
* 柱状图中最大的矩形：给定 n 个非负整数，用来表示柱状图中各个柱子的高度。每个柱子彼此相邻，且宽度为 1 。求在该柱状图中，能够勾勒出来的矩形的最大面积。

###### 图论
* 课程表：你这个学期必须选修 numCourses 门课程，记为 0 到 numCourses - 1 。
  在选修某些课程之前需要一些先修课程。 先修课程按数组 prerequisites 给出，其中 prerequisites[i] = [ai, bi] ，表示如果要学习课程 ai 则 必须 先学习课程  bi 。

  例如，先修课程对 [0, 1] 表示：想要学习课程 0 ，你需要先完成课程 1 。
  请你判断是否可能完成所有课程的学习？如果可以，返回 true ；否则，返回 false 。
  > queue = collections.deque([u for u in range(numCourses) if indegree[u] == 0])
* 岛屿数量：给你一个由 '1'（陆地）和 '0'（水）组成的的二维网格，请你计算网格中岛屿的数量。
  岛屿总是被水包围，并且每座岛屿只能由水平方向和/或竖直方向上相邻的陆地连接形成。
  此外，你可以假设该网格的四条边均被水包围。
  > dfs，访问过的岛屿修改数值为2
* 腐烂的橘子：在给定的 m x n 网格 grid 中，每个单元格可以有以下三个值之一：    
  值 0 代表空单元格；  
  值 1 代表新鲜橘子；  
  值 2 代表腐烂的橘子。  
  每分钟，腐烂的橘子 周围 4 个方向上相邻 的新鲜橘子都会腐烂。  
  返回 直到单元格中没有新鲜橘子为止所必须经过的最小分钟数。如果不可能，返回 -1 。
  > 队列的应用，queue = collections.deque()，将腐烂的橘子坐标和分钟放到队列中。
* 实现 Trie (前缀树)：
  Trie（发音类似 "try"）或者说 前缀树 是一种树形数据结构，用于高效地存储和检索字符串数据集中的键。这一数据结构有相当多的应用情景，例如自动补完和拼写检查。
  请你实现 Trie 类：  
  
  Trie() 初始化前缀树对象。  
  void insert(String word) 向前缀树中插入字符串 word 。  
  boolean search(String word) 如果字符串 word 在前缀树中，返回 true（即，在检索之前已经插入）；否则，返回 false 。  
  boolean startsWith(String prefix) 如果之前已经插入的字符串 word 的前缀之一为 prefix ，返回 true ；否则，返回 false 。
  > 用set来存储单词和前缀

###### 二分查找
* 寻找重复数：给定一个包含 n + 1 个整数的数组 nums ，其数字都在 [1, n] 范围内（包括 1 和 n），可知至少存在一个重复的整数。
  假设 nums 只有 一个重复的整数 ，返回 这个重复的数 。你设计的解决方案必须 不修改 数组 nums 且只用常量级 O(1) 的额外空间。
* 在排序数组中查找元素的第一个和最后一个位置：给你一个按照非递减顺序排列的整数数组 nums，和一个目标值 target。请你找出给定目标值在数组中的开始位置和结束位置。
  如果数组中不存在目标值 target，返回 [-1, -1]。
  你必须设计并实现时间复杂度为 O(log n) 的算法解决此问题。
* 搜索插入位置：给定一个排序数组和一个目标值，在数组中找到目标值，并返回其索引。如果目标值不存在于数组中，返回它将会被按顺序插入的位置。
  请必须使用时间复杂度为 O(log n) 的算法。
* 搜索二维矩阵：给你一个满足下述两条属性的 m x n 整数矩阵：
  每行中的整数从左到右按非严格递增顺序排列。
  每行的第一个整数大于前一行的最后一个整数。
  给你一个整数 target ，如果 target 在矩阵中，返回 true ；否则，返回 false 。
  > 上界与下届的二分查找。上界：if matrix[mid] <= target: return r; 下界：if matrix[mid] < target: return l
* 搜索旋转排序数组：整数数组 nums 按升序排列，数组中的值 互不相同 。
  在传递给函数之前，nums 在预先未知的某个下标 k（0 <= k < nums.length）上进行了 旋转，使数组变为 [nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ...,    nums[k-1]]（下标 从 0 开始 计数）。例如， [0,1,2,4,5,6,7] 在下标 3 处经旋转后可能变为 [4,5,6,7,0,1,2] 。
  给你 旋转后 的数组 nums 和一个整数 target ，如果 nums 中存在这个目标值 target ，则返回它的下标，否则返回 -1 。
  你必须设计一个时间复杂度为 O(log n) 的算法解决此问题。
  > 先判断左右哪段是排序数组
* 寻找两个正序数组的中位数：给定两个大小分别为 m 和 n 的正序（从小到大）数组 nums1 和 nums2。请你找出并返回这两个正序数组的 中位数 。
  算法的时间复杂度应该为 O(log (m+n)) 。
  > 每次处理k//2个数，初始k=(m+n+1)//2,(m+n+2)//2
* 寻找旋转排序数组中的最小值：已知一个长度为 n 的数组，预先按照升序排列，经由 1 到 n 次 旋转 后，得到输入数组。给你一个元素值 互不相同 的数组 nums ，它原来是一个升序排列的数组，并按上述情形进行了多次旋转。请你找出并返回数组中的 最小元素 。

你必须设计一个时间复杂度为 O(log n) 的算法解决此问题。
  > 设计二分查找，找出小于第一个值的元素在list中的index，if nums[mid]>=target: left=mid+1

###### 基础算法结构
* 数组的最小堆化：
  ```python
  class Solution:

    def min_heapify(arr, n, i):
        smallest = i
        l = 2 * i + 1
        r = 2 * i + 2

        if l < n and arr[l] < arr[smallest]:
            smallest = l
        if r < n and arr[r] < arr[smallest]:
            smallest = r

        if smallest != i:
            arr[i], arr[smallest] = arr[smallest], arr[i]
            min_heapify(arr, n, smallest)

    def build_min_heap(arr):
        n = len(arr)
        for i in range(n//2 - 1, -1, -1):
            min_heapify(arr, n, i)  
  ```

* oc的输入输出
  ```python
  T = int(input()) 
  n, a, b = map(int, input().strip().split())
  data = list(map(int, input(),strip().split()))
  a,b=b,a 
  ```

###### 笔试题
* 输入为压缩字符串2(a)3(b)4(c),给出最小分割数，使每个字串的值小于等于k，值的计算公式为 字符数*字符种类
  ```python
  words = ['a', 'b', 'c']
  nums = [2, 3, 4]
  kinds = set()
  count = 0
  ans = 0
  for i, s in enumerate(words):
    kinds.add(s)
    v = (count+num)*len(kinds)
    if v==k:
      ans+=1
      kinds = set()
      count = 0
    elif v<k: count+=num
    # v>k, 
    else:
        need = k-(len(kinds)*(count+1))
        need = ceil(need//len(kinds))
        need = 0 if need <=0 else need
        res = num-need-1
        ans += 1
        ans += res//k
        count = res%k
        if count==0: kinds=set() 
  print(ans)    
  ```

* 在圆内随机生成点 : 圆内等概率随机采样. 逆变换定理
```python
for i in range(10000):
    u1 = random.uniform(0, 1)
    u2 = random.uniform(0, 1)
    a.append(np.sqrt(u1) * math.cos(np.pi * 2 * u2))
    b.append(np.sqrt(u1) * math.sin(np.pi * 2 * u2))
```

* 圆周上任意取三点，组成锐角三角形的概率
  ![image](https://github.com/Feve1986/coding/assets/67903547/203da0b7-0871-429d-b2dc-17cb8d3c71d8)


* 快速排序
```python
def quick_sort(l,i,j):
  if i>=j: return l
  pivot=l[i]
  low=i
  high=j
  while i<j:
    while i<j and l[j]>=pivot:
      j-=1
    l[i]=l[j]
    while i<j and l[i]<=pivot:
      i+=1
    l[j]=l[i]
  l[i]=pivot
  quick_sort(l,low,i-1)
  quick_sort(l,i+1,high)
```

* 小红书笔试题：塔子哥有n个账号，每个账号粉丝数为。

这天他又创建了一个新账号，他希望新账号的粉丝数恰好等于
。为此他可以向自己已有账号的粉丝们推荐自己的新账号，这样以来新账号就得到了之前粉丝的关注。

他想知道，他最少需要在几个旧账号发“推荐新账号”的文章，可以使得他的新账号粉丝数恰好为
，除此以外，他可以最多从中选择一个账号多次发“推荐新账号”的文章。

假设一个旧账号粉丝数为ai，如果仅推荐一次，那么新账号粉丝数增加[ai/2]，如果多以推荐，则粉丝数增加ai。

输入
```
5 8
1 2 3 4 10
```

输出
```
2
```

```python
# 定义定义f[i][j][k]为从前i个旧账号中选择，且粉丝量为j，使用了k次多次推广（注意题目说了最多只能使用一个旧账号做多次推广）的最小选择的旧账号数量
n, x = map(int, input().split())
w = [0] + list(map(int, input().split()))
f=[[float("inf")*2 for _ in range(x+1)] for _ in range(n+1)]
f[0][0][0]=0
for i in range(1, n+1):
  for j in range(x+1):
    for k in range(2):
      f[i][j][k]=min(f[i][j][k], f[i-1][j][k])
      if j>=w[i]//2:
        f[i][j][k]=min(f[i][j][k], f[i][j-w[i]//2][k]+1)
      if j>=w[i] and k>0:
        f[i][j][k]=min(f[i][j][k], f[i][j-w[i]][k-1]+1)
res=min(f[-1][-1][0], f[-1][-1][1])
if res==float("inf"): return -1
else: return res
```

* 塔子哥每次查看他的题解数据，发现都会有一篇题解的赞数+1，并且之后赞数增加的，必是另一篇题解。塔子哥想知道，当某一篇题解赞数最多时，所有题解赞数和的最小值是多少？

输入：
```
3
3 1 4
```

输出
```
9
15
8
```

```python
n=int(input())
a=list(map(int, input().split()))
sum_a=sum(a)
maxa=max(a)
def check(a_i, x):
  return a_i-x<=(n-1)*x-(sum_a-a_i)+1

if n==2:
  for i in range(n):
    if a[i]==maxa:
      print(sum_a)
    else:
      print(-1)
else:
  for i in range(n):
    l,r=a[i],int(1e12)
    while l<=r:
      mid=(l+r)//2
      if check(a[i], mid):
        r=mid-1
      else:
        l=mid+1
    res=sum_a+l-a[i]+max(l-a[i]-1, 0)
    print(res)
```
