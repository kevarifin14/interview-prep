import pdb

"""
Contents
* Sorting Analysis
* List Operations
* Set Operations
* Dictionary Operations
* LRU Cache
* Algorithms
  * Switch Variables
  * Add One Bitwise
  * Decode Ways
  * Two Sum
  * Three Sum
  * Median of a Stream
  * Game of Life
  * Linked List Cycle
  * Unique Paths
  * Reverse Linked List
  * Majority Element
  * Max Subarray
  * Permutations
* Tricks
  * Digital Root
"""

"""
Sorting Analysis
Quicksort - pick pivot, move all elements less than to the left and greater elements
to the right. Recursively select pivots to subarray of smaller and larger elements
* not stable
Mergesort - divide and conquer algorithm where we break down arrays in half and then
recursively conquer
* stable
"""

"""
List operations
Copy: O(n)
Append[1]: O(1)
Insert: O(n)
Get Item: O(1)
Set Item: O(1)
Delete Item: O(n)
Iteration: O(n)
Get Slice: O(k)
Del Slice: O(n)
Set Slice: O(k+n)
Extend[1]: O(k)
Sort: O(n log n)
Multiply: O(nk)
x in s: O(n)
min(s), max(s): O(n)
Get Length: O(1)
"""

"""
Set Operations
x in s: average O(1), amortized O(n)
Union s|t: O(len(s)+len(t))
Intersection s&t: average O(min(len(s), len(t)), amortized O(len(s) * len(t))
replace "min" with "max" if t is not a set
Multiple intersection s1&s2&..&sn: (n-1)*O(l) where l is max(len(s1),..,len(sn))
Difference s-t: O(len(s))
s.difference_update(t): O(len(t))
Symmetric Difference s^t: average O(len(s)), amortized O(len(s) * len(t)):
s.symmetric_difference_update(t): average O(len(t)), amortized O(len(t) * len(s))
"""

"""
Dictionary Operations
Copy[2]: O(n)
Get Item: O(1)
Set Item[1]: O(1)
Delete Item: O(1)
Iteration[2]: O(n)
Contains[key]: O(1)
"""

"""
LRU Cache
- Dictionary for key, value where key is your page number and
value is tuple of (data, pointer to link)
- Implement "least recently used" logic in a doubly linked list
  - if you use an element already in the cache, remove it from the doubly linked
  list and append it to the head
  - if doubly linked list is full, remove from the back
"""

"""
Switch Variables
"""
def temp(a, b):
  a = a + b
  b = a - b
  a = a - b

"""
Add One Bitwise
"""
def addOne(num):
  while num & 1:
    num = num ^ 1
    num <<= 1
  num = num ^ 1
  return num

"""
Decode Ways: DP approach with O(n) runtime and O(n) space
Follow ups:
- O(1) space complexity. Basically we only look at the previous two values so define
two variables one to hold index i-1 and one to hold index i-2
"""
def decodeWays(nums):
  # Check if it's an empty string
  if nums == "": return 0
  # Initialize the DP array.
  # Indices 1 through len(nums) represent num ways to decode up to that index
  ways = [0 for _ in range(len(nums) + 1)]
  # First val is initialized as 1 which is our start value that we will propogate
  ways[0] = 1
  for i in range(1, len(nums) + 1):
    # if the number we add is nonzero, we know we can at least decode it in the same ways as previous index
    if nums[i-1] != 0:
      ways[i] += ways[i-1]
    # if we can decode it in two digits, we also add the number of ways we can decode prev two
    if i != 1 and nums[i-2:i] >= "10" and nums[i-2:i] < "27":
      ways[i] += ways[i-2]
  return ways[len(nums)]

"""
Two Sum: Return indices of two numbers that add up to the target
Naive solution: O(n^2) solution just iterating through each of elements
and checking if they sum up to the target
Runtime: O(n)
Space Complexity: O(n) for dictionary to store saved sums
"""
def twoSum(nums, target):
  # Save precomputed values in values
  values = {}
  for i in range(len(nums)):
    # Constant check in for dictionary
    if nums[i] in values.keys(): return [values[nums[i]], i]
    # Save as key whatever amount is need to reach target, with value as the index
    values[target - nums[i]] = i

"""
Three Sum: Return indices of three numbers that add up to target
Runtime: O(n^2)
"""
def threeSum(nums):
  nums.sort()
  threesomes = []
  # using sorted property we just search through all numbers greater
  for i, num in enumerate(nums):
    # we don't want to check the same number at a different index
    if i > 0 and nums[i] == nums[i - 1]: continue
    start = i + 1
    end = len(nums) - 1
    while start < end:
      total = num + nums[start] + nums[end]
      # number is too small so we search through bigger numbers
      if total < 0: start += 1
      # number is too big so we search through smaller numbers
      elif total > 0: end -= 1
      # match so we go to next set of unique numbers
      elif total == 0:
        threesomes.append([num, start, end])
        # move to the next unique start and end number
        while start < end and nums[start] == nums[start+1]:
          start += 1
        while start < end and nums[end] == nums[end-1]:
          end -= 1
        start += 1
        end -= 1
  return threesomes

"""
Median of a Stream
- The main idea is to use two heaps, a left heap to keep track of elements less
than and a right heap to keep track of elements greater
- Invariant: The two heaps are at most 1 element greater than the other
- When returning the median, we either return the average of the top element of the
heap or take the top element from a heap that is greater in length
Runtime: O(logn) for each insertion and possibly shifting elements from one heap to
another but O(1) time to get median
"""
from heapq import *

class MedianFinder:
    def __init__(self):
        # minHeap holds elements greater than median and maxHeap holds elements less than
        self.minHeap, self.maxHeap = [], []
        self.totalNums = 0

    def addNum(self, num):
        self.totalNums += 1
        if len(self.minHeap) == 0: heappush(self.minHeap, num)
        # elements greater go in the minHeap
        elif num >= self.minHeap[0]: heappush(self.minHeap, num)
        # else max heap (take negative because it's the underlying data structure is a minHeap)
        else: heappush(self.maxHeap, -1 * num)

        # If they differ by more than 2 we need to balance
        if len(self.minHeap) - len(self.maxHeap) == 2:
            heappush(self.maxHeap, -1 * heappop(self.minHeap))
        elif len(self.maxHeap) - len(self.minHeap) == 2:
            heappush(self.minHeap, -1 * heappop(self.maxHeap))

    def findMedian(self):
        if self.totalNums % 2 == 0:
            return (self.minHeap[0] + -1 * self.maxHeap[0]) / 2.0
        else:
            if len(self.maxHeap) > len(self.minHeap): return -self.maxHeap[0]
            else: return self.minHeap[0]

"""
Game of Life
"""
def gameOfLife(self, board):
  if board == [[]]:return

  def adj(i, j):
    coordinates = [(i-1, j+1), (i-1, j), (i-1, j-1), (i, j+1), (i, j-1), (i+1, j+1), (i+1, j), (i+1, j-1)]
    liveNeighbors = 0
    for coord in coordinates:
      x, y = coord
      if x < 0 or y < 0: continue
      if x > len(board) - 1 or y > len(board[0]) - 1: continue
      if ogBoard[x][y] == 1:
        liveNeighbors += 1
    return liveNeighbors

  # create copy of board to maintain original values
  import copy
  ogBoard = copy.deepcopy(board)
  m = len(board)
  n = len(board[0])
  for i in range(m):
    for j in range(n):
      # adj gets the number of live neighbors from adjacent
      liveNeighbors = adj(i, j)
      currCell = ogBoard[i][j]
      # update according to rules
      if currCell == 1 and liveNeighbors < 2: board[i][j] = 0
      elif currCell == 1 and liveNeighbors == 2 or liveNeighbors == 3: board[i][j] = 1
      elif currCell == 1 and liveNeighbors > 3: board[i][j] = 0
      elif currCell == 0 and liveNeighbors == 3: board[i][j] = 1

"""
Linked List Cycle
"""
def hasCycle(self, head):
  slow, fast = head, head
  while fast is not None and fast.next is not None:
      slow = slow.next
      fast = fast.next.next
      if slow == fast: return True
  return False

"""
Unique Paths: Given mxn grid, find number of unique paths where start top left
and can only go down or right.
Follow ups: Given these unique paths and paths sums, can you find the value
of the smallest path sum
"""
def uniquePaths(m, n):
  # initialize mxn array for value 1
  paths = [[1] * n] * m
  # we know anything along top or left has only 1 path so start from position (1, 1)
  for i in range(1, n):
    for j in range(1, m):
      # each position is the sum of the ways you can use to get there
      paths[i][j] = paths[i-1][j] + paths[i][j-1]
  return paths[m-1][n-1]

def minPathSum(grid):
  m, n = len(grid), len(grid[0])
  # Set up saved path sums
  paths = [[0 for _ in range(n)] for _ in range(m)]
  # Initialize first path
  paths[0][0] = grid[0][0]
  # paths where m = 0 or n = 0 are just sum of values along row/column
  for i in range(1, m):
    paths[i][0] = paths[i-1][0] + grid[i][0]
  for j in range(1, n):
    paths[0][j] = paths[0][j-1] + grid[0][j]

  for i in range(1, m):
    for j in range(1, n):
      # recurrence relation
      paths[i][j] = min(paths[i-1][j], paths[i][j-1]) + grid[i][j]
  return paths[m-1][n-1]

"""
Reverse a Linked List: basically at each step, we want to have the node point to
its predecessor and then move on to the next node
Follow ups: do it recursively/iteratively depending which you did first
"""
# iterative O(n) time O(1) space
def reverseList(head):
  if head is None: return head
  next = head.next
  head.next = None
  while next is not None:
      temp = next.next
      next.next = head
      head = next
      next = temp
  return head
# recursive O(n) time but O(n) space
def reverseList(head):
  if head is None or head.next is None: return head
  p = reverseList(head.next)
  # sets the next nodes (in OG list) next to head
  head.next.next = head
  # curr head is none
  head.next = None
  return p

"""
Majority Element: Assuming one exists, return the element that appears more than
n/2 times in the array
Follow ups: Divide and conquer solution - can you do it in O(log n) time
"""
def majorityElement(nums):
  counts = {}
  for num in nums:
    if num in counts:
      counts[num] += 1
    else:
      counts[num] = 1
  maxNum = 0
  maxCount = 0
  for num, count in counts.items():
    if count > maxCount:
      maxNum = num
      maxCount = count
  return maxNum

"""
Max Subarray: Find the maximum contiguous subarray sum
"""
def maxSubArray(nums):
  # max subarray ending at index i
  maxSum = [0 for _ in range(len(nums))]
  maxSum[0] = nums[0]
  for i in range(1, len(nums)):
    # either you add the number to the previous subarray ending there, or just
    # start building up new subarray
    maxSum[i] = max(maxSum[i-1] + nums[i], nums[i])
  return max(maxSum)

"""
Combination Sum: How many ways can you sum elements to reach target?
"""
def combinationSum4(nums, target):
    # dp array of num ways to sum to subproblem target
    t = [0 for _ in range(target+1)]
    # for each target exactly equal to a number, there is at least one way
    for num in nums:
        if num <= target: t[num] = 1
    # for every target, if num is less than target, that it is sum of smaller targets
    for i in range(target+1):
        for num in nums:
            if num <= i: t[i] += t[i - num]
    return t[target]

"""
Permutations - different variations of finding permutations.
"""
"""
Digital Root - when you sum up the digits of a number until you
just get one number
The formula for digital root: if n is 0, return 0. Otherwise return 1 + (n-1) % 9
"""



