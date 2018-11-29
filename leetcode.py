from queue import PriorityQueue, Queue

from typing import (
    List,
    Dict,
    Set,
    Tuple,
    Any
)

"""
Utility Classes
"""

# Definition for a undirected graph node
class UndirectedGraphNode(object):
    def __init__(self, x):
        self.label = x
        self.neighbors = []

# Definition for singly-linked list.
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None

class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

def swap(A, swp1, swp2):
    tmp = A[swp2]
    A[swp2] = A[swp1]
    A[swp1] = tmp

"""
Given a sorted array, count the number of occurances of a number n

Time: O(Logn)
"""
def occurances(A, n):
    def search(l, r):
        if l > r:
            return 0

        if A[l] == A[r] and A[l] == n:
            return r - l + 1

        if n < A[l] or n > A[r]:
            return 0

        m = (l + r) // 2
        left = search(l, m)
        right = search(m + 1, r)

        return left + right

    return dfs(0, len(A) - 1)


class ZigzagIterator(object):

    def __init__(self, v_list):
        self.v = []
        for x in v_list:
            if len(x) > 0:
                self.v.append(x)

        self.rotator = 0
        self.count = len(self.v)

    def next(self):
        self.rotator %= count
        num = self.v[self.rotator].pop(0)

        if len(self.v[self.rotator]) <= 0:
            del self.v[self.rotator]
            count -= 1

        self.rotator += 1
        return num


    def hasNext(self):
        return len(self.v) > 0

"""
582. Kill Process

Time: O(n) + O(n + n) => O(n), n = # of processes
"""
def kill_process(self, pid, ppid, kill):
    graph = {}
    output = []

    def dfs(v):
        output.append(v)
        for w in graph.get(v, []):
            dfs(w)

    for process, parent in zip(pid, ppid):
        graph[parent] = graph.get(parent, [])
        graph[parent].append(process)

    dfs(kill)
    return output

"""
724. Find Pivot Index

Time: O(n)
"""
def pivot_index(nums):
    if len(nums) <= 0:
        return -1

    left = [0 for x in range(len(nums))]
    s = 0
    for j, n in enumerate(nums):
        left[j] = s
        s += n

    s = 0
    j = len(nums) - 1
    while j >= 0:
        if s == left[j]:
            return j

        s += nums[j]
        j -= 1

    return -1


"""
Solve the N Queens Problem And Return All Possible Positions Of Queens

Time: O(N!)
"""
def solve_n_queens(n):
    # Co-ordinates of N-Queens on the grid
    positions = [()] * n
    # Create n x n grid

    output = []

    def solve(row):
        # All Queens have been placed correctly
        if row >= n:
            grid = [['.' for x in range(n)] for x in range(n)]
            for x, y in positions:
                grid[x][y] = 'Q'

            grid = ["".join(x) for x in grid]
            output.append(grid)
            return

        # Check all positions on the current row
        for k in range(0, n):
            valid_pos = True

            for j in range(0, row):
                x, y = positions[j]
                if k == y or y - x == k - row or x + y == k + row:
                    valid_pos = False
                    break

            # Found a valid position to place the nth Queen
            if valid_pos:
                positions[row] = (row, k)
                # Recurse and find position for next Queen
                solve(row + 1)

    solve(0)
    return output


def solve_n_queens_two(n):
    # Co-ordinates of N-Queens on the grid
    positions = [()] * n

    def solve(row):
        # All Queens have been placed correctly
        if row >= n:
            return 1

        total = 0
        # Check all positions on the current row
        for k in range(0, n):
            valid_pos = True

            for j in range(0, row):
                x, y = positions[j]
                if k == y or y - x == k - row or x + y == k + row:
                    valid_pos = False
                    break

            # Found a valid position to place the nth Queen
            if valid_pos:
                positions[row] = (row, k)
                # Recurse and find position for next Queen
                total += solve(row + 1)

        return total

    return solve(0)


"""
282. Expression Add Operators

Time: O(3^n)
"""
def expression_add(num, target):
    output = []

    def dfs(k, prev, cur, expr):
        if k >= len(num) and cur == target:
            output.append(expr)
            return

        for j in range(k, len(num)):
            s = num[k : j + 1]
            num = int(s)

            # Simply append first number if string is empty
            if expr == "":
                dfs(j + 1, cur, cur + num, s)
            else:
                # Add
                dfs(j + 1, cur, cur + num, expr + '+' + s)

                # Subtract
                dfs(j + 1, cur, cur - num, expr + '-' + s)

                # Multiplication
                # Note: Understand logic behind (cur - prev) * num + prev
                dfs(j + 1, prev, (cur - prev) * num + prev, expr + '*' + s)

            if num[k] == '0':
                return

    return dfs(0, 0, 0, "")


"""
Leetcode 133. Clone Graph

Time: O(|V| + |E|)
"""
def clone_graph(node):
    def dfs(v, node_map):
        if not v:
            return None

        # Return vertex if it has already been cloned
        if v in node_map:
            return node_map[v]

        # Store clones of vertices in a hash table
        root = UndirectedGraphNode(v.label)
        node_map[v] = root

        for w in v.neighbors:
            node_map[v].append(dfs(w, node_map))

        return root


"""
Leetcode 22. Generate Parentheses

Given n pairs of parentheses,
write a function to generate all combinations of well-formed parentheses.

Time: O(2 * 2^n)

Notes:

1) Keep track of open and close paranthesis
2) Open paranthesis can't be > than n since the total string len is 2*n
3) # of close paranthesis can't be > # of open paranthesis
"""
def generate_paranthesis(n):
    output = []

    def dfs(close_paran, open_paran, stack):
        # Valid combination found
        if len(stack) == 2 * n and close_paran == open_paran:
            output.append(stack)
            return

        # Don't proceed since paranthesis sequence is bound to be
        # invalid
        if open_paran > n or close_paran > open_paran:
            return

        dfs(close_paran, open_paran + 1, stack + '(')
        dfs(close_paran + 1, open_paran, stack + ')')

    dfs(0, 0, "")


"""
216. Combination Sum III

Find all possible combinations of k numbers that add up to a number n,
given that only numbers from 1 to 9 can be used and each combination should
be a unique set of numbers.

Time: O(n C k) where n == 10
"""
def combination_sum_three(k, n):
    output = []

    def combine(s, stack, target):
        if len(stack) > k:
            return

        if target == 0 and len(stack) == k:
            output.append(list(stack))
            return

        for j in range(s, 10):
            if target - j >= 0:
                stack.append(j)
                combine(j + 1, stack, target - j)
                stack.pop()

    combine(0, [], n)
    return output


"""
Find all combinations that sum up to k (You can repeat numbers)
Input has repeated numbers
"""
def combination_sum_two(candidates, target):
    output = []

    def combine_sum(k, stack, target):
        if target < 0:
            return

        if target == 0:
            output.append(list(stack))
            return

        if k >= len(candidates):
            return

        for j in range(k, len(candidates)):
            # Skip duplicates
            if j > k and candidates[j] == candidates[j - 1]:
                continue

            stack.append(candidates[j])
            # j + 1 because we can't reuse the same element
            combine_sum(j + 1, stack, target - candidates[j])
            stack.pop()

    # Sort so that duplicates group together
    candidates.sort()
    combine_sum(0, [], target)
    return output


"""
Find all combinations that sum up to k (You can repeat numbers)
"""
def combination_sum(candidates, target):
    output = []

    def combine_sum(k, stack, target):
        if target < 0:
            return

        if target == 0:
            output.append(list(stack))
            return

        if k >= len(candidates):
            return

        for j in range(k, len(candidates)):
            stack.append(candidates[j])
            combine_sum(j, stack, target - candidates[j])
            stack.pop()

    combine_sum(0, [], target)
    return output


"""
Generate n Choose k combinations
Time: O(nCk)
"""
def combinations(n, k):
    output = []

    def combine(stack, j):
        if len(stack) >= k:
            output.append(list(stack))
        else:
            for m in range(j, n + 1):
                stack.append(m)
                combine(stack, m + 1)
                stack.pop()

    combine([], 0)
    return output


"""
Time: O(2^n)
"""
def subsets(nums):
    def dfs(stack, output, k):
        output.append(list(stack))

        while k < len(nums):
            stack.append(nums[k])
            dfs(stack, output, k + 1)
            stack.pop()
            k += 1

    output = []
    dfs(nums, [], output, 0)
    return output


"""
Time : O(nLogn) + O(2^n) => O(2^n)
"""
def subsets_two(nums):
    output = []

    def dfs(stack, k):
        output.append(list(stack))
        for j in range(k, len(nums)):
            if j > k and nums[j] == nums[j - 1]:
                continue

            stack.append(nums[j])
            dfs(stack, j + 1)
            stack.pop()

    nums.sort()
    dfs([], 0)
    return output


"""
Heaps Algorithm

Generates N! permutations

Time: O(N!)
"""
def permutations(nums):
    def dfs(k, output):
        if k >= len(nums):
            output.append(list(nums))
        else:
            for j in range(k, len(nums)):
                nums[j], nums[k] = nums[k], nums[j]
                dfs(k + 1, output)
                nums[k], nums[j] = nums[j], nums[k]

    output = []
    dfs(0, output)
    return output


def removeInvalidParentheses(s):
    def is_valid(w):
        count = 0
        for paran in w:
            if paran == '(':
                count += 1
            elif paran == ")":
                count -= 1

            if count < 0:
                return False
        return count == 0

    q = Queue()
    seen = set()
    found = False
    output = []
    q.put(s)

    seen.add(s)

    while not q.empty():
        transform = q.get()

        if is_valid(transform):
            output.append(transform)
            found = True

        if found:
            continue

        for index, ch in enumerate(transform):
            if ch == "(" or ch == ")":
                next_transform = transform[:index] + transform[index + 1:]
                if next_transform not in seen and next_transform[0] == '(' and next_transform[-1] == ')':
                    seen.add(next_transform)
                    q.put(next_transform)

    return output


"""
Given inorder and postorder traversal of a tree, construct the binary tree.

"""
def buildTree(self, inorder, postorder):
    def dfs(inorder, postorder):
        if not inorder:
            return None

        top = postorder.pop()
        root = TreeNode(top)
        at = inorder.index(top)

        root.right = dfs(inorder[at + 1:], postorder)
        root.left = dfs(inorder[:at], postorder)

        return root

    return self.dfs(inorder, postorder)


"""
Permute a string with repeating characters

https://www.youtube.com/watch?v=nYFd7VHKyWQ
"""


def permute_with_repeating_characters(word):
    def _permute(d, count, stack):
        if d >= len(word):
            print("".join(stack))
        else:
            for w in count.keys():
                # If all characters are used then skip
                if count[w] != 0:
                    # Find first character with non zero count
                    count[w] -= 1
                    stack.append(w)

                    _permute(d + 1, count, stack)

                    # Backtrack
                    count[w] += 1
                    stack.pop()

    # Count of all characters in word
    count = {}
    for w in word:
        count[w] = count.get(w, 0) + 1
    _permute(0, count, [])

"""
Two sum problem variation

Find two numbers in both lists that sum up to k
"""


def two_sum_two_array(a1: [int], a2: [int], target: int) -> set:
    difference_map = {}
    pairs = set()

    for num in a1:
        difference_map[target - num] = 1

    for num in a2:
        if num in difference_map:
            pairs.add((num, target - num))

    return pairs

"""
https://leetcode.com/problems/roman-to-integer/description/
"""


def roman_to_integer(roman: str) -> int:
    integer_val = 0

    roman_char_map = {
        "I": 1,
        "V": 5,
        "X": 10,
        "L": 50,
        "C": 100,
        "D": 500,
        "M": 1000
    }

    precedence = list(roman_char_map.keys())

    start = 0
    while start < len(roman) - 1:
        char = roman[start]
        next_char = roman[start + 1]

        # Compare if the character has a lower precedence than the next one
        if precedence.index(char) < precedence.index(next_char):
            # Subtract current character's value if it has lower precedence
            # eg IV -1 + 5
            integer_val -= roman_char_map[char]
        else:
            # Add character's value
            # eg XI 10 + 1
            integer_val += roman_char_map[char]
        start += 1

    integer_val += roman_char_map[roman[start]]

    return integer_val

"""'
https://leetcode.com/problems/best-time-to-buy-and-sell-stock/description/'
"""


def buy_and_sell_stock(a: [int]) -> int:
    def dp() -> int:
        max_profit = float('-inf')
        current_min = float('inf')

        for price in a:
            # Keep track of the smallest price
            current_min = min(current_min, price)
            # Update profit if the profit on the current trade is > global
            # profit
            max_profit = max(max_profit, price - current_min)

        return max_profit

    """
    1. The correct buy/sell pair occurs completely within the first half.
    2. The correct buy/sell pair occurs completely within the second half.
    3. The correct buy/sell pair occurs across both halves - we buy in the
    first half, then sell in the second half.

    For option (3), the way to make the highest profit would be to buy at the
    lowest point in the first half and sell in the greatest point in the second half.
    """
    def divide_and_conquer(left: int, right: int) -> int:
        if right == left:
            return 0

        # Split the list
        mid = int((left + right) / 2)

        # Get max profit on the left side
        left_profit = divide_and_conquer(left, mid)

        # Get max profit on the right side
        right_profit = divide_and_conquer(mid + 1, right)

        # Buy on the min price and sell on max price on the right half
        buy_and_sell_on_sides = max(
            a[mid + 1: right], default=0) - min(a[left: mid + 1], default=0)

        return max(left_profit, max(right_profit, buy_and_sell_on_sides))

"""
https://www.programcreek.com/2014/02/leetcode-best-time-to-buy-and-sell-stock-iii-java/;l.

Time: O(n)
"""


def buy_and_sell_stock_two_transactions(prices: [int]) -> int:
    # We use left[i] to track the maximum profit for transactions before i
    left = []

    # We use right[i] to track the maximum profit for transactions after i
    right = []

    max_profit = float('-inf')
    min_buy_price = float('inf')

    for p in prices:
        min_buy_price = min(min_buy_price, p)
        max_profit = max(max_profit, p - min_buy_price)
        left.append(max_profit)

    max_profit = float('-inf')
    max_sell_price = float('-inf')

    for p in prices[::-1]:
        max_sell_price = max(max_sell_price, p)
        max_profit = max(max_profit, max_sell_price - p)
        right.insert(0, max_profit)

    max_profit = float('-inf')

    # For each day we combine the profit
    # This yields the maximum profit if we buy and sell once before the current
    # day and once after the current day
    for l, r in zip(left, right):
        max_profit = max(max_profit, l + r)

    return max_profit


"""
https://leetcode.com/problems/jump-game/description/
"""


def jump_game(jumps: [int], k: int, mem: [int]) -> bool:
    if k >= len(jumps) - 1:
        return True

    if jumps[k] == 0:
        return False

    # Either take a jump or don't
    return jump_game(jumps, k + 1) or jump_game(jumps, k + jumps[k])

"""
Time: O(n)
Space: O(n)
"""


def jump_game_two(jumps: [int], k: int, mem: [int]):
    if k >= len(jumps) - 1:
        return 0

    if jumps[k] == 0:
        return float('inf')

    if mem[k] != -1:
        return mem[k]

    max_jumps = jumps[k]
    mem[k] = float('inf')

    while max_jumps > 0:
        mem[k] = min(mem[k], jump_game_two(jumps, k + max_jumps, mem) + 1)
        max_jumps -= 1

    return mem[k]

from datastructures import LinkedListNode as LLNode


"""
You are given two non-empty linked lists representing two non-negative integers.
The digits are stored in reverse order and each of their nodes contain a single digit.
Add the two numbers and return it as a linked list.
"""


def add_two_numbers(num1, num2):
    answer = LinkedListNode(-1)
    new_head = answer
    num1_tmp = num1
    num2_tmp = num2
    carry = 0

    while num1_tmp or num2_tmp:
        if num1_tmp:
            carry += num1_tmp.val
            num1_tmp = num1_tmp.next

        if num2_tmp:
            carry += num2_tmp.val
            num2_tmp = num2_tmp.next

        new_head.next = LinkedListNode(int(carry % 10))
        carry = int(carry / 10)
        new_head = new_head.next

    if carry > 0:
        new_head.next = LinkedListNode(carry)

    return answer.next


"""
Merge two sorted linked lists and return it as a new list.
The new list should be made by splicing together the nodes of the first two lists.

Input: 1->2->4, 1->3->4
Output: 1->1->2->3->4->4
"""


def merge_two_sorted_lists(sorted_list1: LLNode, sorted_list2: LLNode) -> LLNode:
    if not sorted_list2:
        return sorted_list1

    if not sorted_list1:
        return sorted_list2

    runner1 = sorted_list1
    runner2 = sorted_list2

    while runner1 and runner2:
        if runner1.val > runner2.val:
            tmp = runner2
            runner2 = runner2.next
            tmp.next = runner1
        else:
            tmp = runner1
            runner1 = runner1.next
            tmp.next = runner2

    if sorted_list1.next == sorted_list2:
        return sorted_list1
    return sorted_list2


"""
http://www.techiedelight.com/longest-palindromic-subsequence-using-dynamic-programming/

TODO: Print the longest palindromic substring

Time: O(n^2)
Space: O(n^2)
"""


def longest_palindromic_subsequence(s: str) -> int:
    def dp(s: str, m: int, n: int, mem: [[int]]) -> int:
        if m > n:
            return 0

        # A single alphabet is a palindrome of size 1
        if m == n:
            return 1

        if mem[m][n] != 0:
            return mem[m][n]

        # First and last character match
        if s[m] == s[n]:
            mem[m][n] = 2 + dp(s, m + 1, n - 1, mem)
        else:
            mem[m][n] = max(dp(s, m + 1, n, mem), dp(s, m, n - 1, mem))

        return mem[m][n]

    mem = [[0 for x in s] for x in s]
    return dp(s, 0, len(s) - 1, mem)


"""
Reverse a Linked List
TODO: Recursive
"""


def reverse_linked_list(linked_list: LLNode) -> LLNode:
    new_head = None
    next_node = None
    current = linked_list

    while current:
        next_node = current.next
        current.next = new_head
        new_head = current
        current = next_node

    return new_head


def product_of_array_except_self(array: List[int]) -> [int]:
    result = [1] * len(array)

    # Calculate the product of all numbers to left of i in first pass
    left_product = 1
    for idx, num in enumerate(array):
        result[idx] = left_product
        left_product *= num

    # Iterate array in reverse order
    # Calculate product of numbers to the right of i
    # Multiply products to left and right of i to get product at position i
    right_product = 1
    for idx, num in reversed(list(enumerate(array))):
        # Multiply right product with left product
        result[idx] *= right_product
        # Increase right product
        right_product *= num

    return result


"""
200. Number of Islands

Notes:
This problem is related to finding the number of connected components in an undirected graph

The intuition here is that once we find a “1” we could initiate a new group.
If we do a DFS from that cell in all 4 directions we can reach all 1’s connected
 to that cell and thus belonging to same group.
"""


def num_islands(grid):
    def _num_islands_dfs(x, y, grid):
        # Initialize new group if we find a cell that is visited or has a 0
        if x < 0 or x >= len(grid) or y < 0 or y >= len(grid[0]) or grid[x][y] != 1:
            return

        # Mark cell as visited
        grid[x][y] = -1

        # Check in all four direction
        _num_islands_dfs(x - 1, y, grid)
        _num_islands_dfs(x, y - 1, grid)
        _num_islands_dfs(x + 1, y, grid)
        _num_islands_dfs(x, y + 1, grid)

    count = 0
    for row, x in enumerate(grid):
        for col, y in enumerate(x):
            # If cell is not visited
            if grid[row][col] == 1:
                count += 1
                _num_islands_dfs(row, col, grid)

    return count


"""
56. Merge Intervals

Given a collection of intervals, merge all overlapping intervals.
"""


def merge_intervals(intervals):
    merged_intervals = []
    sorted_intervals = sorted(intervals, key=lambda x: x[0])

    start = sorted_intervals[0][0]
    end = sorted_intervals[0][1]

    for intv in sorted_intervals[1:]:
        if intv[0] <= end:
            end = max(end, intv[1])
        else:
            merged_intervals.append([start, end])
            start = intv[0]
            end = intv[1]

    merged_intervals.append([start, end])
    return merged_intervals


"""
TEST FOR PALINDROMIC PERMUTATIONS

EPI pg. 212

If the string is of even length, a necessary and sufficient condition
for it to be a palindrome is that each character in the string appear an even
number of times. If the length is odd, all but one character should appear an
even number of times.
"""



def test_palindromic_permutation(word):
    letter_map = {}

    for w in word:
        letter_map[w] = letter_map.get(w, 0) + 1

    is_odd_len = len(word) % 2 != 0
    letter_counts = [letter_map[k] for k in letter_map.keys()]

    odd_count = 0
    for count in letter_counts:
        if count % 2 != 0:
            odd_count += 1

        if not is_odd_len and odd_count > 0:
            return False

        if is_odd_len and odd_count > 1:
            return False

    return True


"""
17. Letter Combinations of a Phone Number

Given a digit string, return all possible letter combinations that the number
could represent.
"""


def phone_number_combinations(digits):
    mapping = {
        '0': "",
        '1': "",
        '2': "abc",
        '3': "def",
        '4': "ghi",
        '5': "jkl",
        '6': "mno",
        '7': "pqrs",
        '8': "tuv",
        '9': "wxyz"
    }

    def _phone_number_combination(k, digits, stack, mapping):
        if k >= len(digits):
            print("".join(stack))
        else:
            for d in mapping[digits[k]]:
                stack.append(d)
                _phone_number_combination(k + 1, digits, stack, mapping)
                stack.pop()

    _phone_number_combination(0, digits, [], mapping)


"""
Given a magic number k, check if numbers in an array can added/subtracted in
any order to give us k

Time: O(n^3)

Is a pseudo polynomial algorithm with memoization
"""


def magic_number(num: int, number_list: List[int]) -> bool:
    def solve(k: int, acc: int, mem: dict, seq) -> bool:
        if acc == num:
            # Print the sequence of numbers of operations that lead to k
            print(seq)

            return True

        if k < 0 and acc != num:
            return False

        # Memoize subproblems
        if (acc, k) in mem:
            return mem[(acc, k)]

        # Knapsack
        add = solve(k - 1, acc + number_list[k],
                    mem, seq + " + " + str(number_list[k]))
        subtract = solve(
            k - 1, acc - number_list[k], mem, seq + " - " + str(number_list[k]))
        skip = solve(k - 1, acc, mem, seq)

        mem[(acc, k)] = add or subtract or skip
        return mem[(acc, k)]

    return solve(len(number_list) - 1, 0, {}, "")


"""
38. Count and Say

The count-and-say sequence is the sequence of integers with the first five terms as following:

1.     1
2.     11
3.     21
4.     1211
5.     111221

1 is read off as "one 1" or 11.
11 is read off as "two 1s" or 21.
21 is read off as "one 2, then one 1" or 1211.
"""


def count_and_say(n: int):
    start = "1"

    print(start)
    for x in range(0, n):
        say = ""
        # Set current character
        start_char = start[0]
        count = 0

        for char in start:
            if char == start_char:
                # Count occurances of character
                count += 1
            else:
                # If a new character is found
                say += str(count) + start_char
                # Change current character
                start_char = char
                count = 1

        if count > 0:
            say += str(count) + start_char

        print(say)
        start = say
        say = ""


"""
67. Add Binary

Given two binary strings, return their sum (also a binary string).
"""


def add_binary(b1: str, b2: str) -> str:
    carry = 0
    result = ""

    b1_ptr = len(b1) - 1
    b2_ptr = len(b2) - 1

    while b1_ptr >= 0 or b2_ptr >= 0:
        if b1_ptr >= 0:
            carry += int(b1[b1_ptr])
            b1_ptr -= 1

        if b2_ptr >= 0:
            carry += int(b2[b2_ptr])
            b2_ptr -= 1

        result = str(carry % 2) + result
        carry //= 2  # Floor / Integer division

    if carry > 0:
        result = str(carry) + result

    return result


"""
139. Word Break

TODO: Print all ways in which word can be broken
"""


def word_break(word: str, word_list: List[str]) -> bool:
    def dp(start: int, mem: Dict[int, bool]) -> bool:
        if start >= len(word):
            return True

        if start in mem:
            return mem[start]

        j = start
        while j <= len(word):
            left_break = word[start:j]

            if left_break in word_list:
                right_break = dp(j, mem)

                if right_break:
                    mem[start] = True
                    return mem[start]
            j += 1

        mem[start] = False
        return mem[start]

    return dp(0, {})

from queue import Queue


"""
127. Word Ladder
"""


def word_ladder(begin: str, end: str, word_list: Set[str]) -> int:
    parent = {}

    def shortest_word_ladder() -> int:
        bfs_queue = Queue()
        visited = set()
        seq_chars = {}  # Dict of characters that can be substitued at begin[i]

        # The only possible characters that can appear at begin[i] are all
        # ith characters in word present in the word_list
        for w in word_list:
            for pos, char in enumerate(w):
                if pos not in seq_chars:
                    seq_chars[pos] = set()
                seq_chars[pos].add(char)

        # Bfs to find the shorted path to a transformed word
        bfs_queue.put((begin, 1))
        parent[begin] = None

        while not bfs_queue.empty():
            front, dist = bfs_queue.get()
            visited.add(front)

            if front == end:
                return dist

            # Construct all possible word transformations
            for pos, char in enumerate(front):
                str_trans = list(front)

                for trans in seq_chars[pos]:
                    str_trans[pos] = trans
                    vertex = "".join(str_trans)

                    if vertex not in visited and vertex in word_list:
                        bfs_queue.put((vertex, dist + 1))
                        # Remember where we came from i.e. parent vertex
                        parent[vertex] = front
        return 0

    """ Print the sequence of transformations """
    def transformation_sequence(k: str):
        if parent[k]:
            transformation_sequence(parent[k])
            print(parent[k])

    shortest = shortest_word_ladder()
    transformation_sequence(end)

    return shortest


"""
283. Move Zeroes

Given an array nums, write a function to move all 0's to the end of it
while maintaining the relative order of the non-zero elements.
"""


def move_zeros(num_list: List[int]):
    zero_index = 0
    while zero_index < len(num_list) and num_list[zero_index] != 0:
        zero_index += 1

    non_zero = zero_index + 1
    while non_zero < len(num_list):
        if num_list[non_zero] != 0:
            swap(num_list, zero_index, non_zero)
            zero_index += 1
        non_zero += 1


"""
79. Word Search

TODO: See if this could be made more efficient
"""


def word_search(word: str, grid: List[List[str]]) -> bool:
    def dfs(m: int, n: int, k: int, visited: List[List[int]]) -> bool:
        if k == len(word) - 1 and grid[m][n] == word[k]:
            return True

        if m < 0 or m >= len(grid) or n < 0 or n >= len(grid[0]):
            return False

        if grid[m][n] != word[k]:
            return False

        if visited[m][n]:
            return False

        visited[m][n] = True

        left = dfs(m, n - 1, k + 1, visited)
        right = dfs(m, n + 1, k + 1, visited)
        up = dfs(m + 1, n, k + 1, visited)
        down = dfs(m - 1, n, k + 1, visited)

        visited[m][n] = False

        return left or right or up or down

    visited = [[False for x in grid[0]] for x in grid]

    for x, row in enumerate(grid):
        for y, val in enumerate(row):
            # Only initialize dfs is word[0] == letter in grid[x, y]
            if val == word[0] and dfs(x, y, 0, visited):
                return True
    return False


""" Partition a list at some pivot k """


def partition(start: int, end: int, numbers: List[int]) -> int:
    # Pick partition pivot
    mid = (start + end) // 2
    swap(numbers, mid, end)

    pivot = end
    left = start
    right = end - 1

    while left <= right:
        while left <= right and numbers[left] <= numbers[pivot]:
            left += 1

        while left <= right and numbers[right] > numbers[pivot]:
            right -= 1

        if left < right:
            swap(numbers, left, right)
            left += 1
            right -= 1

    swap(numbers, left, pivot)
    return left


def quick_sort(numbers: List[int]):
    def _quick_sort(low: int, high: int):
        if low < high:
            pivot = partition(low, high, numbers)
            _quick_sort(low, pivot - 1)
            _quick_sort(pivot + 1, high)

    return _quick_sort(0, len(numbers) - 1)


"""
215. Kth Largest Element in an Array

Quick select algorithm using Quick Sort's partition
Time: Best O(n) Worst O(n^2)
"""


def k_largest_element(numbers: List[int], k: int) -> int:
    low = 0
    high = len(numbers) - 1
    find_index = len(numbers) - k

    while low < high:
        pivot = partition(low, high, numbers)

        if find_index < pivot:
            high = pivot - 1
        elif find_index > pivot:
            low = pivot + 1
        else:
            return numbers[find_index]

    return -1


"""
11. Container With Most Water
"""


def container_with_most_water(A: List[int]) -> int:
    left = 0
    right = len(A) - 1
    max_water = float('-inf')

    while left < right:
        # Area of the square
        water = (right - left) * min(A[left], A[right])
        max_water = max(max_water, water)

        if A[left] < A[right]:
            left += 1
        else:
            right -= 1

    return max_water

"""
155. Min Stack

Design a stack that supports push, pop, top, and retrieving the minimum
element in constant time.
"""


class MinStack(object):

    def __init__(self):
        self.stack = []
        self.min_stack = []
        self.current_min = float('inf')

    def push(self, val):
        self.stack.append(val)
        self.current_min = min(self.current_min, val)
        self.min_stack.append(self.current_min)

    def top(self):
        return self.stack[-1]

    def pop(self):
        val = self.stack.pop()
        self.min_stack.pop()
        return val

    def min(self):
        return self.min_stack[-1]


import heapq

"""
K Largest Elements In a Continious Stream Of Numbers

Space: O(k)
Time: O(n Log(k))
"""


def top_k_largest_elements(stream: List[int], k: int) -> List[int]:
    min_heap = []

    # Insert first k elements into a min heap
    start = 0
    while start < k:
        min_heap.append(stream[start])
        start += 1

    heapq.heapify(min_heap)

    # Iterate through remaining elements in the stream
    # If the element is greater than current min in heap, remove element
    # from heap and push current element
    while start < len(stream):
        top = heapq.heappop(min_heap)
        if stream[start] > top:
            heapq.heappush(min_heap, stream[start])
        else:
            heapq.heappush(min_heap, top)

        start += 1

    return min_heap


"""
268. Missing Number

Given an array containing n distinct numbers taken from 0, 1, 2, ..., n,
find the one that is missing from the array.
"""


def missing_number(nums: List[int]) -> int:
    nums_sum = sum(nums)
    n = len(nums)
    expected_sum = ((n) * (n + 1)) // 2

    return expected_sum - nums_sum


"""
16. 3Sum Closest
"""


def three_sum_closest(nums: List[int], k: int) -> int:
    diff = float('inf')
    answer = 0
    nums.sort()

    left = 0
    while left < len(nums):
        mid = left + 1
        right = len(nums) - 1

        while mid < right:
            three_sum = nums[left] + nums[right] + nums[mid]
            current_diff = abs(three_sum - k)

            # Minimumze abs difference between k and sum
            if current_diff < diff:
                diff = current_diff
                answer = three_sum

                # If we have found 3 numbers that sum to k
                # That's the closest we can get
                if three_sum == k:
                    return answer

            if three_sum > k:
                right -= 1
            else:
                mid += 1

        left += 1

    return answer


"""
287. Find the Duplicate Number

Given an array nums containing n + 1 integers where each integer is between
1 and n (inclusive), prove that at least one duplicate number must exist.
Assume that there is only one duplicate number, find the duplicate one.
"""


def find_duplicate_number(nums: List[int]) -> int:
    fast = nums[0]
    slow = nums[0]

    # Use tortoise and hare cycle detection algorithm
    while True:
        fast = nums[nums[fast]]
        slow = nums[slow]

        if fast == slow:
            break

    # Find start of the cycle
    slow = nums[0]
    while slow != fast:
        slow = nums[slow]
        fast = nums[fast]

    return slow

"""
62. Unique Paths

Time: O(m*n)
"""


def unique_paths(m: int, n: int) -> int:
    def dp(i: int, j: int, mem: Dict[Tuple[int, int], int]) -> int:
        if i >= m or j >= n:
            return 0

        if i == m - 1 and j == n - 1:
            return 1

        if (i, j) in mem:
            return mem[(i, j)]

        right = dp(i, j + 1, mem)
        left = dp(i + 1, j, mem)

        mem[(i, j)] = right + left
        return mem[(i, j)]

    return dp(0, 0, {})


def minimum_path_sum(grid: List[List[int]]):
    def dp(m: int, n: int, mem: Dict[int, int]):
        if m >= len(grid) or n >= len(grid[0]):
            return float('inf')

        if (m, n) in mem:
            return mem[(m, n)]

        if m == len(grid) - 1 and n == len(grid[0]) - 1:
            return grid[m][n]

        right = dp(m, n + 1, mem)
        down = dp(m + 1, n, mem)

        mem[(m, n)] = min(right, down) + grid[m][n]
        return mem[(m, n)]

    return dp(0, 0, {})

"""
131. Palindrome Partitioning
"""


def palindrome_partition(word: str):
    def is_palindrome(string: str) -> bool:
        # Check for empty string
        if not string:
            return False

        result = True
        str_len = len(string)

        for i in range(0, int(str_len / 2)):
            if string[i] != string[str_len - i - 1]:
                result = False
                break

        return result

    def dp(start: int, stack: List[str]):
        if start >= len(word):
            print(stack)
        else:
            j = start + 1
            while j <= len(word):
                substr = word[start : j]

                if is_palindrome(substr):
                    stack.append(substr)
                    dp(j, stack)
                    stack.pop()
                j += 1

    """
    132. Palindrome Partitioning II
    """
    def min_palindrome(i: int, j: int, palindrome_mem, cuts_mem) -> int:
        # Cache to store palindromes
        if (i, j) in palindrome_mem:
            return palindrome_mem[(i, j)]

        if i == j or is_palindrome(word[i : j]):
            return 0

        # Cache to store minimum number of cuts
        if (i, j) in cuts_mem:
            return cuts_mem[(i, j)]

        min_cuts = float('inf')
        k = i + 1
        while k < j:
            left_cut = min_palindrome(i, k, palindrome_mem, cuts_mem)
            right_cut = min_palindrome(k, j, palindrome_mem, cuts_mem)
            min_cuts = min(min_cuts, left_cut + right_cut + 1)

            k += 1

        cuts_mem[(i, j)] = min_cuts
        return min_cuts

    print(min_palindrome(0, len(word), {}, {}))


"""
Lintcode: (92) Backpack

https://algorithm.yuanbin.me/zh-hans/dynamic_programming/backpack.html
"""


def backpack(items: List[int], max_size: int) -> int:
    def dp(k: int, size: int, mem) -> int:
        if k >= len(items) or size > max_size:
            return 0

        if (k, size) in mem:
            return mem[(k, size)]

        # Current item is larger than the knapsack
        left_space = max_size - size
        if items[k] > left_space:
            return dp(k + 1, size, mem)

        take = dp(k + 1, size + items[k], mem) + items[k]
        skip = dp(k + 1, size, mem)

        mem[(k, size)] = max(take, skip)
        return mem[(k, size)]

    return dp(0, 0, {})


"""
Lintcode: (125) Backpack II

https://algorithm.yuanbin.me/zh-hans/dynamic_programming/backpack_ii.html
"""


def backpack_two(items: List[int], value: List[int], max_size: int) -> int:
    def dp(k: int, size: int) -> int:
        if k < 0 or size < 0:
            return 0

        if items[k] > size:
            return dp(k - 1, size)

        take = dp(k - 1, size - items[k]) + value[k]
        skip = dp(k - 1, size)

        return max(take, skip)

    return dp(len(items) - 1, max_size)



"""
97. Interleaving String
"""


def interleave_words(s1: str, s2: str, target: str) -> bool:
    def solve(i: int, j: int, k: int, mem) -> bool:
        if (i, j, k) in mem:
            return mem[(i, j, k)]

        mem[(i, j, k)] = False

        if i >= len(s1) and j >= len(s2) and k >= len(target):
            return True

        if k >= len(target):
            return False

        if i < len(s1) and j < len(s2) and s1[i] == target[k] and s2[j] == target[k]:
            mem[(i, j, k)] = solve(i + 1, j, k + 1, mem) or solve(i, j + 1, k + 1, mem)
        elif i < len(s1) and s1[i] == target[k]:
            mem[(i, j, k)] = solve(i + 1, j, k + 1, mem)
        elif j < len(s2) and s2[j] == target[k]:
            mem[(i, j, k)] = solve(i, j + 1, k + 1, mem)

        return mem[(i, j, k)]

    return solve(0, 0, 0, {})


"""
341. Flatten Nested
"""


def flatten_list(nested_list: List[Any]) -> List[int]:
    def solve(nest: List[Any], flat_list: List[int]):
        if not isinstance(nest, list):
            flat_list.append(nest)
        else:
            for k in nest:
                solve(k, flat_list)

    flat_list = []
    solve(nested_list, flat_list)

    return flat_list


def find_longest_consecutive_path(grid: List[List[int]]) -> int:
    def dp(i: int, j: int, prev: int, visited, mem) -> int:
        if i < 0 or j < 0 or i >= len(grid) or j >= len(grid[0]):
            return 0

        if grid[i][j] != prev + 1:
            return 0

        if (i, j) in mem:
            return mem[(i, j)]

        visited[(i, j)] = True

        left = dp(i, j - 1, grid[i][j], visited, mem)
        right = dp(i, j + 1, grid[i][j], visited, mem)
        top = dp(i - 1, j, grid[i][j], visited, mem)
        down = dp(i + 1, j, grid[i][j], visited, mem)

        visited[(i, j)] = False

        mem[(i, j)] = max(max(left, right), max(top, down)) + 1
        return mem[(i, j)]

    overall_max_path = float('-inf')
    visited = {}
    mem = {}
    for x, row in enumerate(grid):
        for y, val in enumerate(row):
            overall_max_path = max(overall_max_path, dp(x, y, val - 1, visited, mem))

    return overall_max_path


def longest_increasing_subsequence(nums: List[int]) -> int:
    """
    Method 1
    Time: O(n^2)

    Let dp(i) be the lcs ending at element at index i i.e. nums[i]
    For all elements k such that k < i and nums[i] > nums[k]
        Get the max lcs ending at k and add 1 to extend
    """
    def dp(end: int, mem: Dict[int, int]):
        if end in mem:
            return mem[end]

        k = 0
        max_pred = 0
        while k < end:
            if nums[k] < nums[end]:
                max_pred = max(dp(k, mem), max_pred)
            k += 1

        mem[end] = max_pred + 1
        return mem[end]

    """
    Method 2: Knapsack
    Time: O(n^2)

    1) Include element in lis if current element is greater than prev element
    2) Exclude current element
    """
    def knapsack(k: int, prev: int, mem):
        if k >= len(nums):
            return 0

        if (k, prev) in mem:
            return mem[(k, prev)]

        add = 0
        if nums[k] > prev:
            add = knapsack(k + 1, nums[k], mem) + 1

        dont_add = knapsack(k + 1, prev, mem)
        mem[(k, prev)] = max(add, dont_add)

        return mem[(k, prev)]


def increasing_subsequence_max_sum(nums: List[int]) -> int:
    def dp(k: int, prev: int) -> int:
        if k >= len(nums):
            return 0

        dont_add = dp(k + 1, prev)

        add = 0
        if nums[k] > prev:
            add = dp(k + 1, nums[k]) + nums[k]

        return max(add, dont_add)

    return dp(0, float('-inf'))

"""
Two player game

Consider a row of n coins of values v1 . . . vn, where n is even. We play a
game against an opponent by alternating turns. In each turn, a player selects
either the first or last coin from the row, removes it from the row permanently
and receives the value of the coin. Determine the maximum possible amount of
money we can definitely win if we move first.

Notes:
Sample Input: 5, 3, 7, 10

There are two choices:

1) If user1 picks ith coin, user 2 can either pick i + 1th coin or j coin
2) If user1 picks jth coin, user 2 can either pick ith or j - 1th coin

Let Best(i, j) be the maximum value the user can collect

Best(i, j) = Max {
                    Vi + Min{Best(i + 2, j), Best(i + 1, j - 1)} ,
                    Vj + Min{Best(i, j - 2), Best(i + 1, j - 1)}}
                }

Best(i, j) = Vi if j = i // We only have one coin left
Best(i, j) = max(Vi, Vj) if i + 1 = j // Pick max coins if we only have two left
"""

def coins(nums: List[int]):
    def dp(i: int, j: int, coin_count: List[int]):
        if i == j:
            return nums[j]

        if i + 1 == j:
            return max(nums[i], nums[j])

        pick_left = nums[i] + min(dp(i + 2, j), dp(i + 1, j - 1))
        pick_right = nums[j] + min(dp(i, j - 2), dp(i + 1, j - 1))

        return max(pick_right, pick_left)

    coint_count = [0, 0]
    return dp(0, len(nums) - 1)


"""
494. Target Sum

Time: O(2^n)
"""
def target_sum(nums: List[int], target: int) -> int:
    def dp(k: int, total: int) -> int:
        if k < 0:
            if target == total:
                return 1
            return 0

        make_addition = dp(k - 1, total + nums[k])
        make_subtraction = dp(k - 1, total - nums[k])

        return make_subtraction + make_addition

    return dp(len(nums) - 1, 0)


"""
http://www.gohired.in/2017/04/15/minimum-insertions-form-palindrome/
"""
def minimum_insertions_to_palindrome(word: str) -> int:
    def dp(i: int, j: int) -> int:
        if i > j:
            return float('inf')

        # Single character is always a palindrome
        # No insertions since it's already a palindrome
        if i == j:
            return 0

        # Two adjacent words like aa or ab
        if i + 1 == j:
            # If they are equal then it's a palindrome eg aa
            # No insertions
            if word[i + 1] == word[j]:
                return 0
            else:
                return 1 # If they arent equal then one insertion. ab => aba or cb => cbc

        if word[i] == word[j]:
            return dp(i + 1, j - 1)

        return min(dp(i + 1, j), dp(i, j - 1))


"""
403: Frog Jump
"""
def frog_jumps(stones: List[int]) -> bool:
    def jump(k: int, pos: int, jump_pos: Set[int], mem) -> bool:
        # Memoization table
        if (k, pos) in mem:
            return mem[(k, pos)]

        # Stone doesn't exist
        if pos not in jump_pos:
            return False

        # We have reached the last stone
        if pos == stones[-1]:
            return True

        # If we are at the first position
        if k == -1 and pos == 0:
            return jump(1, 1, jump_pos, mem)

        # All the jumps that you can make
        valid_jumps = [k - 1, k, k + 1]
        can_jump = False
        for j in valid_jumps:
            # Only jump forward
            if j > 0:
                can_jump = can_jump or jump(j, pos + j, jump_pos, mem)

        mem[(k, pos)] = can_jump
        return mem[(k, pos)]

    # Put all stone locations into a set for O(1) access
    jump_stones = set(stones)

    return jump(-1, 0, jump_stones, {})


def longest_valid_paranthesis(s):
    def is_valid(s):
        count = 0
        for ch in s:
            if ch == '(':
                count += 1
            else:
                count -= 1

            if count < 0:
                return False

        return count == 0

    def dfs(s, i, j, mem):
        if i == j:
            return 1

        if (i, j) in mem:
            return mem[(i, j)]

        if is_valid(s[i : j]):
            return j - i

        mem[(i, j)] = max(dfs(s, i + 1, j, mem), dfs(s, i, j - 1, mem))
        return mem[(i, j)]

    mem = {}
    return dfs(s, 0, len(s), {})


def string_compression(word):
    if not word:
        return ""

    compressed = []
    ch = word[0]
    count = 1

    for x in word[1:]:
        if x == ch:
            count += 1
        else:
            compressed.append(ch)
            compressed.append(count)
            ch = x
            count = 1

    compressed.append(ch)
    compressed.append(count)

    return compressed


"""
349. Intersection of Two Arrays I
Using two pointers

Time: O(nLogn)
Space: O(1)

What if one array is very larger than the other?
1) Loop shorter array, binary search in longer array
"""
def intersection(nums1, nums2):
    nums1.sort()
    nums2.sort()

    m = 0
    n = 0
    output = set() # Change set() to list() and it solves LC 350
    while n < len(nums1) and m < len(nums2):
        if nums1[n] == nums2[m]:
            output.add(nums1[n])
            n += 1
            m += 1
        elif nums1[n] < nums2[m]:
            n += 1
        else:
            m += 1

    return list(output)


"""
350. Intersection of Two Arrays II
Time: O(n)
Space: Theta(m) => m is the number of unique elements
"""
def intersection_two(nums1, nums2):
    nums1_freq = {}
    for n in nums1:
        nums1_freq[n] = nums1_freq.get(n, 0) + 1

    output = []
    for n in nums2:
        if n in nums1_freq and nums1_freq[n] > 0:
            nums1_freq[n] -= 1
            output.append(n)
    return output


"""
Leetcode 621 - Task Scheduler

Notes:

This is a greedy algorithm. We pick the most frequent task one after the other.
If we don't have more tasks to pick in a round, then there is a pause.

To work on the same task again, CPU has to wait for time n,  therefore we can think of as if there is a cycle, of time n + 1.

1) To avoid leave the CPU with limited choice of tasks and having to sit there cooling down frequently at the end, it is critical the keep the diversity of the task pool for as long as possible.
2) In order to do that, we should try to schedule the CPU to always try round robin between the most popular tasks at any time.
"""
def task_schedule(tasks, n):
    heap = PriorityQueue()
    freq = {}
    time = 0

    # Count frequency of tasks
    for task in tasks:
        freq[task] = freq.get(task, 0) + 1

    for k, v in freq.items():
        heap.put(-v) # Negate value to make it a max-heap

    # Use heap to get tasks with frequency high -> low
    while not heap.empty():
        items = []
        for k in range(0, n + 1):
            # Check to make sure we have tasks left to run in the current round
            if not heap.empty():
                top = heap.get() # Get task with the highest frequency
                if -top > 1:
                    items.append(- top - 1)

            time += 1
            # No more tasks to finish
            if len(items) <= 0 and heap.empty():
                break

        # Put unfinished tasks back in the heap to run again
        for task in items:
            heap.put(-task)

    return time

"""
Leetcode 554

In this approach, we make use of a HashMap mapmap which is used to store entries in the form: (sum,count).
Here, sum refers to the cumulative sum of the bricks’ widths encountered in the current row, and countcount
refers to the number of times the corresponding sum is obtained. Thus, sum in a way, represents the positions
of the bricks’s boundaries relative to the leftmost boundary – means this position can bypass the current brick in the row.
"""
def least_bricks(wall):
    if len(wall) == 0:
        return 0

    prefix = {}
    max_bricks = 0
    for row in wall:
        s = 0
        # Count prefix sum of all rows
        for width in row[:-1]:
            s += width
            prefix[s] = prefix.get(s, 0) + 1
            max_bricks = max(max_bricks, prefix[s])

    return len(wall) - max_bricks


"""
Greedy

Find largest set of non-overlapping intervals
"""
def interval_scheduling(intervals):
    # Sort by finishing time in ascending order
    intervals.sort(key=lambda x: x[1])
    tasks = set()
    finish_time = -1

    # Only add intervals to the set if they don't overlap
    for start, end in intervals:
        if start >= finish_time:
            tasks.add((start, end))
            finish_time = end

    return len(tasks)


"""
Dutch National Flag Problem

Time: O(n)
"""
def sort_colors(nums):
    left = 0
    mid = 0
    right = len(nums) - 1

    while mid <= right:
        if nums[mid] == 0:
            nums[mid], nums[left] = nums[left], nums[mid]
            mid += 1
            left += 1
        elif nums[mid] == 1:
            mid += 1
        elif nums[mid] == 2:
            nums[mid], nums[right] = nums[right], nums[mid]
            right -= 1

"""
494. Target Sum

Time Complexity:
n recursion level
2 branches on every level

O(2^n)
"""
def target_sum(nums, S):
    def dp(k, total, mem):
        if (k, total) in mem:
            return mem[(k, total)]

        if k >= len(nums) and total == 0:
            return 1

        if k >= len(nums):
            return 0

        add = dp(k + 1, total + nums[k], mem)
        subtract = dp(k + 1, total - nums[k], mem)

        mem[(k, total)] = add + subtract
        return mem[(k, total)]

    return dp(0, S, {})


"""
Seperate unique numbers on one side of array and return count
[1, 1, 2, 2, 3, 3, 4, 4] => [1, 2, 3, 4, x, x, x] => 4

Time: O(n)
Space: O(k), k => Count of unique numbers
"""
def seperate_unique_numbers(nums):
    seen = set(nums)

    index = 0
    for value in seen:
        nums[index] = value
        index += 1

    return len(seen)

"""
Find the lowest common ancestor of a Binary Tree

Time: O(n)
Space: O(n)
"""
def lowest_common_ancestor(self, root, p, q):
    """
    :type root: TreeNode
    :type p: TreeNode
    :type q: TreeNode
    :rtype: TreeNode
    """
    if not root:
        return None

    if root.val == p.val or root.val == q.val:
        return root

    left = lowest_common_ancestor(root.left, p, q)
    right = lowest_common_ancestor(root.right, p, q)

    if left is not None and right is not None:
        return root
    elif left is not None:
        return left
    else:
        return right


"""
Merge K Sorted Lists Iterator

Let n be the length of the largest list
Let k be the number of sorted lists

Time: O(nk * Logk)
Space: O(k)

Note: Tuples (n1, n2, n3) are sorted in the order n1 => n2 => n3.
"""
class MergeKSortedIterator(object):
    def __init__(self, lists):
        self.lists = lists
        self.heap = []

        # Put first element from k lists into a heap
        for index, nested_list in enumerate(self.lists):
            self.heap.append((nested_list[0], index, 1))
        heapq.heapify(self.heap)

    def next(self):
        val, list_no, index = heapq.heappop(self.heap)

        # Remove top from Min-Heap,
        # put next element of that list into the heap
        if index < len(self.lists[list_no]):
            item = self.lists[list_no][index]
            heapq.heappush(self.heap, (item, list_no, index + 1))
        return val

    def has_next(self):
        return len(self.heap) != 0


"""
128. Longest Consecutive Sequence

Then go through the numbers. If the number x is the start of a streak (i.e., x-1 is not in the set),
then test y = x+1, x+2, x+3, ... and stop at the first number y not in the set.
The length of the streak is then simply y-x and we update our global best with that.

Time: O(n)

Alternative solution: Use Union Find to form group of numbers that are consecutive
"""
def longest_consecutive_sequence(nums):
    nums = set(nums)
    best = 0

    for x in nums:
        # Walk the sequence if x is the start
        if x - 1 not in nums:
            y = x + 1
            while y in nums:
                y += 1
            best = max(best, y - x)
    return best


"""
Binary Search Tree Iterator
Space: O(h) h => Maximum height of tree
Time: O(1)
"""
class BSTIterator(object):
    def __init__(self, root):
        self.stack = []
        current = root
        while current:
            self.stack.append(current)
            current = current.left

    def has_next(self):
        return len(self.stack) != 0

    def next(self):
        top = self.stack.pop()
        current = top.right
        while current:
            self.stack.append(current)
            current = current.left

        return top.val


def spiral_matrix(matrix: List[List[int]]) -> None:
    row = 0
    col = 0

    last_row = len(matrix) - 1
    last_col = len(matrix[0]) - 1

    while row <= last_row and col <= last_col:
        # Traverse first row
        k = col
        while k <= last_col:
            print(matrix[row][k])
            k += 1
        row += 1

        # Traverse last column
        k = row
        # Implicit boundry check in the while loop
        while k <= last_row:
            print(matrix[k][last_col])
            k += 1
        last_col -= 1

        # Traverse last row
        # Check boundries to make sure we don't fall off
        if row <= last_row:
            k = last_col
            while k >= col:
                print(matrix[last_row][k])
                k -= 1
        last_row -= 1

        # Traverse first column
        if col <= last_col:
            k = last_row
            while k >= row:
                print(matrix[k][col])
                k -= 1
            col += 1
