#include <iostream>
#include <vector>
#include <array>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <set>
#include <algorithm>
#include <list>
#include <queue>
#include <stack>
#include <list>
#include <limits>
#include <memory>
#include <algorithm>

#define INF 1000000007

using namespace std;

typedef vector<int> vi;
typedef vector<bool> vb;
typedef vector<string> vs;
typedef vector<char> vc;
typedef vector<pair<int, int>> vii;
typedef pair<int, int> pii;
typedef pair<char, int> pci;

// Definition for a binary tree node.
struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode(int x) : val(x), left(NULL), right(NULL) {}
};

class Node {
public:
    int val;
    Node* left;
    Node* right;

    Node() {}

    Node(int _val, Node* _left, Node* _right) {
        val = _val;
        left = _left;
        right = _right;
    }
};

/**
 * Print the antidiagonals of a matrix
 * Time: O(n^2)
 *
 * @param A
 * @return
 */
vector<vector<int>> diagonal(vector<vector<int>> &A) {
    vector<vector<int>> output;
    int n = A.size();

    // Iterate column to column through the first row excluding the last element
    for (int i = 0; i < n - 1; i++) {
        int x = 0;
        int y = i;
        vector<int> diag;

        while (x >= 0 && y >= 0 && x < n && y < n) {
            diag.push_back(A[x][y]);
            x++;
            y--;
        }

        output.push_back(diag);
    }

    // Iterate through the last column row by row
    for (int i = 0; i < n; i++) {
        int x = i;
        int y = n - 1;
        vector<int> diag;

        while (x >= 0 && y >= 0 && x < n && y < n) {
            diag.push_back(A[x][y]);
            x++;
            y--;
        }

        output.push_back(diag);
    }

    return output;
}

/**
 * Convert Binary Search Tree (BST) to Sorted Doubly-Linked List
 * Time: O(n)
 * https://articles.leetcode.com/convert-binary-search-tree-bst-to/
 * @param root
 * @param head
 * @param prev
 */
void treeToDoublyList(Node* root, Node*& head, Node*& prev) {
    if (!root) return;

    treeToDoublyList(root->left, head, prev);

    root->left = prev;
    if (prev) {
        prev->right = root;
    } else {
        head = root;
    }

    auto right = root->right;
    head->left = root;
    root->right = head;
    prev = root;

    treeToDoublyList(right, head, prev);
}

Node* treeToDoublyList(Node* root) {
    Node* prev = NULL;
    Node* head = NULL;

    treeToDoublyList(root, head, prev);
    return head;
}

/**
 * 938. Range Sum of BST
 * Time: O(n)
 *
 * @param root
 * @param L
 * @param R
 * @return
 */
int rangeSumBST(TreeNode* root, int L, int R) {
    if (!root) return 0;
    int sum = 0;

    // Add to the sum if current node is in range
    if (root->val >= L && root->val <= R)
        sum += root->val;

    // Explore left subtree if it has values in [L, R]
    if (L < root->val)
        sum += rangeSumBST(root->left, L, R);

    if (R > root->val)
        sum += rangeSumBST(root->right, L, R);

    return sum;
}

/**
 * 91. Decode Ways
 * https://www.youtube.com/watch?v=qli-JCrSwuk
 * Time: O(n)
 *
 * @param s
 * @param k
 * @param mem
 * @return
 */
int numDecodings(string s, int k, vector<int> mem) {
    if (s[k] == '0') return 0;
    if (k >= s.size()) return 1;

    if (mem[k] != -1) return mem[k];
    int decode = numDecodings(s, k + 1, mem);

    if (k + 2 <= s.size() && stoi(s.substr(k, 2)) <= 26) {
        decode += numDecodings(s, k + 2, mem);
    }

    mem[k] = decode;
    return mem[k];
}

int numDecodings(string s) {
    vector<int> mem(s.size(), -1);
    return numDecodings(s, 0, mem);
}

/**
 * Leetcode 896. Monotonic Array
 *
 * Time: O(n)
 * @param A
 * @return
 */
bool isMonotonic(vector<int>& A) {
    bool increasing = true;
    bool decreasing = true;

    for (int i = 0; i < A.size() - 1; i++) {
        if (A[i] > A[i + 1]) {
            increasing = false;
        } else if (A[i] < A[i + 1]) {
            decreasing = false;
        }
    }

    return increasing || decreasing;
}

/**
 * Leetcode 277. Find the Celebrity
 * Time: O(2n)
 *
 * @param n
 * @return
 */
bool knows(int a, int b) { return 0; }
int findCelebrity(int n) {
    // Assume first person to be the celeb
    int cand = 0;
    for (int i = 1; i < n; i++) {
        // If cand knows i then cand can't be the celeb but i might be the celeb
        if (knows(cand, i)) {
            cand = i;
        }
    }

    // This loop checks if cand is an actual celeb
    for (int i = 0; i < n; i++) {
        // If cand knows someone or someone knows the candidate then cand can't be the celeb
        if ((cand != i) && (knows(cand, i) || !knows(i, cand))) {
            return -1;
        }
    }

    return cand;
}

/**
 * 31. Next Permutation
 *https://leetcode.com/problems/next-permutation/discuss/13994/Readable-code-without-confusing-ij-and-with-explanation
 *
 * @param nums
 */
void nextPermutation(vector<int>& nums) {
    if (nums.size() <= 1) return;
    auto n = nums.size();

    // Find first number starting from the least significant number st nums[i]  < nums[i + 1]
    int k = -1;
    for (auto i = n - 1; i > 0; i--) {
        if (nums[i - 1] < nums[i]) {
            k = i - 1;
            break;
        }
    }

    // If next permutation doesn't exist, just sort numbers in ascending order / reverse numbers (They are already
    // in descending order
    if (k < 0) {
        reverse(nums.begin(), nums.end());
        return;
    }

    // Find the rightmost next largest number to the right
    int pos = INT_MAX;
    int abs_diff = INT_MAX;
    for (int i = k + 1; i < n; i++) {
        if (nums[i] > nums[k] && abs(nums[i] - nums[k]) <= abs_diff) {
            abs_diff = abs(nums[i] - nums[k]);
            pos = i;
        }
    }

    // Swap and reverse the remaining numbers.
    // Note that reversing is the same as sorting since the remaining n - k + 1 numbers are in descending order
    swap(nums[k], nums[pos]);
    reverse(nums.begin() + k + 1, nums.end());
}

int countLeaves(TreeNode* root) {
    if (!root) return 0;

    if (!root->right && !root->left) return 1;

    int left = countLeaves(root->left);
    int right = countLeaves(root->right);

    return left + right;
}

int longestSubstring(string s, int k) {
    map<char, int> count;
    int i = 0;
    int j = 0;
    int sum = 0;
    int winLen = 0;

    while (j < s.length()) {
        count[s[j]]++;
        sum += 1;

        while (i <= j && sum >= ((int) count.size() * k)) {
            winLen = max(winLen, j - i + 1);

            count[s[i]]--;
            sum -= 1;

            if (count[s[i]] <= 0) count.erase(s[i]);
            i++;
        }

        j++;
    }

    return winLen;
}

int numSubarrayProductLessThanK(vector<int>& nums, int k) {
    if (k == 0)
        return 0;

    int start, end, total = 0;

    int prod = 1;
    while (end < nums.size()) {
        prod *= nums[end];

        while (start <= end && prod >= k) {
            prod /= nums[start];
            start++;
        }

        total += (end - start + 1);
        end++;
    }

    return total;
}

/**
 * Minimum additions to a stting to make parenthesis valid
 *
 * @param S
 * @return
 */
int minAddToMakeValid(string S) {
    if (!S.length()) return 0;
    int ans = 0;

    stack<char> s;
    for (int j = 0; j < S.length(); j++) {
        // Push open paranthesis
        if (S[j] == '(')
            s.push('(');
        // If a closing paran encounteres, check if an open paran preceeds it.
        // If not, we need to add a paran
        else if (S[j] == ')' && s.empty())
            ans++;
        // If closing paran exists, remove open paran and move on
        else if (S[j] == ')' && !s.empty())
            s.pop();
    }

    ans += s.size();
    return ans;
}

/**
 * Iterative postorder traversal of a binary tree
 * Time: O(n)
 * @param root
 * @return
 */
vector<int> postorderTraversal(TreeNode* root) {
    if (!root)
        return vi();

    vector<int> output;
    stack<TreeNode*> s;
    s.push(root);

    // Traverse tree in preorder but with a twist
    // Preorder is root - left - right but we traverse it in a
    // root - right - left and then reverse for left - right - root to get a post order traversal
    while (!s.empty()) {
        auto top = s.top(); s.pop();
        output.push_back(top->val);

        if (top->left != NULL)
            s.push(top->left);
        if (top->right != NULL)
            s.push(top->right);
    }

    // Reverse preorder traversal of BT to get postorder
    reverse(output.begin(), output.end());

    return output;
}

/*
 * 157. Read N Characters Given Read4
 * Time: O(n) => n characters to be read into the buffer
 */

int read4(char *buf) {
    return 0;
}

int read(char *buf, int n) {
    char buffer[4];
    // Index into buf array
    int index = 0;
    // While characters read are less than n
    while (index < n) {
        // Read upto 4 characters into buffer[]
        int c = read4(buffer);

        // Put read characters into buf[] until either we have reached n
        // or we are out of characters in the buffer
        for (int i = 0; i < c && index < n; i++) {
            buf[index++] = buffer[i];
        }

        // if characters read into buffer are less than 4 means
        // that there are not more characters to read.
        if (c < 4)
            break;
    }

    return index;
}

queue<char> q;
int readTwo(char *buf, int n) {
    // Read characters from Queue into buffer first if Queue is not empty
    int index = 0;
    while (!q.empty() && index < n) {
        buf[index++] = q.front(); q.pop();
    }

    // If n characters have been put from the Queue into buf
    if (index >= n) {
        return index;
    }

    char r4[4];
    // If more characters need to be read into the buf
    while (index < n) {
        // Read 4 characters
        int c = read4(r4);
        int i = 0;

        for (; i < c && index < n; i++) {
            buf[index++] = r4[i];
        }

        // Put remaining read characters into the Queue
        while (i < c) {
            q.push(r4[i++]);
        }

        // Break if no more characters can be read4
        if (c < 4) break;
    }

    return index;
}

class MedianFinder {
public:
    // Lower half of numbers
    priority_queue<int, vector<int>, less<int>> max_heap;
    // Upper half of numbers
    priority_queue<int, vector<int>, greater<int>> min_heap;

    void addNum(int num) {
        if (max_heap.empty() || num < max_heap.top()) {
            max_heap.push(num);
        } else {
            min_heap.push(num);
        }

        if (abs((int) max_heap.size() - (int) min_heap.size()) <= 1) {
            return;
        }

        if (max_heap.size() > min_heap.size()) {
            auto top = max_heap.top(); max_heap.pop();
            min_heap.push(top);
        } else {
            auto top = min_heap.top(); min_heap.pop();
            max_heap.push(top);
        }
    }

    double findMedian() {
        if (max_heap.size() == min_heap.size()) {
            auto left = max_heap.top();
            auto right = min_heap.top();

            return ((double) (left + right)) / 2;
        } else if (max_heap.size() > min_heap.size()) {
            return max_heap.top();
        } else {
            return min_heap.top();
        }
    }
};

bool isPalindrome(string s, int i, int j) {
    while (i < j) {
        if (s[i++] != s[j--])
            return false;
    }

    return true;
}

string minWindow(string s, string t) {
    if (!t.length())
        return "";

    unordered_map<char, int> prefix;

    int i = 0;
    int j = 0;
    int winLeft = 0;
    int winLen = INT_MAX;
    int count = 0;

    for (auto ch : t)
        prefix[ch]++;

    while (j < s.length()) {
        prefix[s[j]]--;

        if (prefix[s[j]] >= 0) count++;

        while (i < j && count == t.length()) {
            if (j - i + 1 < winLen) {
                winLen = j - i + 1;
                winLeft = i;
            }

            prefix[s[i]]++;
            if (prefix[s[i]] > 0) count--;

            i++;
        }

        j++;
    }

    if (winLen > s.length()) return "";

    return s.substr(winLeft, winLen);
}

/**
 * Leetcode 674. Longest Continuous Increasing Subsequence
 *
 * Given an unsorted array of integers, find the length of longest continuous increasing subsequence (subarray).
 * Time: o(n)
 * @param nums
 * @return
 */
int findLengthOfLCIS(vector<int>& nums) {
    if (nums.empty())
        return 0;

    int max_len = 1;
    int start = 0;
    int j = 0;
    for (j = 1; j < nums.size(); j++) {
        if (nums[j] <= nums[j - 1]) {
            max_len = max(max_len, j - start);
            start = j;
        }
    }

    max_len = max(max_len, j - start);
    return max_len;
}

bool isBipartite(vector<vector<int>>& graph, int v, vb& color, vb& seen) {
    seen[v] = true;

    for (auto w : graph[v]) {
        if (!seen[w]) {
            // Assign opposite color of parent to child vertex
            color[w] = !color[v];
            if (!isBipartite(graph, w, color, seen)) return false;
        } else {
            // If parent and child has the same color, then graph can't be biparte
            if (color[w] == color[v])
                return false;
        }
    }

    return true;
}

/**
 * Is Graph Bipartite?
 *
 * A graph is biparte if it is 2 colorable i.e. No two adjacent vertices should have the same color.
 * We run a dfs and try to color adjacent vertices with opposite colors. If we encounter a vertex that has been colored
 * and it has the same color as its parent, then the graph is not biparte
 *
 * @param graph
 * @return
 */
bool isBipartite(vector<vector<int>>& graph) {
    auto n = graph.size();
    vb seen(n, false);
    vb color(n, false);

    for (int i = 0; i < n; i++) {
        if (!seen[i] && !isBipartite(graph, i, color, seen))
            return false;
    }

    return true;
}

bool verifyPreorder(vector<int> &preorder, int start, int end, int lower, int upper) {
    if (start > end)
        return true;

    int val = preorder[start];
    int i = 0;
    // Check if value at root is correct
    if (val <= lower || val >= upper)
        return false;

    // Find pivot point in preorder traversal
    for (i = start + 1; i <= end; ++i) {
        if (preorder[i] >= val)
            break;
    }

    return verifyPreorder(preorder, start + 1, i - 1, lower, val) && verifyPreorder(preorder, i, end, val, upper);
}

/**
 * Leetcode 255. Verify Preorder Sequence in Binary Search Tree
 *
 * Verify if the given preorder traversal forms a valid BST
 * Use divide and conquer and the validate a given BST method
 * @param preorder
 * @return
 */
bool verifyPreorder(vector<int>& preorder) {
    return verifyPreorder(preorder, 0, preorder.size() - 1, -INF, INF);
}

/**
 * Leetcode 3. Longest Substring Without Repeating Characters
 *
 * Time: O(n)
 * @param s
 * @return
 */
int lengthOfLongestSubstring(string s) {
    unordered_set<char> seen;
    int i, j = 0;
    int max_len = 0;

    // Use sliding window and two pointers
    while (j < s.length()) {
        // Remove duplicate characters from the set
        while (i < j && seen.find(s[j]) != seen.end()) {
            seen.erase(s[i]);
            i++;
        }

        // Put character in the set
        seen.insert(s[j]);
        // Check if it's the currently longest substring
        max_len = max(max_len, (int) seen.size());
        j++;
    }

    return max_len;
}

/**
 * Leetcode 159. Longest Substring with At Most Two Distinct Characters
 * Time: O(n)
 * Space: O(1)
 * @param s
 * @return
 */
int lengthOfLongestSubstringTwoDistinct(string s) {
    int max_size = 0;
    unordered_map<char, int> counts;

    int i = 0;
    int j = 0;

    // Using a sliding window with two pointers i and j
    while (j < s.length()) {
        // Add character to the hash table and count
        counts[s[j]]++;

        // If the number of distinct characters seen so far are greater than 2
        // move i until we have have at most 2 distinct characters
        while (i <= j && counts.size() > 2) {
            // Move i and decrement count
            counts[s[i]]--;
            // If count is <= 0, decrement the character count
            if (counts[s[i]] <= 0)
                counts.erase(s[i]);
            i++;
        }

        max_size = max(max_size, j - i + 1);
        j++;
    }

    return max_size;
}

/**
 * Leetcode 53. Maximum Subarray
 *
 * Given an integer array nums, find the contiguous subarray (containing at least one number) w
 * which has the largest sum and return its sum.
 *
 * @param nums
 * @return
 */
int maxSubArray(vector<int>& nums) {
    if (nums.empty()) return 0;

    int sum = 0;
    int max_sum = INT_MIN;
    for (int i = 0; i < nums.size(); i++) {
        sum += nums[i];
        max_sum = max(max_sum, sum);

        if (sum < 0)
            sum = 0;
    }

    return max_sum;
}

void wallsAndGates(vector<vector<int>>& rooms, int m, int n, int dist) {
    if (m < 0 || n < 0 || m >= rooms.size() || n >= rooms[0].size() || rooms[m][n] < dist)
        return;

    rooms[m][n] = dist;
    wallsAndGates(rooms, m, n + 1, dist + 1);
    wallsAndGates(rooms, m, n - 1, dist + 1);
    wallsAndGates(rooms, m + 1, n, dist + 1);
    wallsAndGates(rooms, m + 1, n, dist + 1);
}

/**
 * Leetcode 286. Walls and Gates
 *
 * @param rooms
 */
void wallsAndGates(vector<vector<int>>& rooms) {
    for (int i = 0; i < rooms.size(); i++) {
        for (int j = 0; j < rooms[0].size(); j++) {
            if (rooms[i][j] == 0)
                wallsAndGates(rooms, i, j, 0);
        }
    }
}

/**
 * Longest Increasing Path In A Matrix
 * Time: O(n^2)
 * @param matrix
 * @param dp
 * @param seen
 * @param prev
 * @param m
 * @param n
 * @return
 */
int dfs(vector<vector<int>>& matrix, vector<vi>& dp, vector<vi>& seen, int prev, int m, int n) {
    if (m < 0 || n < 0 || m >= matrix.size() || n >= matrix[0].size() || seen[m][n])
        return 0;

    if (matrix[m][n] < prev)
        return 0;

    if (dp[m][n] != -1)
        return dp[m][n];

    seen[m][n] = 1;
    int top = dfs(matrix, dp, seen, matrix[m][n], m - 1, n);
    int bottom = dfs(matrix, dp, seen, matrix[m][n], m + 1, n);
    int left = dfs(matrix, dp, seen, matrix[m][n], m, n - 1);
    int right = dfs(matrix, dp, seen, matrix[m][n], m, n + 1);
    seen[m][n] = 0;

    dp[m][n] = max(max(top, bottom), max(left, right)) + 1;
    return dp[m][n];
}

int longestIncreasingPath(vector<vector<int>>& matrix) {
    if (!matrix.size())
        return 0;

    int m = matrix.size();
    int n = matrix[0].size();
    vector<vi> seen(m, vi(n));
    vector<vi> dp(m, vi(n));

    int longest = 1;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            longest = max(longest, dfs(matrix, dp, seen, INT_MIN, i, j));
        }
    }

    return longest;
}

/**
 * Leetcode 523. Continuous Subarray Sum
 * Time: O(n)
 * Space: O(k)
 *
 * @param nums
 * @param k
 * @return
 */
bool checkSubarraySum(vector<int>& nums, int k) {
    unordered_map<int, int> m;
    m[0] = -1;
    int sum = 0;

    for (int i = 0; i < nums.size(); i++) {
        sum += nums[i];
        if (!k) {
            sum %= k;
        }

        if (m.find(sum) != m.end()) {
            if (i - m[sum] > 1) return true;
        } else {
            m[sum] = i;
        }
    }

    return false;
}

string alienOrder(vector<string>& words) {
    map<char, vector<char>> dag;
    map<char, int> degree;
    set<char> vertices;
    string s;
    queue<char> q;

    for (auto word : words) {
        for (auto ch : word) {
            vertices.insert(ch);
        }
    }

    for (int i = 1; i < words.size(); i++) {
        string prev = words[i - 1];
        string cur = words[i];

        for (int k = 0; k < prev.length() && k < cur.length(); k++) {
            if (prev[k] != cur[k]) {
                dag[prev[k]].push_back(cur[k]);
                degree[cur[k]]++;

                break;
            }
        }
    }

    for (auto v : vertices) {
        if (!degree[v]) {
            q.push(v);
        }
    }

    while (!q.empty()) {
        char top = q.front(); q.pop();
        s += top;

        for (auto v : dag[top]) {
            degree[v]--;

            if (!degree[v])
                q.push(v);
        }
    }

    if (s.length() != vertices.size())
        return "";

    return s;
}

int maxPathSum(TreeNode* root, int& sum) {
    if (!root) return 0;

    // Max sum path from left and right subtree
    int left = maxPathSum(root->left, sum);
    int right = maxPathSum(root->right, sum);

    // Max sum path including root
    int root_sum = max(root->val, max(left, right) + root->val);

    // Max sum path going through root
    int sum_through_root = max(root_sum, left + right + root->val);
    sum = max(sum, sum_through_root);
    return root_sum;
}

/**
 * Leetcode: 124. Binary Tree Maximum Path Sum
 * Time: O(n)
 * Space: O(n)
 *
 * @param root
 * @return
 */
int maxPathSum(TreeNode* root) {
    if (!root)
        return 0;
    int sum = INT_MIN;
    maxPathSum(root, sum);
    return sum;
}

void pathSumThree(TreeNode* root, int sum, int k, int& count, map<int, int>& prefix) {
    if (!root) return;

    k += root->val;

    if (prefix.find(k - sum) != prefix.end()) {
        count += prefix[k - sum];
    }

    prefix[k]++;
    pathSumThree(root->left, sum, k, count, prefix);
    pathSumThree(root->right, sum, k, count, prefix);
    prefix[k]--;
}

/**
 * LeetCode 437 Path Sum III
 *
 * Using map for a prefix sum based solution
 *
 * @param root
 * @param sum
 * @return
 */
int pathSumThree(TreeNode* root, int sum) {
    map<int, int> prefix;
    prefix[0]++;
    int count = 0;
    pathSumThree(root, sum, 0, count, prefix);
    return count;
}

/**
 * Leetcode 325
 *
 * Given an array nums and a target value k, find the maximum length of a subarray that sums to k. If there isn't one, return 0 instead.
 *
 * @param nums
 * @param k
 * @return
 */
int maxSubArrayLen(vector<int>& nums, int k) {
    unordered_map<int, int> prefix;
    int sum = 0;
    int max_len = 0;
    for (int i = 0; i < nums.size(); i++) {
        sum += nums[i];

        if (sum == k)
            max_len = max(max_len, i + 1);

        if (prefix.find(sum - k) != prefix.end())
            max_len = max(max_len, i - prefix[sum - k] + 1);

        if (prefix.find(sum) == prefix.end())
            prefix[sum] = i;
    }

    return max_len;
}

bool validPalindrome(string s, int l, int r, int k) {
    if (l >= r)
        return true;

    if (s[l] == s[r])
        return validPalindrome(s, l + 1, r - 1, k);

    return k > 0 && (validPalindrome(s, l + 1, r, k - 1) || validPalindrome(s, l, r - 1, k - 1));
}

bool validPalindrome(string s) {
    return validPalindrome(s, 0, s.length() - 1, 1);
}

/**
 * Given a preorder and inorder traversal, construct a binary tree
 * Leetcode 105
 * Time: O(n)
 *
 * @param preorder
 * @param inorder
 * @param in_l
 * @param in_r
 * @param pre_start
 * @return
 */
TreeNode* buildTree(vector<int>& preorder, vector<int>& inorder, int in_l, int in_r, int pre_start) {
    if (in_l > in_r || pre_start >= preorder.size())
        return NULL;

    auto root = new TreeNode(preorder[pre_start]);

    // Find splitting point in in order traversal
    int in_index = 0;
    for (int i = in_l; i <= in_r; i++) {
        if (root->val == inorder[i]) {
            in_index = i;
        }
    }

    root->left = buildTree(preorder, inorder, in_l, in_index - 1, pre_start + 1);
    root->right = buildTree(preorder, inorder, in_index + 1, in_r, pre_start + in_index - in_l + 1);
}

TreeNode* buildTree(vector<int>& preorder, vector<int>& inorder) {
    return buildTree(preorder, inorder, 0, inorder.size() - 1, 0);
}

bool exist(vector<vector<char>>& board, string word, int k, int x, int y) {
    // Check if all characters of the word has been checked in the board
    if (k >= word.size())
        return true;

    if (x < 0 || y < 0 || x >= board.size() || y >= board[0].size() || board[x][y] == '#' || board[x][y] != word[k])
        return false;

    char tmp = board[x][y];
    // Mark cell as visited in place
    board[x][y] = '#';

    // Check for word in all four directions
    bool found = false;
    found |= exist(board, word, k + 1, x + 1, y) || exist(board, word, k + 1, x, y + 1);
    found |= exist(board, word, k + 1, x, y - 1) || exist(board, word, k + 1, x - 1, y);

    // Backtrack and mark as unvisited
    board[x][y] = tmp;
    return found;
}

bool exist(vector<vector<char>>& board, string word) {
    for (int x = 0; x < board.size(); x++) {
        for (int y = 0; y < board[0].size(); y++) {
            // Check if current cell can be used as a starting point for the word search
            if (board[x][y] == word[0]) {
                if (exist(board, word, 0, x, y))
                    return true;
            }
        }
    }

    return false;
}

void numIslands(vector<vector<char>>& grid, int x, int y) {
    if (x < 0 || x > grid.size() || y < 0 || y >= grid[0].size() || grid[x][y] == '#' || grid[x][y] == '0')
        return;

    grid[x][y] = '#';

    numIslands(grid, x + 1, y);
    numIslands(grid, x, y + 1);
    numIslands(grid, x - 1, y);
    numIslands(grid, x, y - 1);
}

int numIslands(vector<vector<char>>& grid) {
    int count = 0;
    for (int x = 0; x < grid.size(); x++) {
        for (int y = 0; y < grid[0].size(); y++) {
            if (grid[x][y] == '1') {
                count++;
                numIslands(grid, x, y);
            }
        }
    }

    return count;
}

struct Interval {
  int start;
  int end;
  Interval() : start(0), end(0) {}
  Interval(int s, int e) : start(s), end(e) {}
};

int minMeetingRooms(vector<Interval>& intervals) {
    if (!intervals.size())
        return 0;

    priority_queue<int, vector<int>, greater<int>> pq;
    sort(intervals.begin(), intervals.end(), [](Interval& lhs, Interval& rhs) {
        return lhs.start < rhs.start;
    });

    pq.push(intervals[0].end);

    int count = 1;
    for (int i = 1; i < intervals.size(); i++) {
        auto cur = intervals[i];
        auto top = pq.top();
        if (cur.start < top) {
            count += 1;
        } else {
            pq.pop();
        }

        pq.push(cur.end);
    }

    return count;
}

vector<Interval> merge(vector<Interval>& intervals) {
    if (intervals.size() <= 1)
        return intervals;

    vector<Interval> joint;
    sort(intervals.begin(), intervals.end(), [](Interval& lhs, Interval& rhs) {
       return lhs.start < rhs.start;
    });

    int s = intervals[0].start;
    int e = intervals[0].end;
    for (int i = 1; i < intervals.size(); i++) {
        auto cur = intervals[i];
        if (cur.start <= e) {
            e = max(cur.end, e);
        } else {
            joint.push_back({s, e});
            s = cur.start;
            e = cur.end;
        }
    }

    joint.push_back({s, e});
    return joint;
}

vector<int> twoSum(vector<int>& nums, int target) {

    map<int, int> s;
    vector<int> output;

    for (int i = 0; i < nums.size(); i++) {
        int check = target - nums[i];

        if (s.find(check) != s.end()) {
            output.push_back(s[check]);
            output.push_back(i);
            return output;
        }

        s[nums[i]] = i;
    }

    return output;
}

bool isInterleave(string s1, string s2, string s3, int i, int j, int k, vector<vector<vi>>& dp) {
    // Entire s3 has been interleaved
    if (k >= s3.size() && j >= s2.size() && i >= s1.size())
        return true;

    // Entire s3 has not been interleaved
    if (k >= s3.size())
        return false;

    if (dp[i][j][k] != -1) {
        return dp[i][j][k];
    }

    bool can = false;
    if (s3[k] == s1[i] && s3[k] == s2[j] && i < s1.size() && j < s2.size()) {
        can = isInterleave(s1, s2, s3, i + 1, j, k + 1, dp) || isInterleave(s1, s2, s3, i, j + 1, k + 1, dp);
    } else if (s3[k] == s1[i] && i < s1.size()) {
        can = isInterleave(s1, s2, s3, i + 1, j, k + 1, dp);
    } else if (s3[k] == s2[j] && j < s2.size()) {
        can = isInterleave(s1, s2, s3, i, j + 1, k + 1, dp);
    }

    dp[i][j][k] = can;
    return can;
}

bool isInterleave(string s1, string s2, string s3) {
    vector<vector<vi>> dp(s1.size(), vector<vi>(s2.size(), vi(s3.size(), -1)));

    return isInterleave(s1, s2, s3, 0, 0, 0, dp);
}

int largestRectangleArea(vector<int>& heights) {
    if (!heights.size())
        return 0;

    int n = heights.size();

    // We will be using a monotonic stack to build right[i] & left[i]
    vector<int> right;
    vector<int> left;

    stack<int> s;
    for (int i = 0; i < heights.size(); i++) {
        while (!s.empty() && heights[s.top()] >= heights[i])
            s.pop();

        if (s.empty())
            left.push_back(-1);
        else
            left.push_back(s.top());

        s.push(i);
    }

    stack<int> right_s;
    for (int j = heights.size() - 1; j >= 0; j--) {
        while (!right_s.empty() && heights[right_s.top()] >= heights[j])
            right_s.pop();

        if (right_s.empty())
            right.push_back(n);
        else
            right.push_back(right_s.top());

        right_s.push(j);
    }

    reverse(right.begin(), right.end());

    int max_area = INT_MIN;
    for (int i = 0; i < heights.size(); i++) {
        // Calculate area between left and right bounds of bucket i
        max_area = max(max_area, (right[i] - left[i] - 1) * heights[i]);
    }

    return max_area;
}

int maximalRectangle(vector<vector<char>>& matrix) {
    if (matrix.empty())
        return 0;

    int max_rec = 0;
    vi height(matrix[0].size(), 0);

    for (int i = 0; i < matrix.size(); i++){
        for (int j = 0; j < matrix[0].size(); j++){
            if (matrix[i][j] == '0')
                height[j] = 0;
            else
                height[j]++;
        }

        max_rec = max(max_rec, largestRectangleArea(height));
    }

    return max_rec;
}

/**
 * Given a string s, partition s such that every substring of the partition is a palindrome.
 * Return all possible palindrome partitioning of s.
 *
 * Time: O(m), m => number of unique palindromic partitions
 * @param s
 * @param output
 * @param k
 * @param word_stack
 */
void palindromePartition(string s, vector<vs> &output, int k, vs &word_stack) {
    if (k >= s.length()) {
        output.push_back(word_stack);
        return;
    }

    for (int j = k; j < s.length(); j++) {
        if (isPalindrome(s, k, j)) {
            word_stack.push_back(s.substr(k, j - k + 1));
            palindromePartition(s, output, j + 1, word_stack);
            word_stack.pop_back();
        }
    }
}

vector<vector<string>> palindromePartition(string s) {
    vector<vs> output;
    vs word_stack;

    palindromePartition(s, output, 0, word_stack);
    return output;
}

void combinationSum(vi &candidates, vector<vi> &output, vi &s, int j, int target) {
    if (target == 0) {
        output.push_back(s);
    } else {
        for (int i = j; i < candidates.size(); i++) {
            // Prune DFS
            if (target - candidates[i] >= 0) {
                s.push_back(candidates[i]);
                // Not i + 1 because we can reuse same elements
                combinationSum(candidates, output, s, i, target - candidates[i]);
                s.pop_back();
            }
        }
    }
}

/*
 * Leetcode 39
 *
 * Given a set of candidate numbers (candidates) (without duplicates) and a target number
 * find all unique combinations in candidates where the candidate numbers sums to target.
 */
vector<vi> combinationSum(vi& candidates, int target) {
    vector<vi> output;
    vi s;
    combinationSum(candidates, output, s, 0, target);
    return output;
}

/**
 * Detect cycle in a directed graph
 *
 * @param v
 * @param graph
 * @param seen
 * @param done
 * @return
 */
bool detectDirectedCycle(int v, vector<vi> &graph, vb &seen, vb &done) {
    seen[v] = true;
    bool has_cycle = false;

    for (auto w : graph[v]) {
        if (!seen[w]) {
            has_cycle |= detectDirectedCycle(w, graph, seen, done);
        } else if (!done[w]) {
            // Cycle detected
            return true;
        }
    }

    done[v] = true;
    return has_cycle;
}

/**
 * Leetcode 207 Course Schedule I
 *
 * @param numCourses
 * @param prerequisites
 * @return
 */
bool canFinish(int numCourses, vector<pair<int, int>>& prerequisites) {
    int n = numCourses;
    vector<vi> graph(n);
    vb seen(n);
    vb done(n);

    for (auto e : prerequisites) {
        graph[e.second].push_back(e.first);
    }

    for (int i = 0; i < n; i++) {
        if (!seen[i]) {
            auto detect = detectDirectedCycle(i, graph, seen, done);

            if (detect)
                return false;
        }
    }

    return true;
}

/**
 * Convert a roman string to decimal number
 *
 * @param s
 * @return
 */
int romanToInt(string s) {
    unordered_map<char, int> mapping = {
            {'I', 1}, {'V', 5}, {'X', 10}, {'L', 50}, {'C', 100}, {'D', 500}, {'M', 1000}
    };

    int decimal = 0;

    int i = 0;
    for (i = 0; i < s.length() - 1; i++) {
        auto current = s[i];
        auto next = s[i + 1];

        if (mapping[current] < mapping[next]) {
            decimal -= mapping[current];
        } else {
            decimal += mapping[current];
        }
    }

    decimal += mapping[s[i]];
    return decimal;
}

/**
 * Leetcode 208. Implement Trie
 *
 * Implement a trie with insert, search, and startsWith methods.
 */
typedef struct TrieNode {
    bool is_word = false;
    unordered_map<char, TrieNode*>  children;
} TrieNode;

class Trie {
    TrieNode* root;

public:
    Trie() {
        root = new TrieNode();
    }

    void insert(string word) {
        auto current = root;
        for (int j = 0; j < word.size(); j++) {
            char ch = word[j];
            if (current->children.find(ch) == current->children.end()) {
                current->children[ch] = new TrieNode();
            }

            current = current->children[ch];
        }

        current->is_word = true;
    }

    bool search(string word) {
        auto current = root;
        for (int j = 0; j < word.size(); j++) {
            char ch = word[j];
            if (current->children.find(ch) == current->children.end()) {
                return false;
            }

            current = current->children[ch];
        }

        return current->is_word;
    }

    bool startsWith(string prefix) {
        auto current = root;
        for (int j = 0; j < prefix.size(); j++) {
            char ch = prefix[j];
            if (current->children.find(ch) == current->children.end()) {
                return false;
            }

            current = current->children[ch];
        }

        queue<TrieNode*> s;
        s.push(current);
        while (!s.empty()) {
            auto top = s.front(); s.pop();
            if (top->is_word)
                return true;

            for (auto child : top->children) {
                s.push(child.second);
            }
        }

        return false;
    }

};


/**
 * Given an array of n positive integers and a positive integer s,
 * find the minimal length of a contiguous subarray of which the sum ≥ s.
 *
 * @param s
 * @param nums
 * @return
 */
int minSubArrayLen(int s, vi& nums) {
    int start = 0;
    int end = 0;
    int sum_so_far = 0;
    int min_sub = INT_MAX;

    // Use two pointers to scan the array
    while (end < nums.size()) {
        // Increase sum and move right pointer
        sum_so_far += nums[end++];

        // Try minimizing the subarray window
        while (sum_so_far >= s) {
            min_sub = min(min_sub, end - start);
            sum_so_far -= nums[start++]; // Decrease sum and move left pointer
        }
    }

    if (min_sub == INT_MAX)
        return 0;

    return min_sub;
}

/**
 * Given an unsorted array return whether an increasing subsequence of length 3 exists or not in the array.
 * Return true if there exists i, j, k, such that arr[i] < arr[j] < arr[k].
 * @param nums
 * @return
 */
bool increasingTriplet(vi& nums) {
    int max_one = INT_MAX;
    int max_two = INT_MAX;

    for (int i = 0; i < nums.size(); i++) {
        if (nums[i] <= max_one) {
            max_one = nums[i];
        } else if (nums[i] <= max_two) {
            max_two = nums[i];
        } else {
            return true;
        }
    }

    return false;
}

/**
 * Given a sorted array nums, remove the duplicates in-place such that duplicates appeared at most K times and
 * return the new length.
 *
 * @param nums
 * @param K Number of allowed repeats
 * @return
 */
int removeDuplicatesTwo(vector<int>& nums, int K) {
    if (nums.empty())
        return 0;

    int index = 0;
    int count = 1;
    for (int i = 1; i < nums.size(); i++) {
        if (nums[i] != nums[i - 1]) {
            nums[++index] = nums[i];
            count = 1;
        } else if (count < K) {
            nums[++index] = nums[i];
            count++;
        }
    }

    return ++index;
}

/**
 * Given a sorted array nums, remove the duplicates in-place such that each element appear only once and return the new length.
 *
 * @param nums
 * @return
 */
int removeDuplicates(vector<int> &nums) {
    if (nums.empty())
        return 0;

    int index = 0;
    for (int i = 1; i < nums.size(); i++) {
        if (nums[i] != nums[i - 1]) {
            nums[++index] = nums[i];
        }
    }

    return ++index;
}

int min_falling_path_sum(int row, int col, vector<vi> &A, vector<vi> mem) {
    if (mem[row][col] != INT_MAX)
        return mem[row][col];

    if (row == A.size() - 1)
        return A[row][col];

    int min_sum = INT_MAX;

    // Fall left
    if (col > 0)
        min_sum = min(min_sum, min_falling_path_sum(row + 1,  col - 1, A, mem));

    // Falling down
    min_sum = min(min_sum, min_falling_path_sum(row + 1, col, A, mem));

    // Fall right
    if (col < A.size() - 1)
        min_sum = min(min_sum, min_falling_path_sum(row + 1, col + 1, A, mem));

    mem[row][col] = min_sum + A[row][col];
    return mem[row][col];
}

int min_falling_path_sum(vector<vi> &A) {
    vector<vi> mem(A.size() + 1, vector<int>(A.size() + 1, INT_MAX));

    int min_sum = INT_MAX;
    for (int i = 0; i < A.size(); i++) {
        min_sum = min(min_sum, min_falling_path_sum(0, i, A, mem));
    }

    return min_sum;
}

/**
 * Leetcode 127. Word Ladder
 *
 * @param begin_word
 * @param end_word
 * @param word_list
 * @return
 */
int word_ladder(string begin_word, string end_word, vs &word_list) {
    queue<pair<string, int>> q;
    unordered_set<string> seen;
    unordered_set<string> word_set(word_list.begin(), word_list.end());

    q.push({begin_word, 0});

    while (!q.empty()) {
        auto top = q.front(); q.pop();
        int dist = top.second;
        string v = top.first;

        if (v == end_word)
            return dist + 1;

        seen.insert(v);

        for (int i = 0; i < v.size(); i++) {
            for (int j = 'a'; j <= 'z'; j++) {
                string transform = v;
                transform[i] = (char) j;

                if (seen.find(transform) == seen.end() && word_set.find(transform) != word_set.end()) {
                    seen.insert(transform);
                    q.push({transform, dist + 1});
                }
            }
        }
    }

    return 0;
}

void subsets(vi &nums, vector<vi> &output, vi &path, int k) {
    output.push_back(path);

    for (int j = k; j < nums.size(); j++) {
        path.push_back(nums[j]);
        subsets(nums, output, path, j + 1);
        path.pop_back();
    }
}

/**
 * Generate the power set of a set
 * @param nums
 * @return
 */
vector<vi> subsets(vi &nums) {
    vi path;
    vector<vi> output;
    subsets(nums, output, path, 0);

    return output;
}

void subsets_two(vi &nums, vector<vi> &output, vi &path, int k) {
    output.push_back(path);

    for (int j = k; j < nums.size(); j++) {
        // Skip if two adjacent elements are equal
        if (j > k && nums[j] != nums[j - 1])
            continue;

        path.push_back(nums[j]);
        subsets_two(nums, output, path, j + 1);
        path.pop_back();

    }
}

/**
 * Generate the power set of a set with duplicate elements
 * @param nums
 * @return
 */
vector<vi> subsets_two(vi &nums) {
    vi path;
    vector<vi> output;

    sort(nums.begin(), nums.end());
    subsets_two(nums, output, path, 0);

    return output;
}


void combinations(int n, int j, int k, vi &path, vector<vi> &output) {
    if (k == 0) {
        output.push_back(path);
    } else {
        for (int x = j; x <= n; x++) {
            path.push_back(x);
            combinations(n, j + 1, k - 1, path, output);
            path.pop_back();
        }
    }
}

/**
 * Generate all combinations give n Choose k
 * @param n
 * @param k
 * @return
 */
vector<vi> combinations(int n, int k) {
    vi path;
    vector<vi> output;
    combinations(n, 1, k, path, output);
    return output;
}

bool valid_parenthesis(string s) {
    stack<char> word_stack;

    for (char ch : s) {
        char cmp;

        if (ch == '(' || ch == '{' || ch == '[')
            word_stack.push(ch);
        else {
            if (ch == ')')
                cmp = '(';
            else if (ch == '}')
                cmp = '{';
            else if (ch == ']')
                cmp = '[';

            if (word_stack.empty() || word_stack.top() != cmp)
                return false;

            word_stack.pop();
        }
    }

    return word_stack.empty();
}

int hamming_distance(int x, int y) {
    int count = 0;
    int d = x ^ y;

    // Count the number of set bits
    while (d) {
        count += (d & 1);
        d >>= 1;
    }

    return count;
}

/**
 * Find Peak Element In An Array In Log(n) Time
  * @param nums
  * @return
 */
int find_peak_element(vi &nums) {
    int mid = 0;
    int left = 0;
    int right = nums.size() - 1;
    int n = nums.size();

    while (left <= right) {
        mid = (left + right) / 2;
        int left_bound = INT_MIN;
        int right_bound = INT_MIN;

        if (mid - 1 >= 0)
            left_bound = nums[mid - 1];
        if (mid + 1 < n)
            right_bound = nums[mid + 1];

        // Found a peak element
        if (left_bound < nums[mid] && right_bound < nums[mid])
            return mid;
        // Move in the direction of the larger element
        else if (left_bound > nums[mid])
            right = mid - 1;
        else
            left = mid + 1;
    }

    return 0;
}

int main() {
    vi v{1, 2};
    nextPermutation(v);

    for (auto j : v)
        cout << j << " ";

    return 0;
}