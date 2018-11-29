import random
from queue import deque

from typing import (
    List,
    Dict,
    Set,
    Tuple
)

random.seed(9000)

"""
Implementation of Union Find / Disjoint Set datastructure

https://www.cs.princeton.edu/~wayne/kleinberg-tardos/pdf/UnionFind.pdf
"""

class UnionFind(object):
    def __init__(self):
        self._tree = {}

        # Number of vertices in a group rooted by j
        self.group_size = {}

        # Height / Rank of each group's root
        self._rank = {}

        # Count of number of disjoint sets
        self.disjoint_sets = 0

    def set(self, item):
        self._tree[item] = item
        self._rank[item] = 0

        # Only one vertex is currently in the group
        self.group_size[item] = 1
        self.disjoint_sets += 1

    """
    This union operation links by rank and uses path compression
    Time: O(α(n)) ~ O(1), where α is the inverse ackermann function
    α(n) is a very very slow growing function, even slower in log(n). 1/α(n) ~< 5 for all practical values of n
    """

    def union(self, source, target):
        """ Link groups by rank """
        def link(source, target):
            if self._rank[source] > self._rank[target]:
                self._tree[target] = source
                self.group_size[source] += self.group_size[target]
            elif self._rank[source] < self._rank[target]:
                self._tree[source] = target
                self.group_size[target] += self.group_size[source]
            else:
                self._rank[source] += 1
                self._tree[target] = source
                self.group_size[source] += self.group_size[target]

            return (source, target)

        source_parent = self.find(source)
        target_parent = self.find(target)

        if target_parent == source_parent:
            return None

        self.disjoint_sets -= 1
        return link(source_parent, target_parent)

    def get_group_size(self, item):
        root = self.find(item)
        return self.group_size[root]

    def find(self, item):
        """ Compress path to root on find """
        def path_compression(item):
            if item != self._tree[item]:
                self._tree[item] = path_compression(self._tree[item])
                return self._tree[item]
            return item

        return path_compression(item)

    def group(self, item):
        root = self.find(item)
        print(self.group_size)
        return self.group_size[root]


class LinkedListNode(object):
    def __init__(self, key=-1, val=-1):
        self.prev = None
        self.next = None
        self.val = val
        self.key = key

    def __repr__(self):
        return '[' + str(self.val) + ']'


"""
Least Recently Used (LRU) cache
"""


class LRUCache(object):
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.size = 0

        self.map = {}

        self.head = LinkedListNode()
        self.tail = LinkedListNode()

        self.head.next = self.tail
        self.tail.prev = self.head

    """
    Move recently used node to the front of the list
    """
    def _move_to_front(key: int) -> None:
        recent_access_node = self.map[key]

        recent_access_node.prev.next = recent_access_node.next
        recent_access_node.next.prev = recent_access_node.prev

        prev_node = self.tail.prev
        recent_access_node.next = self.tail
        self.tail.prev = recent_access_node

        recent_access_node.prev = prev_node
        prev_node.next = recent_access_node

    """
    Get the value of the key if the key exists in the cache, otherwise return -1.
    """

    def get(self, key: int) -> int:
        if key not in self.map:
            return -1

        self._move_to_front(key)
        return self.map[key].val

    """
    Set or insert the value if the key is not already present.

    When the cache reached its capacity, it should invalidate the least
    recently used item before inserting a new item.
    """

    def put(self, key: int, value: int) -> Tuple[int, int]:
        if key in self.map:
            node = self.map[key]
            node.val = value

            self._move_to_front(key)
            return (key, value)

        if self.size >= self.capacity:
            del_key = self.head.next.key
            node_to_delete = self.map[del_key]
            del self.map[del_key]

            node_to_delete.prev.next = node_to_delete.next
            node_to_delete.next.prev = node_to_delete.prev
            node_to_delete.next = None
            node_to_delete.prev = None

            self.size -= 1

        new_node = LinkedListNode(key, value)
        second_last_node = self.tail.prev

        new_node.next = self.tail
        self.tail.prev = new_node
        new_node.prev = second_last_node
        second_last_node.next = new_node

        self.map[key] = new_node
        self.size += 1

        return (key, value)


"""
Trie
"""


class PrefixTree(object):
    def __init__(self):
        self.word_count = 0
        self.root = TrieNode()

    """ Insert a word into the tree """

    def insert(self, word: str) -> None:
        current = self.root

        for w in word:
            if w not in current.children:
                current[w] = TrieNode()
            current = current[w]
        current.is_word = True
        self.word_count += 1

    """ Find a word in the Prefix Tree """

    def search(self, word: str) -> bool:
        current = self.root

        for w in word:
            if w not in current:
                return False
            current = current[w]
        return current.is_word

    """ Find starting node using the given prefix """

    def _find_starting_node(self, prefix: str):
        current = self.root

        for w in prefix:
            if w not in current:
                return TrieNode()
            current = current[w]
        return current

    """ See if words with a given prefix exist in the tree """

    def exists_with_prefix(self, prefix: str) -> bool:
        start = self._find_starting_node(prefix)

        return len(start.children.keys()) > 0

    """ Get all words starting with a prefix """

    def starts_with(self, prefix: str) -> List[str]:
        """ Perform a dfs on the Trie """
        def _starts_with(node, word_list: List[str], stack: List[str]) -> None:
            if node.is_word:
                word_list.append("".join(stack))
            else:
                # Explore children of a node
                for start in node.children.keys():
                    stack.append(start)
                    _starts_with(node.children[start], word_list, stack)
                    stack.pop()

        start = self._find_starting_node(prefix)
        word_list = []
        stack = [prefix]

        _starts_with(start, word_list, stack)
        return word_list

    def longest_common_prefix(self, words: List[str]) -> str:
        for w in words:
            self.insert(w)

        prefix = []
        current = self.root
        while len(current.children.keys()) == 1:
            keys = [k for k in current.children.keys()]
            k = keys[0]
            prefix.append(k)
            current = current[k]

        return "".join(prefix)


class TrieNode(object):
    def __init__(self):
        self.is_word = False
        self.children = {}

    def __contains__(self, key):
        return key in self.children

    def __getitem__(self, key):
        return self.children[key]

    def __setitem__(self, key, val):
        self.children[key] = val

    def __repr__(self):
        return str([k for k in self.children.keys()])


"""
HashMap + Heap designed for Dijkstra's Single Source Shortest Path
"""


class HashHeap(object):
    def __init__(self):
        self._heap = []
        self._map = {}

    """
    Insert key and value into the HashMap + Heap
    :param k: Key
    :param v: Value
    """

    def insert(self, k, v):
        self._heap.append(self.Node(k, v))
        end = len(self._heap) - 1
        self._map[k] = end
        self._sift_up(end)

    """
    Remove minimum value from the Heap
    """

    def delete_min(self):
        if self.is_empty():
            return None

        if len(self._heap) <= 1:
            min_vert = self._heap.pop()
            del self._map[min_vert.key]
            return min_vert

        min_vert = self._heap[0]
        del self._map[min_vert.key]

        self._heap[0] = self._heap.pop()
        # Heapify after removing root
        self._sift_down(0)

        return min_vert

    def contains(self, k):
        return k in self._map.keys()

    """
    Update key and value pair
    :param k: Key to be updated
    :param v: Value
    """

    def update_key(self, k, v):
        index = self._map[k]
        self._heap[index].val = v

        # Heapify
        if self._heap[index] < self._heap[self._parent(index)]:
            self._sift_up(index)
        else:
            self._sift_down(index)

    def _sift_up(self, x):
        while x != 0 and self._heap[x] < self._heap[self._parent(x)]:
            self._swap(self._parent(x), x)
            x = self._parent(x)

    def _sift_down(self, x):
        while self._left(x) < len(self._heap) and self._right(x) < len(self._heap) and self._heap[x] >= self._heap[self._get_min_child(x)]:
            min_child = self._get_min_child(x)
            self._swap(x, min_child)
            x = min_child

    def _swap(self, x, y):
        tmp = self._heap[x]
        self._heap[x] = self._heap[y]
        self._heap[y] = tmp

        # Update index of node in the HashMap
        self._map[self._heap[x].key] = x
        self._map[self._heap[y].key] = y

    def is_empty(self):
        return len(self._heap) == 0

    """
    Peek at the minimum value in the Heap
    """

    def get_min(self):
        if self.is_empty():
            return None

        return self._heap[0]

    def _parent(self, x):
        return int((x - 1) / 2)

    def _left(self, x):
        return (2 * x) + 1

    def _right(self, x):
        return (2 * x) + 2

    def _get_min_child(self, x):
        left = self._left(x)
        right = self._right(x)

        if self._heap[left] < self._heap[right]:
            return left
        return right

    def __repr__(self):
        return str(self._heap)

    def __getitem__(self, k):
        return self._heap[self._map[k]]

    class Node(object):
        def __init__(self, key, val):
            self.key = key
            self.val = val

        def __lt__(self, other):
            return self.val < other.val

        def __le__(self, other):
            return self.val <= other.val

        def __gt__(self, other):
            return self.val > other.val

        def __ge__(self, other):
            return self.val >= other.val

        def __hash__(self):
            return hash(self.key)

        def __repr__(self):
            return str(self.val)
