---
author: Luca Lombardo
pubDatetime: 2022-12-01T00:00:00Z
title: Competitive Programming Friendly reminders
slug: competitive-programming-reminders
featured: false
draft: true
tags:
  - Competitive Programming
  - Algorithms and Data Structures
description: An incomplete summary of the most important algorithms and data structures to remember before a coding interview
---

## Arrays

Arrays are a collection of values. They are ordered and can be accessed by index. Arrays are zero-indexed, meaning the first element is at index 0. Arrays can be of any type, including other arrays.

## HashTable

An HashTable is a data structure that maps keys to values for highly efficient lookup. It is a structure that can be implemented with arrays, linked lists, binary search trees (in this case we would have $O(n \log n)$ lookup time. The advantage of this is potentially using less space, since we no longer allocate a large array. We can also iterate through the keys i order, which can be useful sometimes), or hash tables.

```python
HashTable = dict()
```

or if we want to use defaultdict

```python
from collections import defaultdict

HashTable = defaultdict(int)
# a default value of 0 is assigned to all keys
```

## ArrayList & ResizableArray

When you need an array-like data structure that offers dynamic resizing, you would usually use an Arraylist. An Arraylist is an array that resizes itself as needed while still providing $O(1)$ access.

```python
Array = [None] * 1000
# array of 1000 elements, all None
```

# Binary Search

Binary search is a search algorithm that finds the position of a target value within a sorted array. Binary search compares the target value to the middle element of the array. If they are not equal, the half in which the target cannot lie is eliminated and the search continues on the remaining half, again taking the middle element to compare to the target value, and repeating this until the target value is found. If the search ends with the remaining half being empty, the target is not in the array.

```python
def binary_search(arr, target):
    left = 0
    right = len(arr) - 1

    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1
```

## Algorithms

### QuickSelect

QuickSelect is a selection algorithm to find the k-th smallest/largest element in an unordered list. It is related to the quick sort sorting algorithm.

```python
def quick_select(arr, k):
    if len(arr) == 1:
        return arr[0]

    pivot = random.choice(arr)
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]

    if k <= len(left):
        return quick_select(left, k)
    elif k <= len(left) + len(middle):
        return middle[0]
    else:
        return quick_select(right, k - len(left) - len(middle))
```

# Stack

The stack data structure is precisely what it sounds like: a stack of data. In certain types of problems, it can be favorable to store data in a stack rather than in an array.

A stack uses LIFO (last-in first-out) ordering. That is, as in a stack of dinner plates, the most recent item added to the stack is the first item to be removed.

It uses the following operations:

- `push(item)`: adds an item to the top of the stack
- `pop()`: removes the top item from the stack
- `peek()`: returns the top of the stack without removing it
- `isEmpty()`: returns true if and only if the stack is empty

Unlike an array, a stack does not offer constant-time access to the ith item. However, it does allow constant-time adds and removes, as it doesn't require shifting elements around.

```python
stack = []

# add to the top of the stack
stack.append(1)
stack.append(2)

# remove from the top of the stack
stack.pop()
```

One case where stacks are often useful is in certain recursive algorithms. Sometimes you need to push temporary data onto a stack as you recurse, but then remove them as you backtrack (for example, because the recursive check failed). A stack offers an intuitive way to do this.

A stack can also be used to implement a recursive algorithm iteratively.

## Queue

A queue is an ordered collection of items where the addition of new items happens at one end, called the "rear," and the removal of existing items occurs at the other end, commonly called the "front." As an element enters the queue it starts at the rear and makes its way toward the front, waiting until that time when it is the next element to be removed.

It uses the following operations:

- `add(item)`: adds an item to the rear of the queue
- `remove()`: removes the front item from the queue
- `peek()`: returns the front of the queue without removing it
- `isEmpty()`: returns true if and only if the queue is empty

```python
from collections import deque

queue = deque() # double-ended queue
```

A queue can also be implemented with a linked list. In fact, they are essentially the same thing, as long as items are added and removed from opposite sides.

One place where queues are often used is in breadth-first search or in implementing a cache. In breadth-first search, for example, we used a queue to store a list of the nodes that we need to process. Each time we process a node, we add its adjacent nodes to the back of the queue. This allows us to process nodes in the order in which they are viewed.

# Linked List

A Linked list is a data structure that represents a sequence of nodes. In as singly linked list, each node points to the next node in the linked list. In a doubly linked list, each node points to the next node and the previous node. In a circular linked list, the tail node points to the head node.

Unlike an array, a linked list does not provide costant access time to a particolar index within the list. This means, that if you'd like to find the kth element in a linked list, you'd have to traverse the list from the head node to the kth node. This is because the nodes in a linked list are not stored in contiguous memory locations.

## Singly Linked List

```python
# Definition for singly-linked list.
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None
```

In this implementation, the head node is the first node in the linked list. The tail node is the last node in the linked list. The tail node's next pointer points to None.

## Techniques

- Slow and Fast Pointer

_This is a technique that is used to solve problems that involve linked list traversal. The slow pointer moves one node at a time, while the fast pointer moves two nodes at a time. This technique is used to find the middle node of a linked list, or to determine if a linked list has a cycle._

- Dummy linked list

```python
dummy = ListNode()
```

- Check if has a cycle

```python
def hasCycle(self, head: ListNode) -> bool:
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True
    return False
```

- Reverse a linked list

```python
def reverseList(self, head: ListNode) -> ListNode:
    prev, curr = None, head

    while curr:
        temp = curr.next
        curr.next = prev
        prev = curr
        curr = temp
    return prev
```

- Sometimes it's useful to split a linked list into two parts

# Trees

A nice way to understand a tree is with a recursive explanation. A tree is a data structure composed of nodes.

- Each tree has a root node. (Actually, this isn't strictly necessary in graph theory, but it's usually how we use trees in programming)

- The root node has zero or more child nodes.

- Each child node has zero or more child nodes, and so on.

The tree cannot contain cycles. The nodes may or may not be in a particular order, they could have any data type as values, and they may or may not have links back to their parent nodes.

```python
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.children = []
```

## Binary Tree

A binary tree is a tree in which each node has up to two children. Not all trees are binary trees. There are occasions when you might have a tree that is not a binary tree. For example, suppose you were using a tree to represent a bunch of phone numbers. In this case, you might use a 10-ary tree, with each node having up to 10 children (one for each digit}.
A node is called a "leaf" node if it has no children.

```python
class BinaryTreeNode:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None
```

## Binary Search Tree

A binary search tree is a binary tree in which every node fits a specific ordering property: _all left descendents_ $\leq n <$ _all right descendents_. This must be true for each node $n$. Note that this inequality must be true for all of a node's descendents, not just its immediate children

> The definition of a binary search tree can vary slightly with respect to equality. Under some definitions, the tree cannot have duplicate values. In others, the duplicate values will be on the right or can be on either side. All are valid definitions, but you should clarify this with your interviewer.

### Balanced vs Unbalanced

While many trees are balanced, not all are. Note that balancing a tree does not mean the left and right subtrees are exactly the same size (like you see under "perfect binary trees")

One way to think about it is that a "balanced" tree really means something more like "not terribly imbalanced:' It's balanced enough to ensure $O(\log n)$ times for insert and find, but it's not necessarily as balanced as it could be.

### Complete Binary Tree

A complete binary tree is a binary tree in which every level of the tree is fully filled, except for perhaps the last level. lo the extent that the last level is filled, it is filled left to right.

### Full Binary Tree

A full binary tree is a binary tree in which every node has either zero or two children. That is, no nodes have only one child.

### Perfect Binary Tree

A perfect binary tree is one that is both full and complete. All leaf nodes will be at the same level, and this level has the maximum number of nodes.

## Binary Tree Traversal

### In-order Traversal

In-order traversal means to "visit" (often, print) the left branch, then the current node, and finally, the right branch.

```python
def inOrderTraversal(self, root):
    if root:
        self.inOrderTraversal(root.left)
        print(root.val)
        self.inOrderTraversal(root.right)
```

### Pre-order Traversal

Pre-order traversal visits the current node before its child nodes (hence the name "pre-order").

```python
def preOrderTraversal(self, root):
    if root:
        print(root.val)
        self.preOrderTraversal(root.left)
        self.preOrderTraversal(root.right)
```

### Post-order Traversal

Post-order traversal visits the current node after its child nodes (hence the name "post-order").

```python
def postOrderTraversal(self, root):
    if root:
        self.postOrderTraversal(root.left)
        self.postOrderTraversal(root.right)
        print(root.val)
```

### Level-order Traversal

Level-order traversal visits the nodes level by level. You'll often see this done with a queue.

```python
def levelOrderTraversal(self, root):
    if not root:
        return
    queue = [root]
    while queue:
        node = queue.pop(0)
        print(node.val)
        if node.left:
            queue.append(node.left)
        if node.right:
            queue.append(node.right)
```

## Binary Heaps (min-heap and max-heap)

We'll just discuss min-heaps here. Max-heaps are essentially equivalent, but the elements are in descending order rather than ascending order

A min-heap is a complete binary tree (that is, totally filled other than the rightmost elements on the last level) where each node is smaller than its children. The root, therefore, is the minimum element in the tree.

We have two key operations on a min-heap: `insert` and `extract_min`.

### Insert

When we insert into a min-heap, we always start by inserting the element at the bottom. We insert at the rightmost spot so as to maintain the complete tree property.

Then, we "fix" the tree by swapping the new element with its parent, until we find an appropriate spot for the element We essentially bubble up the minimum element.

This takes $O(\log n)$ time, where $n$ is the number of elements in the heap.

### Extract Min

Finding the minimum element of a min-heap is easy: it's always at the top. The trickier part is how to remove it. (In fact, this isn't that tricky.)

First, we remove the minimum element and swap it with the last element in the heap (the bottommost, rightmost element). Then, we bubble down this element, swapping it with one of its children until the min-heap property is restored.

Do we swap it with the left child or the right child? That depends on their values. There's no inherent ordering between the left and right element, but you'll need to take the smaller one in order to maintain the min-heap ordering.

This takes $O(\log n)$ time.

# Tries

A trie (sometimes called a prefix tree} is a funny data structure. It comes up a lot in interview questions, but algorithm textbooks don't spend much time on this data structure.

A trie is a variant of an n-ary tree in which characters are stored at each node. Each path down the tree may represent a word.

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_word = False
```

The `* nodes` (sometimes called "null nodes") are often used to indicate complete words. For example, the fact that there is a `* node` under `MANY` indicates that `MANY` is a complete word. The existence of the `MA` path
indicates there are words that start with `MA`.

The actual implementation of these `* nodes` might be a special type of child (such as a `TerminatingTrieNode`, which inherits from `TrieNode`. Or, we could use just a boolean flag terminates within the "parent" node.

A node in a trie could have anywhere from 1 through `ALPHABET_SIZE + 1` children (or, `O` through `ALPHABET_SIZE` if a boolean flag is used instead of a `* node`).

Very commonly, a trie is used to store the entire (English) language for quick prefix lookups. While a hash table can quickly look up whether a string is a valid word, it cannot tell us if a string is a prefix of any valid words. A trie can do this very quickly.

> How quickly? A trie can check if a string is a valid prefix in $O(K)$ time, where K is the length of the string. This is actually the same runtime as a hash table will take. Although we often refer to hash table lookups as being $O(1)$ time, this isn't entirely true. A hash table must read through all the characters in the input which takes $O(K)$ time in the case of a word lookup

Many problems involving lists of valid words leverage a trie as an optimization. In situations when we search through the tree on related prefixes repeatedly

# Backtracking

Backtracking is a general algorithm for finding all (or some) solutions to some computational problems, notably constraint satisfaction problems, that incrementally builds candidates to the solutions, and abandons each partial candidate ("backtracks") as soon as it determines that the candidate cannot possibly be completed to a valid solution.

- Sometimes is useful to think of backtracking as an intelligent "brute force" approach.
- Recursive DFS is often a good way to implement backtracking
- It's usually a good idea to start by writing a recursive backtracking every time you see a subset, combination, permutation, or partition problem.

# Graphs

Many programming problems can be solved by modeling the problem as a graph
problem and using an appropriate graph algorithm.

Atree is actually a type of graph, but not all graphs are trees. Simply put, a tree isa connected graph without cycles. A graph is simply a collection d nodes with edges between (some of) them.

- Graphs can be either **directed** or **undirected**. While directed edges are like a one-way street, undirected edges are like a two-way street.
- The graph might consist of multiple isolated subgraphs. If there is a path between every pair of vertices, it is called a "**connected graph**".
- The graph can also have cycles (or not). An "**acyclic graph**" is one without cycles

## Adjacency Lists

This is the most common way to represent a graph. Every vertex (or node) stores a list of adjacent vertices. In an undirected graph, an edge like `(a, b)` would be stored twice: once in `a`'s adjacent vertices and once in `b`'s adjacent vertices.

A simple class definition for a graph node could look essentially the same as a tree node.

```python
class Graph:
    def __init__(self):
        self.nodes = {}
```

```python
class Node:
    def __init__(self, val = 0, neighbors = None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []
```

You don't necessarily need any additional classes to represent a graph. An array (or a hash table) of lists (arrays, arraylists, linked lists, etc.) can store the adjacency list.

## Adjacency Matrix

An adjacency matrix is an `NxN` boolean matrix (where `N` is the number of nodes), where a true value at `matrix[i][j]` indicates an edge from node `i` to node `j`. (You can also use an integer matrix with Os and 1s.)

Note that in an undirected graph, an adjacency matrix will be symmetric. In a directed graph, it will not (necessarily) be.

The same graph algorithms that are used on adjacency lists (breadth-first search, etc.) can be performed with adjacency matrices, but they may be somewhat less efficient. In the adjacency list representation, you can easily iterate through the neighbors of a node. In the adjacency matrix representation, you will need to iterate through all the nodes to identify a node's neighbors.

## Graph Search

The two most common ways to search a graph are **depth-first search** and **breadth-first search**.

- In depth-first search (DFS), we start at the root (or another arbitrarily selected node) and explore each branch completely before moving on to the next branch. That is. we go deep first (hence the name depth-first search) before we go wide.

- In breadth-first search (BFS), we start at the root (or another arbitrarily selected node) and explore each neighbor before going on to any of their children. That is, we go wide (hence breadth-first search) before
  we go deep.

Breadth-first search and depth-first search tend to be used in different scenarios. DFS is often preferred if we want to visit every node in the graph. Both will work just fine, but depth-first search is a bit simpler. However, if we want to find the shortest path (or just any path) between two nodes, BFS is generally better.

### DFS

In DFS, we visit a node a and then iterate through each of a's neighbors. When visiting `a` node `b` that is a neighbor of `a`, we visit all of `b`'s neighbors before going on to `a`'s other neighbors. That is, a exhaustively
searches `b`'s branch before any of its other neighbors.

Note that pre-order and other forms of tree traversal are a form of DFS. The key difference is that when implementing this algorithm for a graph, we must check if the node has been visited. If we don't, we risk getting stuck in an infinite loop.

```python
def dfs(node):
    if node is None:
        return
    visit(node)
    node.visited = True
    for next in node.adjacent:
        if next.visited == False:
            dfs(next)
```

### BFS

BFS is a bit less intuitive. The main tripping point is the (false) assumption that BFS is recursive. It's not. Instead, it uses a queue.

In BFS, node a visits each of a's neighbors before visiting any of their neighbors. You can think of this as searching level by level out from a. An iterative solution involving a queue usually works best.

```python
def bfs(graph, start):
    queue = []
    queue.append(start)
    start.visited = True
    while queue:
        r = queue.pop(0)
        visit(r)
        for n in r.adjacent:
            if n.visited == False:
                n.visited = True
                queue.append(n)
```

If you are asked to implement BFS, the key thing to remember is the use of the queue. The rest of the algorithm flows from this fact.

### Bidirectional Search

Bidirectional search is used to find the shortest path between a source and destination node. It operates by essentially running two simultaneous breadth-first searches, one from each node. When their searches collide, we have found a path.

![](https://i.imgur.com/AQTtl8H.png)

To see why this is faster, consider a graph where every node has at most k adjacent nodes and the shortest path from node s to node t has length d.

- In traditional breadth-first search, we would search up to `k` nodes in the first "level" of the search. In the second level, we would search up to `k` nodes for each of those first `k` nodes, so $k^2$ nodes total (thus far).
  We would do this $d$ times, so that's $O(k^d)$ nodes.

- In bidirectional search, we have two searches that collide after approximately $d/2$ levels (the midpoint of the path). The search from `s` visits approximately $2k^{d/2}$, as does the search from `t`. That's approximately $O(k^{d/2})$ nodes total.

So the the bidirectional search is actually faster by a factor of $k^{d/2}$.

### Topological Sort

## Shortest Path

Finding a shortest path between two nodes of a graph is an important problem
that has many practical applications. For example, a natural problem related to a road network is to calculate the shortest possible length of a route between two cities, given the lengths of the roads.

In an unweighted graph, the length of a path equals the number of its edges,
and we can simply use breadth-first search to find a shortest path. However, we focus on weighted graphs where more sophisticated algorithms are needed for finding shortest paths

### Bellman-Ford

The Bellman–Ford algorithm finds shortest paths from a starting node to all
nodes of the graph. The algorithm can process all kinds of graphs, provided that the graph does not contain a cycle with negative length. If the graph contains a negative cycle, the algorithm can detect this.

The algorithm keeps track of distances from the starting node to all nodes
of the graph. Initially, the distance to the starting node is 0 and the distance to all other nodes in infinite. The algorithm reduces the distances by finding edges that shorten the paths until it is not possible to reduce any distance

The algorithm is based on the following observation: if we know the shortest path from the starting node to all nodes except one, we can find the shortest path to all nodes by adding one edge to the shortest path. The algorithm iterates through all edges of the graph and relaxes each edge. The relaxation of an edge means that we try to find a shorter path to the destination node of the edge by going through the source node of the edge. The algorithm repeats this process for a number of iterations, which is equal to the number of nodes in the graph. If the algorithm finds a shorter path to a node, it means that the graph contains a negative cycle. The algorithm can detect this by keeping track of the number of iterations. If the number of iterations is equal to the number of nodes in the graph, the graph contains a negative cycle.

```python
def bellman_ford(graph, source):
    distance, predecessor = dict(), dict()
    for node in graph:
        distance[node] = float('inf')
        predecessor[node] = None
    distance[source] = 0
    for _ in range(len(graph) - 1):
        for u, v in graph.edges:
            if distance[u] + graph.edges[u, v] < distance[v]:
                distance[v] = distance[u] + graph.edges[u, v]
                predecessor[v] = u
    for u, v in graph.edges:
        assert distance[u] + graph.edges[u, v] >= distance[v]
    return distance, predecessor
```

The time complexity of the algorithm is $O(nm)$, because the algorithm consists of `n − 1` rounds and iterates through all `m` edges during a round. If there are no negative cycles in the graph, all distances are final after `n − 1` rounds, because each shortest path can contain at most `n − 1` edges.

In practice, the final distances can usually be found faster than in `n − 1` rounds. Thus, a possible way to make the algorithm more efficient is to stop the algorithm if no distance can be reduced during a round.

> **Note:** We need to use this algorithm when the graph contains negative edges. If the graph contains only positive edges, we can use Dijkstra's algorithm, which is more efficient.

### Dijkstra's Algorithm

Dijkstra’s algorithm finds shortest paths from the starting node to all nodes of the graph, like the Bellman–Ford algorithm. The benefit of Dijkstra's algorithm is that it is more efficient and can be used for processing large graphs. However, the algorithm requires that there are no negative weight edges in the graph.

Like the Bellman–Ford algorithm, Dijkstra’s algorithm maintains distances
to the nodes and reduces them during the search. Dijkstra’s algorithm is efficient, because it only processes each edge in the graph once, using the fact that there are no negative edges.

An efficient implementation of Dijkstra’s algorithm requires that it is possible to efficiently find the minimum distance node that has not been processed. An appropriate data structure for this is a priority queue that contains the nodes ordered by their distances. Using a priority queue, the next node to be processed can be retrieved in logarithmic time

```python
def dijkstra(graph, source):
    distance, predecessor = dict(), dict()
    for node in graph:
        distance[node] = float('inf')
        predecessor[node] = None
    distance[source] = 0
    queue = PriorityQueue()
    queue.put((0, source))
    while not queue.empty():
        _, u = queue.get()
        for v in graph[u]:
            alt = distance[u] + graph.edges[u, v]
            if alt < distance[v]:
                distance[v] = alt
                predecessor[v] = u
                queue.put((alt, v))
    return distance, predecessor
```

The time complexity of the above implementation is $O(n + m \log m)$, because
the algorithm goes through all nodes of the graph and adds for each edge at most one distance to the priority queue.

### Floyd-Warshall

The Floyd–Warshall algorithm provides an alternative way to approach the
problem of finding shortest paths. Unlike the other algorithms of this chapter, it finds all shortest paths between the nodes in a single run.

The algorithm maintains a two-dimensional array that contains distances
between the nodes. First, distances are calculated only using direct edges between the nodes, and after this, the algorithm reduces distances by using intermediate nodes in paths.

The advantage of the Floyd–Warshall algorithm that it is easy to implement. The following code constructs a distance matrix where `distance[a][b]` is the shortest distance between nodes a and b. First, the algorithm initializes distance using the adjacency matrix adj of the graph:

```python
def floyd_warshall(graph):
    distance = {u: {v: float('inf') for v in graph} for u in graph}
    for u in graph:
        distance[u][u] = 0
    for u, v in graph.edges:
        distance[u][v] = graph.edges[u, v]
    for k in graph:
        for i in graph:
            for j in graph:
                distance[i][j] = min(distance[i][j], distance[i][k] + distance[k][j])
    return distance
```

The time complexity of the algorithm is $O(n^3)$, because it contains three
nested loops that go through the nodes of the graph. Since the implementation of the Floyd–Warshall algorithm is simple, the algorithm can be a good choice even if it is only needed to find a single shortest path in the graph. However, the algorithm can only be used when the graph is so
small that a cubic time complexity is fast enough.

# Recursion

While there are a large number of recursive problems, many follows similar patterns. A good hint that a problem is recursive is that it can be built of sub-problems of the same type.

When you hear a problem beginning with the following statements, it's often (though not always) a good candidate for recursion:

- _Design an algorithm to compute the nth ..._
- _Write code to list the first n ..._
- _Implement a method to compute all ..,_

Recursive solutions, by definition, are built off of solutions to subproblems. Many times, this will mean simply to compute $f(n)$ by adding something, removing something, or otherwise changing the solution
for $f(n-1)$. In other cases, you might solve the problem for the first half of the data set, then the second half, and then merge those results.
There are many ways you might divide a problem into subproblems.Three of the most common approaches to develop an algorithm are bottom-up, top-down, and half-and-half.

## Bottom-Up

The bottom-up approach is often the most intuitive. We start with knowing how to solve the problem for a simple case, like a list with only one element. Then we figure out how to solve the problem for two
elements, then for three elements, and so on. The key here is to think about how you can build the solution for one case off of the previous case (or multiple previous cases).

## Top-Down

The top-down approach can be more complex since it's less concrete. But sometimes, it's the best way to think about the problem.

In these problems, we think about how we can divide the problem for case `N` into subproblems. Be careful of overlap between the cases.

## Half-and-Half

In addition to top-down and bottom-up approaches, it's often effective to divide the data set in half.

For example, binary search works with a "half-and-half" approach. When we look for an element in a sorted array, we first figure out which half of the array contains the value. Then we recurse and search for it in that half.

Merge sort is also a "half-and-half" approach. We sort each half of the array and then merge together the
sorted halves.

## Recursion vs. Iteration

Recursive algorithms can be very space inefficient. Each recursive call adds a new layer to the stack, which means that if your algorithm recurses to a depth of `n`, it uses at least $O(n)$ memory.

For this reason, it's often better to implement a recursive algorithm iteratively. All recursive algorithms can be implemented iteratively, although sometimes the code to do so is much more complex. Before diving
into recursive code, ask yourself how hard it would be to implement it iteratively,

# Dynamic Programming

Dynamic programming is a technique that combines the correctness of complete search and the efficiency of greedy algorithms. Dynamic programming can be applied if the problem can be divided into overlapping subproblems that can be solved independently.

There are two uses for dynamic programming:

- **Finding an optimal solution**: We want to find a solution that is as large as possible or as small as possible.
- **Counting the number of solutions:** We want to calculate the total number of possible solutions.

<!-- # Greedy

# Intervals -->

# Bit Manipulation

All data in computer programs is internally stored as bits, i.e., as numbers 0 and 1. In programming, an n bit integer is internally stored as a binary number that consists of `n` bits. For example, the `C++` type int is a `32-bit` type, which means that every `int` number consists of 32 bits.

The bits in the representation are indexed from right to left. To convert a bit representation $b_k · · · b_2 b_1 b_0 into a number, we can use the formula

$$ \sum\_{i=0}^k b_i 2^i $$

For example, the number 13 is represented as 1101 in binary. To convert it to a decimal number, we can use the formula:

$$ 1 \cdot 2^3 + 1 \cdot 2^2 + 0 \cdot 2^1 + 1 \cdot 2^0 = 8 + 4 + 0 + 1 = 13 $$

The following code converts a binary number to a decimal number:

```python
def binary_to_decimal(binary):
    decimal = 0
    for digit in binary:
        decimal = decimal * 2 + int(digit)
    return decimal
```

The time complexity of the above implementation is $O(n)$, where $n$ is the length of the binary number.

The bit representation of a number is either signed or unsigned. Usually a signed representation is used, which means that both negative and positive numbers can be represented. signed variable of n bits can contain any integer
between $-2^{n-1}$ and $2^{n-1} - 1$. For example, the `C++` type `int` is a signed `32-bit` type, which means that it can contain any integer between $-2^{31}$ and $2^{31} - 1$.

The first bit in a signed representation is the sign of the number (`0` for nonnegative numbers and `1` for negative numbers), and the remaining `n−1` bits contain the magnitude of the number. Two’s complement is used, which means that the opposite number of a number is calculated by first inverting all the bits in the number, and then increasing the number by one.

For example, the number 13 is represented as 1101 in binary. To convert it to a two’s complement representation, we first invert all the bits:

$$ \overline{1101} = 0010 $$

Then we increase the number by one:

$$ 0010 + 1 = 0011 $$

The result is the two’s complement representation of the number 13, which is 0011. The number -13 is represented as 1101 in binary. To

In an unsigned representation, only nonnegative numbers can be used, but the upper bound for the values is larger. An unsigned variable of `n` bits can contain any integer between 0 and $2^n - 1$. For example, the `C++` type `unsigned int` is an unsigned `32-bit` type, which means that it can contain any integer between 0 and $2^{32} - 1$.

## Bitwise Operators

### Bitwise AND

The bitwise AND operator `&` compares two numbers on a bit level and returns a number where the bits of that number are turned on if the corresponding bits of both numbers are `1`.

For example, the number 13 is represented as 1101 in binary. The number 7 is represented as 0111 in binary. The bitwise AND of 13 and 7 is 0101, which is the number 5.

```python
def bitwise_and(a, b):
    return a & b
```

### Bitwise OR

The bitwise OR operator `|` compares two numbers on a bit level and returns a number where the bits of that number are turned on if the corresponding bits of either of the two numbers are `1`.

For example, the number 13 is represented as 1101 in binary. The number 7 is represented as 0111 in binary. The bitwise OR of 13 and 7 is 1111, which is the number 15.

```python
def bitwise_or(a, b):
    return a | b
```

### Bitwise XOR

The bitwise XOR operator `^` compares two numbers on a bit level and returns a number where the bits of that number are turned on if the corresponding bits of either of the two numbers are `1`, but not both.

For example, the number 13 is represented as 1101 in binary. The number 7 is represented as 0111 in binary. The bitwise XOR of 13 and 7 is 1010, which is the number 10.

```python
def bitwise_xor(a, b):
    return a ^ b
```

### Bitwise NOT

The bitwise NOT operator `~` inverts all the bits of its operand.

For example, the number 13 is represented as 1101 in binary. The bitwise NOT of 13 is 0010, which is the number 2.

```python
def bitwise_not(a):
    return ~a
```

### Bitwise Shifts

The bitwise left shift operator `<<` shifts the bits of its first operand the specified number of positions to the left. The bits shifted off the left end are discarded. The bits on the right end are filled with zeros.

For example, the number 13 is represented as 1101 in binary. The bitwise left shift of 13 by 1 is 1010, which is the number 10.

```python
def bitwise_left_shift(a, b):
    return a << b
```

The bitwise right shift operator `>>` shifts the bits of its first operand the specified number of positions to the right. The bits shifted off the right end are discarded. The bits on the left end are filled with zeros.

For example, the number 13 is represented as 1101 in binary. The bitwise right shift of 13 by 1 is 0110, which is the number 6.

```python
def bitwise_right_shift(a, b):
    return a >> b
```

## Bitwise Tricks

### Check if a number is even or odd

A number is even if the last bit is `0` and odd if the last bit is `1`. We can check if a number is even or odd by checking the last bit of the number. If the last bit is `0`, the number is even, otherwise, it is odd.

```python
def is_even(a):
    return a & 1 == 0
```

### Check if a number is a power of two

A number is a power of two if it has only one bit set to `1`. We can check if a number is a power of two by checking if the number has only one bit set to `1`.

```python
def is_power_of_two(a):
    return a & (a - 1) == 0
```

### Swap two numbers

We can swap two numbers by using the bitwise XOR operator `^`. The bitwise XOR of two numbers is `0` if the two numbers are the same, and it is `1` if the two numbers are different. We can use this property to swap two numbers.

```python
def swap(a, b):
    a ^= b
    b ^= a
    a ^= b
    return a, b
```

### Get the last set bit

We can get the last set bit of a number by using the bitwise AND operator `&` with the two’s complement of the number.

```python
def get_last_set_bit(a):
    return a & -a
```

### Hamming distance

The Hamming distance between two integers is the number of positions at which the corresponding bits are different.

Given two integers `x` and `y`, calculate the Hamming distance.

```python
def hamming_distance(x, y):
    return bin(x ^ y).count('1')
```

### Count the number of set bits

We can count the number of set bits in a number by repeatedly clearing the last set bit of the number until the number becomes `0`.

```python
def count_set_bits(a):
    count = 0
    while a:
        a &= a - 1
        count += 1
    return count
```

### Reverse bits

We can reverse the bits of a number by repeatedly shifting the number to the right and appending the last bit to the result.

```python
def reverse_bits(a):
    result = 0
    for _ in range(32):
        result = (result << 1) | (a & 1)
        a >>= 1
    return result
```
