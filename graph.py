from queue import Queue
from datastructures import HashHeap

"""
Graphs
"""


class Graph(object):
    def __init__(self, vertex_count):
        self.adj = {}
        self.vertex_count = vertex_count
        self.edge_weights = [[float('inf') for i in range(
            self.vertex_count)] for j in range(self.vertex_count)]

        for x in range(0, vertex_count):
            self.adj[x] = []
            self.edge_weights[x][x] = 0

    def add_edge(self, u, v, weight=0):
        self.adj[u].append(v)
        self.adj[v].append(u)

        if weight:
            self._add_edge_weight(u, v, weight)
            self._add_edge_weight(v, u, weight)

    def add_directed_edge(self, u, v, weight=0):
        self.adj[u].append(v)

        self.edge_weights[u][v] = weight

    def get_edge_weight(self, u, v):
        return self.edge_weights[u][v]

    def _add_edge_weight(self, u, v, weight):
        self.edge_weights[u][v] = weight

    """
    Visit all vertices in a depth-first manner
    Time: O(|V| + |E|)
    """

    def dfs(self):
        marked = [False for x in range(0, self.vertex_count)]

        for v in range(0, self.vertex_count):
            if not marked[v]:
                self._dfs(v, marked)

    def _dfs(self, v, marked):
        marked[v] = True
        print(v)

        for w in self.adj[v]:
            if not marked[w]:
                self._dfs(w, marked)

    """
    Visit all vertices in a breadth-first manner
    Time: O(|V| + |E|)
    """

    def bfs(self, v):
        marked = [False for x in range(0, self.vertex_count)]
        q = Queue()
        q.put(v)

        while not q.empty():
            vert = q.get()
            marked[vert] = True
            print(vert)

            for w in self.adj[vert]:
                if not marked[w]:
                    q.put(w)
                    marked[w] = True

    """
    Check if the graph is connected or not
    """

    def is_graph_connected(self):
        marked = [False] * self.vertex_count
        self._dfs(1, marked)
        return all(marked)

    def in_degree(self):
        degree_list = [0] * self.vertex_count

        for v in self.adj.keys():
            for w in self.adj[v]:
                degree_list[w] += 1
        return degree_list

    """
    :param q:           Queue
    :param degree_list: In-degree list of all vertices

    Append all vertices with 0 degree to the queue
    """

    def _build_zero_degree_queue(self, q, degree_list):
        for idx, v in enumerate(degree_list):
            if v == 0 and v != -1:
                q.put(idx)
                degree_list[idx] = -1  # Flag vertex as processed
    """
    Queue based topological sort
    Time: O(|V| + |E|)

    Algorithm:
    1. Store each vertex’s In-Degree in an array D
    2. Initialize queue with all “in-degree=0” vertices
    3. While there are vertices remaining in the queue:
        (a) Dequeue and output a vertex
        (b) Reduce In-Degree of all vertices adjacent to it by 1
        (c) Enqueue any of these vertices whose In-Degree became zero
    4. If all vertices are output then success, otherwise there is a cycle.
    """

    def topological_sort(self):
        topological_order = []
        q = Queue()

        # Get in degree of all vertices
        degree_list = self.in_degree()

        # Enqueue all vertices with degree 0 to the queue
        self._build_zero_degree_queue(q, degree_list)

        while not q.empty():
            vertex = q.get()
            topological_order.append(vertex)

            # Decrement the degree of all vertices adjacent to the removed vertex
            for v in self.adj[vertex]:
                if degree_list[v] > 0:
                    degree_list[v] -= 1

            self._build_zero_degree_queue(q, degree_list)

        if len(topological_order) != self.vertex_count:
            raise Exception("Cycle Detected")

        return topological_order

    def tarjans_reverse_topological_sort(self):
        done = [False for x in range(0, self.vertex_count)]
        marked = [False for x in range(0, self.vertex_count)]
        topological_order = []

        self._tarjans_reverse_topological_sort(
            0, marked, done, topological_order)
        return topological_order

    def _tarjans_reverse_topological_sort(self, v, marked, done, topological_order):
        marked[v] = True

        for w in self.adj[v]:
            if not marked[w]:
                self._tarjans_reverse_topological_sort(
                    w, marked, done, topological_order)
            elif not done[w]:
                raise Exception("Cycle Detected")

        topological_order.append(v)
        done[v] = True

    """
    Find path from vertex u to v
    """

    def find_path(self, u, v):
        marked = [False for x in range(0, self.vertex_count)]
        path = []
        path.append(u)

        self._find_path(u, v, marked, path)

    def _find_path(self, u, v, marked, path):
        marked[u] = True

        if u == v:
            print(path)
        else:
            for w in self.adj[u]:
                if not marked[w]:
                    path.append(w)
                    self._find_path(w, v, marked, path)
                    path.pop()

    """Detect a cycle in a undirected graph

    This algorithm detects a cycle in a undirected graph. In other words, we are looking for a back edge
    If a vertex v is discovered/marked and it's not done yet (not post order exited) then this implies that there is
    a back edge and this a cycle

    Since in a undirected graph, all children and parent have edges to each other, we check to make sure that the vertex in
    the adjaceny list is not a parent because the parent will be discovered and not done but this doesn't mean we have a cycle
    This case is only unique to an undirected graph.
    """

    def find_undirected_cycle(self):
        marked = [False for x in range(0, self.vertex_count)]
        done = [False for x in range(0, self.vertex_count)]
        has_cycle = [False]

        for v in range(0, self.vertex_count):
            if not marked[v]:
                self._find_undirected_cycle(v, -1, marked, done, has_cycle)
        return has_cycle

    """
    param v: vertex
    param parent: parent of vertex v
    marked: bool[]
    done: bool[]
    has_cycle: bool[]
    """

    def _find_undirected_cycle(self, v, parent, marked, done, has_cycle):
        # If a cycle has already been detected, just return from the recursive calls
        if not has_cycle[0]:
            marked[v] = True

            for w in self.adj[v]:
                if not marked[w] and w != parent:
                    self._find_undirected_cycle(w, v, marked, done, has_cycle)
                elif not done[w]:
                    has_cycle[0] = True
            done[v] = True

    def find_directed_cycle(self):
        marked = [False for x in range(0, self.vertex_count)]
        done = [False for x in range(0, self.vertex_count)]
        has_cycle = [False]

        for v in range(0, self.vertex_count):
            if not marked[v]:
                self._find_directed_cycle(v, marked, done, has_cycle)
        return has_cycle

    def _find_directed_cycle(self, v, marked, done, has_cycle):
        marked[v] = True

        for w in self.adj[v]:
            if not marked[w]:
                self._find_directed_cycle(w, marked, done, has_cycle)
            elif not done[w]:
                has_cycle[0] = True
        done[v] = True

    def floyd_warshall(self):
        path_cost = self.edge_weights[:]
        paths = [[0 for i in range(self.vertex_count)]
                 for j in range(self.vertex_count)]

        for i in range(0, self.vertex_count):
            for j in range(0, self.vertex_count):
                if path_cost[i][j] == float('inf'):
                    paths[i][j] = -1
                else:
                    paths[i][j] = 0

        k = 0
        while k < self.vertex_count:
            i = 0
            while i < self.vertex_count:
                j = 0
                while j < self.vertex_count:
                    # If cost from vertex i to k + k to j is less than the direct path from i to j then
                    if path_cost[i][j] > path_cost[i][k] + path_cost[k][j]:
                        path_cost[i][j] = path_cost[i][k] + path_cost[k][j]
                        paths[i][j] = k  # Save intermediate vertex from i to j
                    j += 1
                i += 1
            k += 1

        return (paths, path_cost)

    def dijkstra(self, source):
        p_queue = HashHeap()
        parent = {}
        dist = {}

        for v in self.adj.keys():
            p_queue.insert(v, float('inf'))
        p_queue.update_key(source, 0)

        while not p_queue.is_empty():
            min_vertex = p_queue.delete_min()
            current = min_vertex.key
            distance = min_vertex.val

            dist[current] = distance

            for w in self.adj[current]:
                if not p_queue.contains(w):
                    continue

                new_distance = dist[current] + self.edge_weights[current][w]
                if new_distance < p_queue[w].val:
                    p_queue.update_key(w, new_distance)
                    parent[w] = current

        return (dist, parent)

    def shortest_path_from(self, source, dest):
        dist, parent_map = self.dijkstra(source)
        self._shortest_path_from(source, dest, parent_map)

    def _shortest_path_from(self, source, dest, parent_map):
        if dest != source:
            self._shortest_path_from(source, parent_map[dest], parent_map)
        print(dest)

    """
    Count the number of connected components in an undirected graph
    """

    def connected_components(self):
        visited = [False for x in range(0, self.vertex_count)]
        count = 0

        for x in range(0, self.vertex_count):
            if not visited[x]:
                # If graph is fully connected, all vertices will be marked in _connected_components()
                self._connected_components(x, visited)
                # Count a component
                count += 1
        return count

    def _connected_components(self, v, visited):
        visited[v] = True

        for w in self.adj[v]:
            if not visited[w]:
                self._connected_components(w, visited)

    """
    Whenever you examine the node n in a topological ordering, you have the guarantee that you've already traversed every possible path to n.
    Using this it's clear to see that you can generate the shortest path with one linear scan of the topological ordering

    1. Topologically sort G into L;
    2. Set the distance to the source to 0;
    3. Set the distances to all other vertices to infinity;
    4. For each vertex u in L
    5.    - Walk through all neighbors v of u;
    6.    - If dist(v) > dist(u) + w(u, v)
    7.       - Set dist(v) <- dist(u) + w(u, v);
    """

    def dag_shortest_path(self, u):
        topological_sort = self.topological_sort()
        shortest_path = {}
        for v in range(0, self.vertex_count):
            shortest_path[v] = float('inf')
        shortest_path[u] = 0

        for v in topological_sort:
            for w in self.adj[v]:
                if shortest_path[w] > shortest_path[v] + self.edge_weights[v][w]:
                    shortest_path[w] = shortest_path[v] + \
                        self.edge_weights[v][w]
        return second_shortest_path

    """
    Graph Interview Problems
    """

    def __repr__(self):
        return str(self.adj)


graph = Graph(4)

graph.add_directed_edge(0, 1, 1)
graph.add_directed_edge(0, 2, 2)
graph.add_directed_edge(1, 3, 1)
graph.add_directed_edge(2, 3, 3)

print(graph.dag_shortest_path(0))
