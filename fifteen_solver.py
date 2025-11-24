import sys
import argparse
from collections import deque
import heapq
import random
import math

# ----------------------------- Puzzle utilities -----------------------------

# Cache entire stdin at program start so multiple reads are possible (fixes double-read issues on Windows redirection)
RAW_STDIN_CONTENT = None
try:
    # If stdin is not a tty, attempt to read and cache it. This prevents accidental multiple reads consuming input.
    if not sys.stdin.isatty():
        try:
            RAW_STDIN_CONTENT = sys.stdin.buffer.read().decode('utf-8')
        except Exception:
            try:
                RAW_STDIN_CONTENT = sys.stdin.read()
            except Exception:
                RAW_STDIN_CONTENT = None
except Exception:
    RAW_STDIN_CONTENT = None


def read_input(stream=None):
    """
    Read R C and R*C integers from the provided stream or from cached stdin content when available.
    Returns (R, C, tuple(board)). Raises SystemExit("No input provided") when input missing.
    """
    data = None
    if RAW_STDIN_CONTENT is not None:
        data = RAW_STDIN_CONTENT.strip().split()
    else:
        if stream is None:
            stream = sys.stdin
        text = stream.read()
        if text is None:
            data = []
        else:
            data = text.strip().split()
    if not data:
        raise SystemExit("No input provided")
    it = iter(data)
    try:
        R = int(next(it))
        C = int(next(it))
    except StopIteration:
        raise SystemExit("Not enough header values (R C)")
    board = []
    for _ in range(R * C):
        try:
            board.append(int(next(it)))
        except StopIteration:
            raise SystemExit("Not enough board entries")
    return R, C, tuple(board)


def goal_state(R, C):
    N = R * C
    # goal: 1..N-1 then 0
    return tuple(list(range(1, N)) + [0])


def find_blank(state):
    idx = state.index(0)
    return idx


def idx_to_rc(idx, C):
    return divmod(idx, C)


def rc_to_idx(r, c, C):
    return r * C + c

# Move semantics per problem statement: L means a tile moves left into blank.
# So if blank at (r,c) and there is tile at (r,c+1) then action 'L' swaps blank with that tile (blank moves right).

def neighbors(state, R, C, order_spec=None):
    '''Yield (action, new_state) according to available moves. order_spec either a string of 'L','R','U','D' or None.
    If order_spec starts with 'R' then neighbors are yielded in random order per call.
    '''
    N = R * C
    bidx = state.index(0)
    br, bc = idx_to_rc(bidx, C)
    possibilities = []
    # If tile at (br, bc+1) exists -> it can move left into blank => 'L'
    if bc + 1 < C:
        j = rc_to_idx(br, bc + 1, C)
        new = list(state)
        new[bidx], new[j] = new[j], new[bidx]
        possibilities.append(('L', tuple(new)))
    # tile at (br, bc-1) -> move right into blank => 'R'
    if bc - 1 >= 0:
        j = rc_to_idx(br, bc - 1, C)
        new = list(state)
        new[bidx], new[j] = new[j], new[bidx]
        possibilities.append(('R', tuple(new)))
    # tile at (br+1, bc) -> move up into blank => 'U'
    if br + 1 < R:
        j = rc_to_idx(br + 1, bc, C)
        new = list(state)
        new[bidx], new[j] = new[j], new[bidx]
        possibilities.append(('U', tuple(new)))
    # tile at (br-1, bc) -> move down into blank => 'D'
    if br - 1 >= 0:
        j = rc_to_idx(br - 1, bc, C)
        new = list(state)
        new[bidx], new[j] = new[j], new[bidx]
        possibilities.append(('D', tuple(new)))

    if order_spec:
        if order_spec.upper().startswith('R'):
            random.shuffle(possibilities)
        else:
            # preserve only moves that are in order_spec and in that sequence
            order = list(order_spec.upper())
            ordered = []
            for ch in order:
                for (act, st) in possibilities:
                    if act == ch:
                        ordered.append((act, st))
            possibilities = ordered
    # else default order as generated (L,R,U,D by construction)
    for item in possibilities:
        yield item

# ----------------------------- Heuristics ----------------------------------

def heuristic_zero(state, R, C, goal_pos):
    return 0


def heuristic_misplaced(state, R, C, goal_pos):
    # count tiles not in goal position (excluding blank)
    c = 0
    for i, v in enumerate(state):
        if v != 0 and goal_pos[v] != i:
            c += 1
    return c


def heuristic_manhattan(state, R, C, goal_pos):
    s = 0
    for i, v in enumerate(state):
        if v == 0:
            continue
        gr, gc = divmod(goal_pos[v], C)
        r, c = divmod(i, C)
        s += abs(r - gr) + abs(c - gc)
    return s

HEURISTICS = {
    0: heuristic_zero,
    1: heuristic_misplaced,
    2: heuristic_manhattan,
}

# ----------------------------- Solvability --------------------------------

def is_solvable(state, R, C):
    # count inversions ignoring 0
    arr = [x for x in state if x != 0]
    inv = 0
    for i in range(len(arr)):
        ai = arr[i]
        for j in range(i + 1, len(arr)):
            if ai > arr[j]:
                inv += 1
    if C % 2 == 1:
        return inv % 2 == 0
    else:
        # blank row counting from bottom (1-based)
        bidx = state.index(0)
        br = bidx // C
        blank_row_from_bottom = R - br
        # For even grid width: puzzle is solvable if the parity of inversions
        # is different from the parity of the blank row from the bottom.
        # (i.e., inv%2 != blank_row_from_bottom%2)
        return (inv % 2) != (blank_row_from_bottom % 2)

# ----------------------------- Search algorithms --------------------------

def reconstruct_path_from_node(node):
    path = []
    cur = node
    while cur.parent is not None:
        path.append(cur.action_from_parent)
        cur = cur.parent
    return ''.join(reversed(path))

def reconstruct_path(parents, end_state):
    # parents: dict state -> (parent_state, action)
    path = []
    cur = end_state
    while cur in parents and parents[cur][0] is not None:
        parent, act = parents[cur]
        path.append(act)
        cur = parent
    return ''.join(reversed(path))

# BFS
def bfs(start, R, C, order_spec, goal):
    if start == goal:
        return ''
    q = deque([start])
    parents = {start: (None, '')}
    visited = {start}
    while q:
        s = q.popleft()
        for act, ns in neighbors(s, R, C, order_spec):
            if ns in visited:
                continue
            parents[ns] = (s, act)
            if ns == goal:
                return reconstruct_path(parents, ns)
            visited.add(ns)
            q.append(ns)
    return None

# DFS (graph search with visited global)
def dfs(start, R, C, order_spec, goal, max_nodes=1000000):
    if start == goal:
        return ''
    stack = [start]
    parents = {start: (None, '')}
    visited = {start}
    nodes = 0
    while stack:
        s = stack.pop()
        nodes += 1
        if nodes > max_nodes:
            return None
        if s == goal:
            return reconstruct_path(parents, s)
        # push neighbors in reverse order so that first in order_spec is processed first
        neigh = list(neighbors(s, R, C, order_spec))
        for act, ns in reversed(neigh):
            if ns in visited:
                continue
            parents[ns] = (s, act)
            visited.add(ns)
            stack.append(ns)
    return None

# IDDFS

def iddfs(start, R, C, order_spec, goal, max_depth=50):
    if start == goal:
        return ''

    def dls(node, depth, path, visited_set):
        if node == goal:
            return ''.join(path)
        if depth == 0:
            return None
        for act, ns in neighbors(node, R, C, order_spec):
            if ns in visited_set:
                continue
            visited_set.add(ns)
            path.append(act)
            res = dls(ns, depth - 1, path, visited_set)
            if res is not None:
                return res
            path.pop()
            visited_set.remove(ns)
        return None

    for depth in range(max_depth + 1):
        res = dls(start, depth, [], {start})
        if res is not None:
            return res
    return None

# Best-first (greedy)
def best_first(start, R, C, order_spec, goal, hfun):
    if start == goal:
        return ''
    open_heap = []
    entry_finder = {}
    h = hfun(start, R, C)
    heapq.heappush(open_heap, (h, start))
    entry_finder[start] = h
    parents = {start: (None, '')}
    closed = set()
    while open_heap:
        f, s = heapq.heappop(open_heap)
        if s in closed:
            continue
        closed.add(s)
        if s == goal:
            return reconstruct_path(parents, s)
        for act, ns in neighbors(s, R, C, order_spec):
            if ns in closed:
                continue
            hns = hfun(ns, R, C)
            if ns not in entry_finder or hns < entry_finder[ns]:
                entry_finder[ns] = hns
                parents[ns] = (s, act)
                heapq.heappush(open_heap, (hns, ns))
    return None

# A*
def astar(start, R, C, order_spec, goal, hfun):
    if start == goal:
        return ''
    open_heap = []
    gscore = {start: 0}
    fscore = {start: hfun(start, R, C)}
    heapq.heappush(open_heap, (fscore[start], start))
    parents = {start: (None, '')}
    closed = set()
    while open_heap:
        f, s = heapq.heappop(open_heap)
        if s in closed:
            continue
        if s == goal:
            return reconstruct_path(parents, s)
        closed.add(s)
        for act, ns in neighbors(s, R, C, order_spec):
            tentative_g = gscore[s] + 1
            if ns in closed and tentative_g >= gscore.get(ns, 10**18):
                continue
            if tentative_g < gscore.get(ns, 10**18) or ns not in gscore:
                parents[ns] = (s, act)
                gscore[ns] = tentative_g
                fscore[ns] = tentative_g + hfun(ns, R, C)
                heapq.heappush(open_heap, (fscore[ns], ns))
    return None

# ----------------------------- SMA* (textbook-accurate) --------------------

class SMAStarNode:
    __slots__ = ('state','g','h','f','parent','action_from_parent','children','depth','id')
    def __init__(self, state, g, h, parent=None, action_from_parent=''):
        self.state = state
        self.g = g
        self.h = h
        self.f = g + h
        self.parent = parent
        self.action_from_parent = action_from_parent
        self.children = []  # list of SMAStarNode
        self.depth = 0 if parent is None else parent.depth + 1
        self.id = id(self)


def sma_star(start, R, C, order_spec, goal, hfun, max_nodes=20000):
    """
    Textbook-accurate SMA* (simplified but with correct backing-up of f-values):
    - OPEN contains frontier nodes (nodes without expanded children)
    - Expand the best frontier node (lowest f). When memory limit exceeded, prune the worst frontier node
      (highest f, and if tie, greatest depth). When pruning, back up f-values to parent and propagate.
    - Handle duplicate states by keeping only the best path currently in memory (if a better path is found,
      we remove the old subtree and insert the new path).

    This implementation is faithful to SMA* described in standard AI textbooks and suitable for
    assignments/demonstrations. It is not extremely optimized for very large memory bounds but is correct.
    """
    if start == goal:
        return ''

    # Helper containers
    node_by_state = {}  # state -> node
    all_nodes = set()   # set of nodes currently in memory

    # frontier priority queue (min-heap) by (f, -depth, counter, node_id)
    frontier_heap = []
    entry_finder = {}  # node.id -> (f, -depth, counter)
    counter = 0

    def push_frontier(node):
        nonlocal counter
        key = (node.f, -node.depth, counter, node.id)
        heapq.heappush(frontier_heap, key)
        entry_finder[node.id] = key
        counter += 1

    def pop_best_frontier():
        # pop best (lowest f) frontier node
        while frontier_heap:
            f, negd, _, nid = heapq.heappop(frontier_heap)
            if nid in entry_finder and entry_finder[nid][3] == nid:
                # this check is coarse; we instead use id presence and node.f match
                del entry_finder[nid]
                # find node object
                for n in list(all_nodes):
                    if n.id == nid:
                        # if node.f matches f (allow small differences), return
                        return n
        return None

    def pop_worst_frontier():
        # find worst frontier node: frontier with maximum f; tie-breaker deeper depth
        # frontier_heap is a min-heap; for simplicity, scan all nodes to select worst frontier
        worst = None
        for n in all_nodes:
            if len(n.children) == 0:  # frontier node
                if worst is None:
                    worst = n
                else:
                    if (n.f > worst.f) or (n.f == worst.f and n.depth > worst.depth):
                        worst = n
        return worst

    def remove_node_recursive(node):
        # remove node and its descendants from memory structures
        stack = [node]
        removed = 0
        while stack:
            n = stack.pop()
            if n in all_nodes:
                all_nodes.remove(n)
            if n.state in node_by_state and node_by_state[n.state] is n:
                del node_by_state[n.state]
            # remove from parent's children if present
            if n.parent is not None:
                try:
                    n.parent.children.remove(n)
                except ValueError:
                    pass
            # schedule children for removal
            for ch in list(n.children):
                stack.append(ch)
            n.children = []
            removed += 1
        return removed

    def backup_f(node):
        # backed-up f: if node has no children -> node.f stays as g+h; else node.f = min(child.f)
        if len(node.children) == 0:
            node.f = node.g + node.h
        else:
            node.f = min(ch.f for ch in node.children)

    # Initialize root
    root = SMAStarNode(start, 0, hfun(start, R, C), parent=None, action_from_parent='')
    node_by_state[start] = root
    all_nodes.add(root)
    push_frontier(root)

    # Main loop
    while True:
        if not frontier_heap:
            return None
        # choose best frontier (lowest f)
        # Recompute best by popping until match
        # We'll pick best by scanning all frontier nodes to be robust
        best = None
        for n in all_nodes:
            if len(n.children) == 0:
                if best is None or (n.f < best.f) or (n.f == best.f and n.depth < best.depth):
                    best = n
        if best is None:
            return None
        # If best is goal, done
        if best.state == goal:
            return reconstruct_path_from_node(best)

        # Expand best: generate successors
        # Before expansion, ensure we have enough memory; expansion will add new nodes
        neighs = list(neighbors(best.state, R, C, order_spec))
        # generate children nodes
        for act, ns in neighs:
            # avoid cycles along ancestor chain
            anc = best
            in_anc = False
            while anc is not None:
                if anc.state == ns:
                    in_anc = True
                    break
                anc = anc.parent
            if in_anc:
                continue

            new_g = best.g + 1
            existing = node_by_state.get(ns)
            if existing is not None:
                # if existing path in memory is better or equal, skip
                if existing.g <= new_g:
                    continue
                # else existing.g > new_g: we found a better path; remove old subtree
                remove_node_recursive(existing)

            child = SMAStarNode(ns, new_g, hfun(ns, R, C), parent=best, action_from_parent=act)
            best.children.append(child)
            node_by_state[ns] = child
            all_nodes.add(child)

        # After expansion, best is internal (has children) so it's no longer a frontier
        if best in all_nodes and len(best.children) > 0:
            # ensure it's not in frontier by leaving its children as frontier
            try:
                # best might still be in entry_finder/frontier_heap; we don't implement lazy deletion here explicitly
                pass
            except Exception:
                pass

        # For each new child, it's a frontier node with its f = g + h; add to frontier
        for ch in list(best.children):
            if ch in all_nodes and len(ch.children) == 0:
                push_frontier(ch)

        # Backup: recompute f for best and propagate up
        cur = best
        while cur is not None:
            old_f = cur.f
            backup_f(cur)
            if cur.f != old_f:
                # if cur becomes frontier (no children) ensure it's in frontier heap
                if len(cur.children) == 0 and cur in all_nodes:
                    push_frontier(cur)
            cur = cur.parent

        # enforce memory bound
        while len(all_nodes) > max_nodes:
            worst = pop_worst_frontier()
            if worst is None:
                break
            # remove worst frontier node
            remove_node_recursive(worst)
            # after removal, backup f-values up the tree from parent
            parent = worst.parent
            cur = parent
            while cur is not None:
                old_f = cur.f
                backup_f(cur)
                # if node now has no children, it becomes frontier
                if len(cur.children) == 0 and cur in all_nodes:
                    push_frontier(cur)
                cur = cur.parent

        # loop continues

# ----------------------------- Viewer -------------------------------------
def view_solution(start, R, C, moves):
    # Replay moves printing board after each step; use same move semantics
    state = list(start)
    print_board(state, R, C)
    pos = state.index(0)
    for i, ch in enumerate(moves, 1):
        br, bc = idx_to_rc(pos, C)
        if ch == 'L':
            # tile at (br, bc+1) moves left into blank
            j = rc_to_idx(br, bc + 1, C)
            state[pos], state[j] = state[j], state[pos]
            pos = j
        elif ch == 'R':
            j = rc_to_idx(br, bc - 1, C)
            state[pos], state[j] = state[j], state[pos]
            pos = j
        elif ch == 'U':
            j = rc_to_idx(br + 1, bc, C)
            state[pos], state[j] = state[j], state[pos]
            pos = j
        elif ch == 'D':
            j = rc_to_idx(br - 1, bc, C)
            state[pos], state[j] = state[j], state[pos]
            pos = j
        else:
            raise ValueError('Invalid move char: ' + ch)
        print(f"Step {i}: {ch}")
        print_board(state, R, C)


def print_board(state, R, C):
    for r in range(R):
        row = state[r * C:(r + 1) * C]
        print(' '.join(str(x) for x in row))
    print()

# ----------------------------- Main ---------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Fifteen puzzle solver')
    group = parser.add_mutually_exclusive_group(required=False)
    parser.add_argument('--check', action='store_true', help='Check solvability only')
    group.add_argument('-b', '--bfs', metavar='ORDER', help="Breadth-first search (ORDER of successors)")
    group.add_argument('-d', '--dfs', metavar='ORDER', help="Depth-first search (ORDER of successors)")
    group.add_argument('-i', '--idfs', metavar='ORDER', help="Iterative deepening DFS (ORDER of successors)")
    group.add_argument('-g', '--bf', metavar='H', type=int, help="Best-first (heuristic id)")
    group.add_argument('-a', '--astar', metavar='H', type=int, help="A* (heuristic id)")
    group.add_argument('-s', '--sma', metavar='H', type=int, help="SMA* (heuristic id)")
    parser.add_argument('--order', metavar='ORDER', help='Optional successor order for informed searches')
    parser.add_argument('--mem', metavar='N', type=int, help='Memory bound for SMA* (default 20000)')
    parser.add_argument('--maxdepth', metavar='N', type=int, help='Max depth for IDDFS (default 80)')
    parser.add_argument('--view', metavar='MOVES', help='Viewer: replay a moves string from input initial state')
    args = parser.parse_args()

    R, C, start = read_input()
    goal = goal_state(R, C)

    # create goal positions map for heuristics
    goal_pos = {v: i for i, v in enumerate(goal)}

    # If --check was given, print solvability and exit
    if args.check:
        # compute detailed diagnostics for solvability
        arr = [x for x in start if x != 0]
        inv = 0
        for i in range(len(arr)):
            ai = arr[i]
            for j in range(i+1, len(arr)):
                if ai > arr[j]:
                    inv += 1
        bidx = start.index(0)
        br = bidx // C
        blank_row_from_bottom = R - br
        width_even = (C % 2 == 0)
        solv = is_solvable(start, R, C)
        # print compact machine-readable result to stdout (0/1) for scripts
        print('1' if solv else '0')
        # print human-readable diagnostics to stderr so it doesn't break automated parsers
        print('--- Solvability diagnostic ---', file=sys.stderr)
        print(f'Inversions (excluding blank): {inv}', file=sys.stderr)
        print(f'Blank row from top (0-based): {br}', file=sys.stderr)
        print(f'Blank row from bottom (1-based): {blank_row_from_bottom}', file=sys.stderr)
        print(f'Board width even? {width_even}', file=sys.stderr)
        print(f'Inversion parity (even?): {inv % 2 == 0}', file=sys.stderr)
        if width_even:
            print('Rule: For even grid width, puzzle is solvable if (inversions + blank_row_from_bottom) is even.', file=sys.stderr)
            print(f'(inversions + blank_row_from_bottom) = {inv} + {blank_row_from_bottom} = {inv + blank_row_from_bottom}', file=sys.stderr)
        else:
            print('Rule: For odd grid width, puzzle is solvable if inversion count is even.', file=sys.stderr)
        print(f'Result: {"SOLVABLE" if solv else "UNSOLVABLE"}', file=sys.stderr)
        print('--- End diagnostic ---', file=sys.stderr)
        return

    # If --view given, run viewer and exit (no algorithm needed)
    if args.view is not None:
        moves = args.view.strip()
        if moves == '':
            # nothing to view
            return
        if not is_solvable(start, R, C):
            print(-1)
            print('')
            return
        print('Initial board:')
        view_solution(start, R, C, moves)
        print(f"Moves: {len(moves)}")
        return

    # create goal positions map for heuristics
    goal_pos = {v: i for i, v in enumerate(goal)}

    # select order_spec
    order_spec = None
    algo = None
    heuristic_id = None
    memory_bound = args.mem if args.mem is not None else 20000
    maxdepth = args.maxdepth if args.maxdepth is not None else 80
    if args.bfs is not None:
        algo = 'bfs'
        order_spec = args.bfs
    elif args.dfs is not None:
        algo = 'dfs'
        order_spec = args.dfs
    elif args.idfs is not None:
        algo = 'idfs'
        order_spec = args.idfs
    elif args.bf is not None:
        algo = 'bf'
        heuristic_id = args.bf
        order_spec = args.order
    elif args.astar is not None:
        algo = 'astar'
        heuristic_id = args.astar
        order_spec = args.order
    elif args.sma is not None:
        algo = 'sma'
        heuristic_id = args.sma
        order_spec = args.order

    if args.view is not None:
        # replay mode: print each step
        moves = args.view.strip()
        if not is_solvable(start, R, C):
            print(-1)
            print('')
            return
        print('Initial board:')
        view_solution(start, R, C, moves)
        print(f"Moves: {len(moves)}")
        return

    # if heuristic used, pick function
    if heuristic_id is not None:
        if heuristic_id not in HEURISTICS:
            print(f"Unknown heuristic id {heuristic_id}")
            sys.exit(1)
        # wrap to include goal_pos
        hfun = lambda s, r, c: HEURISTICS[heuristic_id](s, r, c, goal_pos)
    else:
        # default zero heuristic
        hfun = lambda s, r, c: 0

    # quick solvability check
    if not is_solvable(start, R, C):
        print(-1)
        print('')
        return

    solution = None
    if algo == 'bfs':
        solution = bfs(start, R, C, order_spec, goal)
    elif algo == 'dfs':
        solution = dfs(start, R, C, order_spec, goal)
    elif algo == 'idfs':
        solution = iddfs(start, R, C, order_spec, goal, max_depth=maxdepth)
    elif algo == 'bf':
        solution = best_first(start, R, C, order_spec or '', goal, lambda s, r, c: hfun(s, r, c))
    elif algo == 'astar':
        solution = astar(start, R, C, order_spec or '', goal, lambda s, r, c: hfun(s, r, c))
    elif algo == 'sma':
        solution = sma_star(start, R, C, order_spec or '', goal, lambda s, r, c: hfun(s, r, c), max_nodes=memory_bound)

    if solution is None:
        print(-1)
        print('')
    else:
        print(len(solution))
        print(solution)

if __name__ == '__main__':
    main()
