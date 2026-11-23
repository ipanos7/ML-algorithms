#  (C) Copyright Wieger Wesselink 2021. Distributed under the GPL-3.0-or-later
#  Software License, (See accompanying file LICENSE or copy at
#  https://www.gnu.org/licenses/gpl-3.0.txt)


from competitive_sudoku.sudoku import GameState, Move, SudokuBoard
import competitive_sudoku.sudokuai

# Points for completing 0, 1, 2 or 3 regions with a single move.
POINTS = (0, 1, 3, 7)
INF = float('inf')


class SudokuAI(competitive_sudoku.sudokuai.SudokuAI):
    """
    Sudoku AI that computes a move for a given sudoku configuration.
    """

    def __init__(self):
        super().__init__()

    def compute_best_move(self, game_state: GameState) -> None:
        board = game_state.board
        N, m, n = board.N, board.m, board.n   # blocks are m rows x n columns
        EMPTY = SudokuBoard.empty

        
        # Static precomputation (cells are flat indices k = row * N + col)
        
        rows = [[r * N + c for c in range(N)] for r in range(N)]
        cols = [[r * N + c for r in range(N)] for c in range(N)]
        blocks = [[(br + i) * N + (bc + j) for i in range(m) for j in range(n)]
                  for br in range(0, N, m) for bc in range(0, N, n)]

        regions_of = [[] for _ in range(N * N)]   # the 3 regions of each cell
        for region in rows + cols + blocks:
            for k in region:
                regions_of[k].append(region)

        # peers[k]: all other cells sharing a row, column or block with k
        peers = [tuple(set(q for region in regions_of[k] for q in region) - {k})
                 for k in range(N * N)]

        # neighbours[k]: the (up to) 8 surrounding cells, incl. diagonals
        neighbours = []
        for k in range(N * N):
            r, c = divmod(k, N)
            neighbours.append(tuple((r + dr) * N + (c + dc)
                                    for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                                    if (dr or dc) and 0 <= r + dr < N and 0 <= c + dc < N))

        
        # Mutable search state (changed by do/undo during the search)
        
        cells = list(board.squares)
        taboo = {(board.square2index(t.square), t.value) for t in game_state.taboo_moves}

        def to_set(squares):
            return None if squares is None else {board.square2index(s) for s in squares}

        # Starting regions per player (None = every cell is allowed, i.e. classic mode)
        base = {1: to_set(game_state.allowed_squares1), 2: to_set(game_state.allowed_squares2)}
        occupied = {1: list(board.square2index(s) for s in (game_state.occupied_squares1 or [])),
                    2: list(board.square2index(s) for s in (game_state.occupied_squares2 or []))}

        me = game_state.current_player
        opponent = 3 - me
        score = {1: game_state.scores[0], 2: game_state.scores[1]}
        empties = [cells.count(EMPTY)]   # list, so the nested functions can update it

        
        # Move generation
        
        def allowed_cells(player):
            """Empty cells where `player` may play right now."""
            if base[player] is None:
                return [k for k in range(N * N) if cells[k] == EMPTY]
            result = {k for k in base[player] if cells[k] == EMPTY}
            for k in occupied[player]:
                for q in neighbours[k]:
                    if cells[q] == EMPTY:
                        result.add(q)

            return sorted(result)

        def legal_moves(player):
            """All (cell, value) pairs that respect C0, are allowed and are not taboo."""
            moves = []
            for k in allowed_cells(player):
                used = {cells[q] for q in peers[k]}
                for v in range(1, N + 1):
                    if v not in used and (k, v) not in taboo:
                        moves.append((k, v))
            return moves

        def move_points(k):
            """Points earned by filling (empty) cell k: count the regions it completes."""
            completed = 0
            for region in regions_of[k]:
                if all(cells[q] != EMPTY for q in region if q != k):
                    completed += 1
            return POINTS[completed]

        def ordered(moves):
            """Scoring moves first: this makes alpha-beta prune much more."""
            return sorted(moves, key=lambda mv: move_points(mv[0]), reverse=True)

        
        # Evaluation function (from the point of view of `me`)
        
        def evaluate():
            return score[me] - score[opponent]

        
        # Minimax with alpha-beta pruning
        
        def minimax(depth, alpha, beta, player, passed):
            if depth == 0 or empties[0] == 0:
                return evaluate()

            moves = legal_moves(player)
            if not moves:
                # The player cannot move so the turn goes to the other player.
                # If the other player also could not move, the game is over.
                # A pass does not fill a cell, so it does not use up search depth.
                if passed:
                    return evaluate()
                return minimax(depth, alpha, beta, 3 - player, True)

            maximizing = (player == me)
            best = -INF if maximizing else INF

            for k, v in ordered(moves):
                # do the move
                gained = move_points(k)
                cells[k] = v
                occupied[player].append(k)
                score[player] += gained
                empties[0] -= 1

                value = minimax(depth - 1, alpha, beta, 3 - player, False)

                # undo the move
                empties[0] += 1
                score[player] -= gained
                occupied[player].pop()
                cells[k] = EMPTY

                if maximizing:
                    best = max(best, value)
                    alpha = max(alpha, best)
                else:
                    best = min(best, value)
                    beta = min(beta, best)
                if alpha >= beta:
                    break   # prune the remaining moves
            return best

        def propose(k, v):
            self.propose_move(Move(board.index2square(k), v))

        
        # Root: propose something immediately, then doiterative deepening
        
        all_values = set(range(1, N + 1))

        def obviously_bad(k, v):
            """True if, after playing v in k, (a) some empty cell has no candidate value
            left, or (b) some region has a missing value that fits in none of its empty
            cells. Such a move is certainly declared taboo by the oracle (lost turn).
            This is a cheap one-step consistency check, not a sudoku solver."""
            # Only cells/regions touched by the move can change, so only those are checked.
            cells[k] = v
            try:
                candidates = {}

                def cand(q):
                    if q not in candidates:
                        candidates[q] = all_values - {cells[p] for p in peers[q]}
                    return candidates[q]

                # every empty peer of k must keep at least one candidate
                for q in peers[k]:
                    if cells[q] == EMPTY and not cand(q):
                        return True
                # regions of k: every missing value still needs a place
                for region in regions_of[k]:
                    missing = all_values - {cells[q] for q in region}
                    for value in missing:
                        if not any(value in cand(q) for q in region if cells[q] == EMPTY):
                            return True
                # other regions that contain a peer of k: value v still needs a place
                touched = {id(r): r for q in peers[k] for r in regions_of[q]}
                for region in touched.values():
                    if k in region:
                        continue
                    if all(cells[q] != v for q in region):
                        if not any(v in cand(q) for q in region if cells[q] == EMPTY):
                            return True
                return False
            finally:
                cells[k] = EMPTY

        all_root_moves = ordered(legal_moves(me))

        if not all_root_moves:
            # No move respects C0. Propose any non-taboo value in an allowed cell,
            # so that we at least propose something.
            for k in allowed_cells(me):
                for v in range(1, N + 1):
                    if (k, v) not in taboo:
                        propose(k, v)
                        return
            return

        propose(*all_root_moves[0])   # propose something right away, before any heavy work

        # Depth 1: filter out obviously bad moves one by one, and propose every
        # improvement immediately (so we stay "anytime" even on very large boards).
        root_moves = []
        best_value = -INF
        for k, v in all_root_moves:
            if obviously_bad(k, v):
                continue
            root_moves.append((k, v))
            value = evaluate() + move_points(k)
            if value > best_value:
                best_value = value
                propose(k, v)
        if not root_moves:
            root_moves = all_root_moves   # every move looks bad: keep searching all of them

        depth = 2
        while depth <= empties[0]:
            values = {}
            best_value, best_move = -INF, None
            alpha = -INF
            for k, v in root_moves:
                gained = move_points(k)
                cells[k] = v
                occupied[me].append(k)
                score[me] += gained
                empties[0] -= 1 

                value = minimax(depth - 1, alpha, INF, opponent, False)

                empties[0] += 1
                score[me] -= gained
                occupied[me].pop()
                cells[k] = EMPTY

                values[(k, v)] = value
                if value > best_value:
                    best_value, best_move = value, (k, v)
                    alpha = max(alpha, value)

            propose(*best_move)   # this depth is fully searched: commit to it

            # Search the most promising moves first in the next iteration
            root_moves.sort(key=lambda mv: values[mv], reverse=True)
            root_moves.remove(best_move)
            root_moves.insert(0, best_move)
            depth += 1