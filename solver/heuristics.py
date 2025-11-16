import random


def naive_search(board):
    empty_cells = [(r, c) for r in range(9) for c in range(9) if board[r][c] == 0]
    if not empty_cells:
        return None
    return random.choice(empty_cells)

def hybrid_mrv(board, domainStore):
    """
    Hybrid MRV + Degree heuristic:
    - MRV: choose the empty cell with the smallest remaining domain.
    - Degree: break ties by choosing the cell that constrains the most other empty cells.
    """
    min_domain_size = 10
    candidates = []

    for r in range(9):
        for c in range(9):
            if board[r][c] == 0:
                domain_size = sum(domainStore[r][c])
                if domain_size < min_domain_size:
                    min_domain_size = domain_size
                    candidates = [(r, c)]
                elif domain_size == min_domain_size:
                    candidates.append((r, c))

    if not candidates:
        return None

    best_cell = None
    max_degree = -1
    for (r, c) in candidates:
        degree = 0

        degree += sum(1 for cc in range(9) if board[r][cc] == 0 and cc != c)
        degree += sum(1 for rr in range(9) if board[rr][c] == 0 and rr != r)
        sr, sc = 3 * (r // 3), 3 * (c // 3)
        degree += sum(
            1
            for rr in range(sr, sr + 3)
            for cc in range(sc, sc + 3)
            if board[rr][cc] == 0 and (rr, cc) != (r, c)
        )

        if degree > max_degree:
            max_degree = degree
            best_cell = (r, c)

    return best_cell