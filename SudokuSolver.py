import numpy as np
import itertools

class SudokuSolver:
    def __init__(self, board):
        self.board = np.array(board)
        self.size = 9
        self.subgrid_size = 3
        self.candidates = np.array([[set(range(1, 10)) if self.board[i][j] == 0 else set() for j in range(self.size)] for i in range(self.size)])

    def print_board(self):
        for row in self.board:
            print(" ".join(str(num) if num != 0 else "." for num in row))
        print()

    def is_valid(self, num, pos):
        row, col = pos
        # Check row
        if any(self.board[row][i] == num for i in range(self.size) if i != col):
            return False

        # Check column
        if any(self.board[i][col] == num for i in range(self.size) if i != row):
            return False

        # Check box
        box_row, box_col = row // 3, col // 3
        for i in range(box_row * 3, box_row * 3 + 3):
            for j in range(box_col * 3, box_col * 3 + 3):
                if self.board[i][j] == num and (i, j) != pos:
                    return False

        return True

    def is_safe(self, row, col, num):
        if num in self.board[row, :]:
            return False
        if num in self.board[:, col]:
            return False
        start_row, start_col = row - row % self.subgrid_size, col - col % self.subgrid_size
        if num in self.board[start_row:start_row + self.subgrid_size, start_col:start_col + self.subgrid_size]:
            return False
        return True

    def find_empty(self):
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i][j] == 0:
                    return i, j
        return None


    def backtrack_brute_force(self):
        empty_location = self.find_empty()
        if not empty_location:
            return True  # Solved
        row, col = empty_location

        for num in range(1,10):
            if self.is_safe(row, col, num):
                self.board[row][col] = num
                if self.backtrack_brute_force():
                    return True
                self.board[row][col] = 0

        return False

    def naked_single(self):
        changed = False
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i][j] == 0:
                    possibilities = self.get_possibilities(i, j)
                    if len(possibilities) == 1:
                        self.board[i][j] = possibilities.pop()
                        changed = True
        return changed

    def get_possibilities(self, row, col):
        possibilities = set(range(1, 10))
        for i in range(self.size):
            possibilities.discard(self.board[row][i])
            possibilities.discard(self.board[i][col])
        box_row, box_col = row // 3 * 3, col // 3 * 3
        for i in range(box_row, box_row + 3):
            for j in range(box_col, box_col + 3):
                possibilities.discard(self.board[i][j])
        return possibilities

    def hidden_single(self):
        changed = False
        for digit in range(1, 10):
            for i in range(self.size):
                # Check rows
                row_possibilities = [j for j in range(self.size) if self.board[i][j] == 0 and digit in self.get_possibilities(i, j)]
                if len(row_possibilities) == 1:
                    self.board[i][row_possibilities[0]] = digit
                    changed = True
                
                # Check columns
                col_possibilities = [j for j in range(self.size) if self.board[j][i] == 0 and digit in self.get_possibilities(j, i)]
                if len(col_possibilities) == 1:
                    self.board[col_possibilities[0]][i] = digit
                    changed = True
                
                # Check boxes
                box_row, box_col = (i // 3) * 3, (i % 3) * 3
                box_possibilities = [(r, c) for r in range(box_row, box_row + 3) for c in range(box_col, box_col + 3) if self.board[r][c] == 0 and digit in self.get_possibilities(r, c)]
                if len(box_possibilities) == 1:
                    self.board[box_possibilities[0][0]][box_possibilities[0][1]] = digit
                    changed = True
        return changed

    def naked_pair_triple_quad(self):
        changed = False
        for unit in self.get_all_units():
            # Collect all unsolved cells and their possibilities in this unit
            unsolved_cells = [(i, j) for i, j in unit if self.board[i][j] == 0]
            cell_possibilities = {cell: self.get_possibilities(cell[0], cell[1]) for cell in unsolved_cells}

            # Check for naked pairs, triples, and quads
            for size in range(2, 5):
                # Collect groups of cells with the same possibilities of length 'size'
                groups = [cell for cell in unsolved_cells if len(cell_possibilities[cell]) == size]
                # Find sets of cells that share the same possibilities
                for cell_group in groups:
                    matching_cells = [cell for cell in groups if cell_possibilities[cell] == cell_possibilities[cell_group]]
                    if len(matching_cells) == size:
                        # Found a naked pair, triple, or quad
                        naked_set = cell_possibilities[cell_group]
                        for cell in unsolved_cells:
                            if cell not in matching_cells:
                                # Remove naked set candidates from other cells in the unit
                                if cell_possibilities[cell].intersection(naked_set):
                                    cell_possibilities[cell] -= naked_set
                                    changed = True
            # Update candidates
            for cell, candidates in cell_possibilities.items():
                self.candidates[cell[0]][cell[1]] = candidates
        return changed

    def get_all_units(self):
        rows = [[(i, j) for j in range(self.size)] for i in range(self.size)]
        cols = [[(i, j) for i in range(self.size)] for j in range(self.size)]
        boxes = [[(i + r, j + c) for r in range(3) for c in range(3)] for i in range(0, self.size, 3) for j in range(0, self.size, 3)]
        return rows + cols + boxes

    def update_candidates(self, row, col, num):
        if num in self.candidates[row][col]:
            self.candidates[row][col].discard(num)
            return True
        return False
    
    def x_wing(self):
        changes = False
        for num in range(1, 10):
            row_positions = [[] for _ in range(self.size)]
            col_positions = [[] for _ in range(self.size)]
            
            for r in range(self.size):
                for c in range(self.size):
                    if self.board[r][c] == 0 and self.is_safe(r, c, num):
                        row_positions[r].append(c)
                        col_positions[c].append(r)
            
            for r1 in range(self.size):
                for r2 in range(r1 + 1, self.size):
                    if len(row_positions[r1]) == 2 and row_positions[r1] == row_positions[r2]:
                        c1, c2 = row_positions[r1]
                        for r in range(self.size):
                            if r != r1 and r != r2:
                                if self.update_candidates(r, c1, num):
                                    changes = True
                                if self.update_candidates(r, c2, num):
                                    changes = True
                        
            for c1 in range(self.size):
                for c2 in range(c1 + 1, self.size):
                    if len(col_positions[c1]) == 2 and col_positions[c1] == col_positions[c2]:
                        r1, r2 = col_positions[c1]
                        for c in range(self.size):
                            if c != c1 and c != c2:
                                if self.update_candidates(r1, c, num):
                                    changes = True
                                if self.update_candidates(r2, c, num):
                                    changes = True
        
        return changes

    def swordfish(self):
        changes = False
        for num in range(1, 10):
            row_positions = [[] for _ in range(self.size)]
            col_positions = [[] for _ in range(self.size)]
            
            for r in range(self.size):
                for c in range(self.size):
                    if self.board[r][c] == 0 and self.is_safe(r, c, num):
                        row_positions[r].append(c)
                        col_positions[c].append(r)
            
            for r1 in range(self.size):
                for r2 in range(r1 + 1, self.size):
                    for r3 in range(r2 + 1, self.size):
                        if (len(row_positions[r1]) == 2 or len(row_positions[r1]) == 3) and \
                           (len(row_positions[r2]) == 2 or len(row_positions[r2]) == 3) and \
                           (len(row_positions[r3]) == 2 or len(row_positions[r3]) == 3):
                            common_columns = set(row_positions[r1]) & set(row_positions[r2]) & set(row_positions[r3])
                            if len(common_columns) == 3:
                                for r in range(self.size):
                                    if r != r1 and r != r2 and r != r3:
                                        for c in common_columns:
                                            if self.board[r][c] == 0 and self.is_safe(r, c, num):
                                                self.board[r][c] = -num
                                                changes = True
                                            
            for c1 in range(self.size):
                for c2 in range(c1 + 1, self.size):
                    for c3 in range(c2 + 1, self.size):
                        if (len(col_positions[c1]) == 2 or len(col_positions[c1]) == 3) and \
                           (len(col_positions[c2]) == 2 or len(col_positions[c2]) == 3) and \
                           (len(col_positions[c3]) == 2 or len(col_positions[c3]) == 3):
                            common_rows = set(col_positions[c1]) & set(col_positions[c2]) & set(col_positions[c3])
                            if len(common_rows) == 3:
                                for c in range(self.size):
                                    if c != c1 and c != c2 and c != c3:
                                        for r in common_rows:
                                            if self.board[r][c] == 0 and self.is_safe(r, c, num):
                                                self.board[r][c] = -num
                                                changes = True
                                            
        return changes

    def xy_wing(self):
        changes = False
        for r in range(self.size):
            for c in range(self.size):
                if self.board[r][c] == 0:
                    candidates = [num for num in range(1, 10) if self.is_valid(num, (r, c))]
                    if len(candidates) == 2:
                        r1, r2 = candidates
                        for r2 in range(self.size):
                            for c2 in range(self.size):
                                if r2 != r and c2 != c and self.board[r2][c2] == 0:
                                    candidates2 = [num for num in range(1, 10) if self.is_valid(num, (r2, c2))]
                                    if len(candidates2) == 2 and len(set(candidates + candidates2)) == 3:
                                        for r3 in range(self.size):
                                            for c3 in range(self.size):
                                                if r3 != r and r3 != r2 and c3 != c and c3 != c2 and self.board[r3][c3] == 0:
                                                    candidates3 = [num for num in range(1, 10) if self.is_valid(num, (r3, c3))]
                                                    if len(candidates3) == 2 and len(set(candidates + candidates2 + candidates3)) == 3:
                                                        for num in candidates:
                                                            if num in candidates2 and num in candidates3:
                                                                if self.board[r][c] == 0:
                                                                    self.board[r][c] = -num
                                                                    changes = True
                                                                if self.board[r2][c2] == 0:
                                                                    self.board[r2][c2] = -num
                                                                    changes = True
                                                                if self.board[r3][c3] == 0:
                                                                    self.board[r3][c3] = -num
                                                                    changes = True
        return changes

    def hidden_pair_triple_quad(self):
        def find_hidden_sets(unit, size):
            count = {num: [] for num in range(1, 10)}
            for cell in unit:
                if self.board[cell[0]][cell[1]] == 0:
                    for num in self.candidates[cell[0]][cell[1]]:
                        count[num].append(cell)
            # Filter to get only those numbers that appear exactly 'size' times
            return {num: cells for num, cells in count.items() if len(cells) == size}

        def eliminate_other_candidates(unit, hidden_set, size):
            changed = False
            hidden_nums = set(hidden_set.keys())
            cells = [cell for cell_list in hidden_set.values() for cell in cell_list]
            for cell in cells:
                current_candidates = self.candidates[cell[0]][cell[1]]
                if not hidden_nums.issubset(current_candidates):
                    continue
                if len(current_candidates) > size:
                    self.candidates[cell[0]][cell[1]] = hidden_nums
                    changed = True
            return changed

        changed = False
        for unit in self.get_all_units():
            for size in range(2, 5):
                hidden_sets = find_hidden_sets(unit, size)
                if len(hidden_sets) >= size:
                    for combination in itertools.combinations(hidden_sets.keys(), size):
                        selected_cells = set(cell for num in combination for cell in hidden_sets[num])
                        if len(selected_cells) == size:
                            hidden_set = {num: hidden_sets[num] for num in combination}
                            if eliminate_other_candidates(unit, hidden_set, size):
                                changed = True
        return changed

    def box_line_reduction(self):
        def find_box_line_interactions():
            interactions = []
            for num in range(1, 10):
                for box_row in range(0, self.size, 3):
                    for box_col in range(0, self.size, 3):
                        rows = [set() for _ in range(3)]
                        cols = [set() for _ in range(3)]
                        
                        # Collect positions of the number in the current box
                        for i in range(3):
                            for j in range(3):
                                r, c = box_row + i, box_col + j
                                if self.board[r][c] == 0 and num in self.candidates[r][c]:
                                    rows[i].add((r, c))
                                    cols[j].add((r, c))
                        
                        # Check if the number is confined to a single row within the box
                        for i in range(3):
                            if len(rows[i]) > 1 and all(self.board[r][c] == 0 for r, c in rows[i]):
                                box_line = rows[i]
                                row_index = box_row + i
                                row_cells = {(row_index, col) for col in range(self.size) if col // 3 != box_col // 3}
                                interactions.append((num, box_line, row_cells))
                        
                        # Check if the number is confined to a single column within the box
                        for j in range(3):
                            if len(cols[j]) > 1 and all(self.board[r][c] == 0 for r, c in cols[j]):
                                box_line = cols[j]
                                col_index = box_col + j
                                col_cells = {(row, col_index) for row in range(self.size) if row // 3 != box_row // 3}
                                interactions.append((num, box_line, col_cells))
            return interactions

        changes = False
        interactions = find_box_line_interactions()

        for num, box_line, line_cells in interactions:
            for r, c in line_cells:
                if num in self.candidates[r][c]:
                    self.candidates[r][c].remove(num)
                    changes = True

        return changes

    def jellyfish(self):
        def find_jellyfish(candidates):
            jellyfish_cells = []
            for num in range(1, 10):
                positions = [[] for _ in range(self.size)]
                
                # Find all positions for each number in rows
                for r in range(self.size):
                    for c in range(self.size):
                        if self.board[r][c] == 0 and num in self.candidates[r][c]:
                            positions[r].append(c)
                
                # Check for jellyfish pattern in rows
                for r1 in range(self.size):
                    for r2 in range(r1 + 1, self.size):
                        for r3 in range(r2 + 1, self.size):
                            for r4 in range(r3 + 1, self.size):
                                common_cols = set(positions[r1]) & set(positions[r2]) & set(positions[r3]) & set(positions[r4])
                                if len(common_cols) == 4:
                                    jellyfish_cells.append((num, [(r1, c) for c in common_cols] +
                                                                [(r2, c) for c in common_cols] +
                                                                [(r3, c) for c in common_cols] +
                                                                [(r4, c) for c in common_cols]))
            
            for num in range(1, 10):
                positions = [[] for _ in range(self.size)]
                
                # Find all positions for each number in columns
                for c in range(self.size):
                    for r in range(self.size):
                        if self.board[r][c] == 0 and num in self.candidates[r][c]:
                            positions[c].append(r)
                
                # Check for jellyfish pattern in columns
                for c1 in range(self.size):
                    for c2 in range(c1 + 1, self.size):
                        for c3 in range(c2 + 1, self.size):
                            for c4 in range(c3 + 1, self.size):
                                common_rows = set(positions[c1]) & set(positions[c2]) & set(positions[c3]) & set(positions[c4])
                                if len(common_rows) == 4:
                                    jellyfish_cells.append((num, [(r, c1) for r in common_rows] +
                                                                [(r, c2) for r in common_rows] +
                                                                [(r, c3) for r in common_rows] +
                                                                [(r, c4) for r in common_rows]))
            return jellyfish_cells

        changes = False
        jellyfish_cells = find_jellyfish(self.candidates)

        for num, cells in jellyfish_cells:
            for r, c in cells:
                if num in self.candidates[r][c]:
                    self.candidates[r][c].remove(num)
                    changes = True

        return changes

    def unique_rectangles(self):
        def find_unique_rectangles():
            unique_rectangles = []
            for r1 in range(self.size):
                for c1 in range(self.size):
                    if len(self.candidates[r1][c1]) == 2:
                        for r2 in range(r1 + 1, self.size):
                            for c2 in range(c1 + 1, self.size):
                                if len(self.candidates[r2][c2]) == 2 and self.candidates[r1][c1] == self.candidates[r2][c2]:
                                    if (len(self.candidates[r1][c2]) == 2 and self.candidates[r1][c2] == self.candidates[r1][c1]) and \
                                    (len(self.candidates[r2][c1]) == 2 and self.candidates[r2][c1] == self.candidates[r1][c1]):
                                        unique_rectangles.append(((r1, c1), (r1, c2), (r2, c1), (r2, c2)))
            return unique_rectangles

        changes = False
        unique_rectangles = find_unique_rectangles()

        for ur in unique_rectangles:
            cells = [ur[0], ur[1], ur[2], ur[3]]
            candidates = self.candidates[ur[0][0]][ur[0][1]]
            for cell in cells:
                r, c = cell
                if len(self.candidates[r][c]) > 2:
                    for candidate in candidates:
                        if candidate in self.candidates[r][c]:
                            self.candidates[r][c].remove(candidate)
                            changes = True

        return changes

    def chains(self):
        def find_chains():
            chains = []
            for num in range(1, 10):
                # Collect all cells where the candidate number is present
                cells_with_num = [(r, c) for r in range(self.size) for c in range(self.size) if self.board[r][c] == 0 and num in self.candidates[r][c]]
                for cell in cells_with_num:
                    chain = explore_chain(cell, num, cells_with_num, set())
                    if chain:
                        chains.append((num, chain))
            return chains

        def explore_chain(cell, num, cells_with_num, visited):
            r, c = cell
            visited.add(cell)
            chain = [cell]
            
            related_cells = self.get_related_cells(r, c)
            for related in related_cells:
                if related in visited:
                    continue
                rr, cc = related
                if num in self.candidates[rr][cc]:
                    chain.append(related)
                    if len(chain) % 2 == 0:  # Even-length chain segment
                        further_chain = explore_chain(related, num, cells_with_num, visited)
                        if further_chain:
                            chain.extend(further_chain)
                            return chain
            return chain if len(chain) > 1 else None

        def apply_chains(chains):
            changes = False
            for num, chain in chains:
                for i in range(0, len(chain), 2):
                    if i + 1 < len(chain):
                        cell1 = chain[i]
                        cell2 = chain[i + 1]
                        related_cells = set(self.get_related_cells(cell1[0], cell1[1])) & set(self.get_related_cells(cell2[0], cell2[1]))
                        for r, c in related_cells:
                            if num in self.candidates[r][c]:
                                self.candidates[r][c].remove(num)
                                changes = True
            return changes


        changes = False
        chains = find_chains()

        if apply_chains(chains):
            changes = True

        return changes
    
    def get_related_cells(self, r, c):
        related = set()
        related.update((r, i) for i in range(self.size))
        related.update((i, c) for i in range(self.size))
        box_r, box_c = r // 3 * 3, c // 3 * 3
        related.update((box_r + i, box_c + j) for i in range(3) for j in range(3))
        related.remove((r, c))
        return related

    def forcing_chains_nets(self):
        def find_forcing_chains():
            chains = []
            for num in range(1, 10):
                cells_with_num = [(r, c) for r in range(self.size) for c in range(self.size) if self.board[r][c] == 0 and num in self.candidates[r][c]]
                for cell in cells_with_num:
                    chain = explore_forcing_chain(cell, num, set())
                    if chain:
                        chains.append((num, chain))
            return chains

        def explore_forcing_chain(cell, num, visited):
            r, c = cell
            visited.add(cell)
            chain = [cell]
            
            related_cells = self.get_related_cells(r, c)
            for related in related_cells:
                if related in visited:
                    continue
                rr, cc = related
                if num in self.candidates[rr][cc]:
                    chain.append(related)
                    further_chain = explore_forcing_chain(related, num, visited)
                    if further_chain:
                        chain.extend(further_chain)
                        return chain
            return chain if len(chain) > 1 else None

        def apply_forcing_chains(chains):
            changes = False
            for num, chain in chains:
                possible_eliminations = set()
                for cell in chain:
                    related_cells = self.get_related_cells(cell[0], cell[1])
                    for related in related_cells:
                        if related not in chain and num in self.candidates[related[0]][related[1]]:
                            possible_eliminations.add(related)
                for r, c in possible_eliminations:
                    if num in self.candidates[r][c]:
                        self.candidates[r][c].remove(num)
                        changes = True
            return changes

        changes = False
        chains = find_forcing_chains()

        if apply_forcing_chains(chains):
            changes = True

        return changes


    def nishio(self):
        def is_unique_solution():
            # Make a copy of the current board and solve it with backtracking
            board_copy = np.copy(self.board)
            if self.backtrack_brute_force():
                solution = np.copy(self.board)
                self.board = board_copy
                if self.backtrack_brute_force():
                    return np.array_equal(solution, self.board)
                self.board = board_copy
            return False
        
        changes = False
        for r in range(self.size):
            for c in range(self.size):
                if self.board[r][c] == 0:
                    original_candidates = self.candidates[r][c]
                    for num in list(original_candidates):
                        # Try placing the number and check if it leads to a contradiction
                        self.board[r][c] = num
                        self.candidates[r][c].clear()
                        if not is_unique_solution():
                            original_candidates.remove(num)
                            changes = True
                        self.board[r][c] = 0
                        self.candidates[r][c] = original_candidates
        return changes

    def coloring(self):
        def find_color_chains():
            color_chains = []
            for num in range(1, 10):
                cells_with_num = [(r, c) for r in range(self.size) for c in range(self.size) if self.board[r][c] == 0 and num in self.candidates[r][c]]
                colors = {}
                color_count = 0
                for cell in cells_with_num:
                    if cell not in colors:
                        color_count += 1
                        if color_count > 2:
                            break
                        explore_color_chain(cell, num, colors, color_count)
                if color_count == 2:
                    color_chains.append((num, colors))
            return color_chains

        def explore_color_chain(cell, num, colors, color):
            stack = [cell]
            while stack:
                current = stack.pop()
                if current in colors:
                    continue
                colors[current] = color
                related_cells = self.get_related_cells(current[0], current[1])
                for related in related_cells:
                    if related not in colors and num in self.candidates[related[0]][related[1]]:
                        stack.append(related)

        def apply_coloring(color_chains):
            changes = False
            for num, colors in color_chains:
                color1_cells = [cell for cell, color in colors.items() if color == 1]
                color2_cells = [cell for cell, color in colors.items() if color == 2]

                # If two cells in the same unit have the same color, we can eliminate the candidate from these cells
                for cell1 in color1_cells:
                    for cell2 in color1_cells:
                        if cell1 != cell2 and cell1 in self.get_related_cells(cell2[0], cell2[1]):
                            changes = eliminate_candidate(num, color1_cells, color2_cells)
                            break
                    if changes:
                        break

                if not changes:
                    for cell1 in color2_cells:
                        for cell2 in color2_cells:
                            if cell1 != cell2 and cell1 in self.get_related_cells(cell2[0], cell2[1]):
                                changes = eliminate_candidate(num, color2_cells, color1_cells)
                                break
                        if changes:
                            break
            return changes

        def eliminate_candidate(num, primary_color_cells, secondary_color_cells):
            changes = False
            for cell in primary_color_cells:
                if num in self.candidates[cell[0]][cell[1]]:
                    self.candidates[cell[0]][cell[1]].remove(num)
                    changes = True
            for cell in secondary_color_cells:
                if num in self.candidates[cell[0]][cell[1]]:
                    self.candidates[cell[0]][cell[1]].remove(num)
                    changes = True
            return changes

        changes = False
        color_chains = find_color_chains()
        if apply_coloring(color_chains):
            changes = True
        return changes

    def multi_coloring(self):
        def find_multi_color_chains():
            multi_color_chains = []
            for num in range(1, 10):
                cells_with_num = [(r, c) for r in range(self.size) for c in range(self.size) if self.board[r][c] == 0 and num in self.candidates[r][c]]
                colors = {}
                color_count = 0
                for cell in cells_with_num:
                    if cell not in colors:
                        color_count += 1
                        explore_color_chain(cell, num, colors, color_count)
                if color_count > 2:
                    multi_color_chains.append((num, colors))
            return multi_color_chains

        def explore_color_chain(cell, num, colors, color):
            stack = [cell]
            while stack:
                current = stack.pop()
                if current in colors:
                    continue
                colors[current] = color
                related_cells = self.get_related_cells(current[0], current[1])
                for related in related_cells:
                    if related not in colors and num in self.candidates[related[0]][related[1]]:
                        stack.append(related)

        def apply_multi_coloring(multi_color_chains):
            changes = False
            for num, colors in multi_color_chains:
                color_groups = {}
                for cell, color in colors.items():
                    if color not in color_groups:
                        color_groups[color] = []
                    color_groups[color].append(cell)

                for primary_color, primary_cells in color_groups.items():
                    for secondary_color, secondary_cells in color_groups.items():
                        if primary_color != secondary_color:
                            changes = eliminate_candidate(num, primary_cells, secondary_cells)
                            if changes:
                                break
                    if changes:
                        break
            return changes

        def eliminate_candidate(num, primary_color_cells, secondary_color_cells):
            changes = False
            for cell in primary_color_cells:
                if num in self.candidates[cell[0]][cell[1]]:
                    self.candidates[cell[0]][cell[1]].remove(num)
                    changes = True
            for cell in secondary_color_cells:
                if num in self.candidates[cell[0]][cell[1]]:
                    self.candidates[cell[0]][cell[1]].remove(num)
                    changes = True
            return changes

        changes = False
        multi_color_chains = find_multi_color_chains()
        if apply_multi_coloring(multi_color_chains):
            changes = True
        return changes

    def wxyz_wing(self):
        def find_wxyz_wings():
            wxyz_wings = []
            for r in range(self.size):
                for c in range(self.size):
                    if len(self.candidates[r][c]) == 4:
                        for r1, c1 in self.get_related_cells(r, c):
                            if len(self.candidates[r1][c1]) == 2:
                                for r2, c2 in self.get_related_cells(r, c):
                                    if (r2, c2) != (r1, c1) and len(self.candidates[r2][c2]) == 2:
                                        common = self.candidates[r][c].intersection(self.candidates[r1][c1], self.candidates[r2][c2])
                                        if len(common) == 1:
                                            wxyz_wings.append((r, c, r1, c1, r2, c2))
            return wxyz_wings

        def apply_wxyz_wings(wxyz_wings):
            changes = False
            for r, c, r1, c1, r2, c2 in wxyz_wings:
                common = self.candidates[r][c].intersection(self.candidates[r1][c1], self.candidates[r2][c2])
                common_num = list(common)[0]
                related_cells = set(self.get_related_cells(r, c)) & set(self.get_related_cells(r1, c1)) & set(self.get_related_cells(r2, c2))
                for rr, cc in related_cells:
                    if common_num in self.candidates[rr][cc]:
                        self.candidates[rr][cc].remove(common_num)
                        changes = True
            return changes

        def get_related_cells(self, r, c):
            related = set()
            related.update((r, i) for i in range(self.size))
            related.update((i, c) for i in range(self.size))
            box_r, box_c = r // 3 * 3, c // 3 * 3
            related.update((box_r + i, box_c + j) for i in range(3) for j in range(3))
            related.remove((r, c))
            return related

        changes = False
        wxyz_wings = find_wxyz_wings()

        if apply_wxyz_wings(wxyz_wings):
            changes = True

        return changes

    def finned_x_wing_swordfish_jellyfish(self):
        def find_finned_fish(fish_type, size):
            finned_fish = []
            for num in range(1, 10):
                positions = [[] for _ in range(self.size)]

                # Collect positions of the number for the given fish type (rows or columns)
                for r in range(self.size):
                    for c in range(self.size):
                        if self.board[r][c] == 0 and num in self.candidates[r][c]:
                            if fish_type == 'row':
                                positions[r].append(c)
                            else:
                                positions[c].append(r)

                # Find potential fish patterns
                for indexes in itertools.combinations(range(self.size), size):
                    common_positions = [positions[idx] for idx in indexes]
                    all_positions = set(pos for sublist in common_positions for pos in sublist)
                    if len(all_positions) == size + 1:
                        for pos in all_positions:
                            if sum(1 for sublist in common_positions if pos in sublist) == size:
                                finned_fish.append((num, indexes, all_positions, pos))
            return finned_fish

        def apply_finned_fish(finned_fish, fish_type):
            changes = False
            for num, indexes, all_positions, fin in finned_fish:
                if fish_type == 'row':
                    for r in indexes:
                        for c in range(self.size):
                            if c not in all_positions and num in self.candidates[r][c]:
                                self.candidates[r][c].remove(num)
                                changes = True
                    for r in range(self.size):
                        if r not in indexes and fin in all_positions:
                            if num in self.candidates[r][fin]:
                                self.candidates[r][fin].remove(num)
                                changes = True
                else:
                    for c in indexes:
                        for r in range(self.size):
                            if r not in all_positions and num in self.candidates[r][c]:
                                self.candidates[r][c].remove(num)
                                changes = True
                    for c in range(self.size):
                        if c not in indexes and fin in all_positions:
                            if num in self.candidates[fin][c]:
                                self.candidates[fin][c].remove(num)
                                changes = True
            return changes

        changes = False
        for fish_type, size in [('row', 2), ('row', 3), ('row', 4), ('col', 2), ('col', 3), ('col', 4)]:
            finned_fish = find_finned_fish(fish_type, size)
            if apply_finned_fish(finned_fish, fish_type):
                changes = True

        return changes

    def candidate_lines(self):
        def find_candidate_lines():
            interactions = []
            for num in range(1, 10):
                for box_row in range(0, self.size, 3):
                    for box_col in range(0, self.size, 3):
                        row_interactions = [set() for _ in range(3)]
                        col_interactions = [set() for _ in range(3)]
                        
                        # Collect positions of the number in the current box
                        for i in range(3):
                            for j in range(3):
                                r, c = box_row + i, box_col + j
                                if self.board[r][c] == 0 and num in self.candidates[r][c]:
                                    row_interactions[i].add((r, c))
                                    col_interactions[j].add((r, c))
                        
                        # Check for candidate lines in rows
                        for i in range(3):
                            if len(row_interactions[i]) > 1 and all(c // 3 == box_col // 3 for r, c in row_interactions[i]):
                                row_index = box_row + i
                                row_cells = {(row_index, col) for col in range(self.size) if col // 3 != box_col // 3}
                                interactions.append((num, row_interactions[i], row_cells))
                        
                        # Check for candidate lines in columns
                        for j in range(3):
                            if len(col_interactions[j]) > 1 and all(r // 3 == box_row // 3 for r, c in col_interactions[j]):
                                col_index = box_col + j
                                col_cells = {(row, col_index) for row in range(self.size) if row // 3 != box_row // 3}
                                interactions.append((num, col_interactions[j], col_cells))
            return interactions

        changes = False
        interactions = find_candidate_lines()

        for num, box_line, line_cells in interactions:
            for r, c in line_cells:
                if num in self.candidates[r][c]:
                    self.candidates[r][c].remove(num)
                    changes = True

        return changes

    def apply_strategies(self):
        while True:
            changes = False

            if self.naked_single(): 
                changes = True
                print("naked_single")
            if self.hidden_single(): 
                changes = True
                print("hidden_single")
            if self.candidate_lines(): 
                changes = True
                print("candidate_lines")
            if self.x_wing(): 
                changes = True
                print("x_wing")
            if self.xy_wing(): 
                changes = True
                print("xy_wing")
            if self.swordfish(): 
                changes = True
                print("swordfish")
            if self.unique_rectangles(): 
                changes = True
                print("unique_rectangles")
            if not changes and self.hidden_pair_triple_quad(): 
                changes = True
                print("hidden_pair_triple_quad")
            if not changes and self.box_line_reduction(): 
                changes = True
                print("box_line_reduction")
            if not changes and self.finned_x_wing_swordfish_jellyfish(): 
                changes = True
                print("finned_x_wing_swordfish_jellyfish")
            if not changes and self.jellyfish(): 
                changes = True
                print("jellyfish")
            #if not changes and self.naked_pair_triple_quad(): 
            # changes = True

            if not changes and self.coloring(): 
                changes = True
                print("coloring")
            if not changes and self.multi_coloring(): 
                changes = True
                print("multi_coloring")
            if not changes and self.wxyz_wing(): 
                changes = True
                print("wxyz_wing")
            if not changes and self.chains(): 
                changes = True
                print("chains")
            if not changes and self.forcing_chains_nets(): 
                changes = True
                print("forcing_chains_nets")
            if not changes and self.nishio(): 
                changes = True
                print("nishio")
            if not changes: 
                break

    def solve(self):
        self.apply_strategies()
        self.print_board()
        print("\nApplying Brute Force")
        if not self.backtrack_brute_force():
            print("No solution exists")
            self.print_board()
        else:
            self.print_board()

if __name__ == "__main__":
    board = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 3, 0, 8, 5],
        [0, 0, 1, 0, 2, 0, 0, 0, 0],
        [0, 0, 0, 5, 0, 7, 0, 0, 0],
        [0, 0, 4, 0, 0, 0, 1, 0, 0],
        [0, 9, 0, 0, 0, 0, 0, 0, 0],
        [5, 0, 0, 0, 0, 0, 0, 7, 3],
        [0, 2, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 4, 0, 0, 0, 9]
    ]
    board2 = [
        [8, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 3, 6, 0, 0, 0, 0, 0],
        [0, 7, 0, 0, 9, 0, 2, 0, 0],
        [0, 5, 0, 0, 0, 7, 0, 0, 0],
        [0, 0, 0, 0, 4, 5, 7, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 3, 0],
        [5, 0, 1, 0, 0, 0, 0, 6, 8],
        [0, 0, 8, 5, 0, 0, 0, 1, 0],
        [0, 9, 0, 0, 0, 0, 4, 0, 0]
    ]

    solver = SudokuSolver(board)
    print("Initial Board:")
    solver.print_board()
    print("Solving...\n")
    solver.solve()

    solver2 = SudokuSolver(board2)
    print("Initial Board:")
    solver2.print_board()
    print("Solving...\n")
    solver2.solve()

