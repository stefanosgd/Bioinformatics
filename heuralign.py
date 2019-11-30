import numpy as np


def heuralign(lang, s_mat, a, b):

    def score(i, j):
        if i == "-":
            return s_mat[-1][lang.index(j)]
        elif j == "-":
            return s_mat[lang.index(i)][-1]
        else:
            return s_mat[lang.index(i)][lang.index(j)]

    def align(diagonal, b_mat, width):
        if diagonal > 0:
            start = (1, abs(diagonal) + 1)
        else:
            start = (abs(diagonal) + 1, 1)

        for i in range(max(start[1] - width, 1), len(a) + 1):
            for j in range(max(start[0] - width, 1), len(b) + 1):
                if ((j < start[0] + width + (i - start[1])) and (i < start[1] + width + (j - start[0]))) or (
                        (j < start[0] + width + (i - start[1])) and (j > start[0] - width + (i - start[1]))) or (
                        (i < start[1] + width + (j - start[0])) and (i > start[1] - width + (j - start[0]))):
                    b_mat[i, j] = max(0,
                                      b_mat[i - 1][j - 1] + score(a[i - 1], b[j - 1]),
                                      b_mat[i - 1][j] + score(a[i - 1], "-"),
                                      b_mat[i][j - 1] + score("-", b[j - 1])
                                      )
        return b_mat

    def get_index(k):
        index_table = {}
        for i in range(len(a) - (k - 1)):
            word = a[i:i + k]
            if word not in index_table:
                index_table[word] = [i]
            else:
                index_table[word].append(i)
        return index_table

    def get_diagonals(index_table, k):
        diagonals = {}
        for in_a in range(len(b) - (k - 1)):
            word = b[in_a:in_a + k]
            if word in index_table:
                for in_b in index_table[word]:
                    try:
                        diagonals[(in_b - in_a)].append((in_a, in_b))
                    except:
                        diagonals[(in_b - in_a)] = [(in_a, in_b)]
        return diagonals

    def score_diagonals(diagonal_index):
        min_score = -2
        diagonal_score = {}
        for offset in diagonal_index:
            scoring = 0
            current_max = np.NINF
            coords = diagonal_index[offset]
            if len(coords) >= 1:
                for i in range(len(coords)-1):
                    diff = coords[i + 1][0] - coords[i][0]
                    if diff == 1:
                        scoring += 1
                    elif (scoring - diff) <= min_score:
                        if scoring > current_max:
                            current_max = scoring
                        scoring = 0
                    else:
                        scoring -= diff

                if scoring > current_max:
                    current_max = scoring
                diagonal_score[offset] = current_max
        return diagonal_score

    def create_alignment(b_mat, start):
        prev_a = start[0]
        prev_b = start[1]
        end = False
        output = [[], []]
        while not end:
            if b_mat[prev_a][prev_b] == b_mat[prev_a - 1][prev_b - 1] + score(a[prev_a - 1], b[prev_b - 1]):
                prev_a -= 1
                prev_b -= 1
                output[0].insert(0, prev_a)
                output[1].insert(0, prev_b)
            elif b_mat[prev_a][prev_b] == b_mat[prev_a - 1][prev_b] + score(a[prev_a - 1], "-"):
                prev_a -= 1
            elif b_mat[prev_a][prev_b] == b_mat[prev_a][prev_b - 1] + score("-", b[prev_b - 1]):
                prev_b -= 1
            elif b_mat[prev_a][prev_b] == 0:
                end = True
        return output

    if len(a) == 0 or len(b) == 0:
        return [0, [], []]
    seed_length = 4
    index = get_index(seed_length)
    diag_index = get_diagonals(index, seed_length)
    diag_score = score_diagonals(diag_index)
    while len(diag_index) < 4 and seed_length > 0:
        seed_length -= 1
        index = get_index(seed_length)
        diag_index = get_diagonals(index, seed_length)
        diag_score = score_diagonals(diag_index)

    max_diagonals = []
    for diagonal in range(min(len(diag_score)-1, 5)):
        max_diagonals.append(max(diag_score, key=diag_score.get))
        diag_score.pop(max_diagonals[diagonal])

    max_diag = (min(max_diagonals) + max(max_diagonals)) // 2
    band_width = max(abs(max(max_diagonals) - max_diag) + 5, abs(min(max_diagonals) - max_diag) + 5)

    backtrack = np.zeros((len(a) + 1, len(b) + 1), dtype=np.int)
    backtrack = align(max_diag, backtrack, band_width)
    ind = np.unravel_index(np.argmax(backtrack, axis=None), backtrack.shape)
    final_score = backtrack[ind]
    out_a, out_b = create_alignment(backtrack, ind)
    return [final_score, out_a, out_b]


if __name__ == '__main__':
    language = "CTGA"
    score_matrix = [[10, -5, -5, -5, -7],
                    [-5, 10, -5, -5, -7],
                    [-5, -5, 10, -5, -7],
                    [-5, -5, -5, 10, -7],
                    [-7, -7, -7, -7, 0]]
    seq_a = "CTCGTC"
    seq_b = "AGCGTAG"
    print(heuralign(language, score_matrix, seq_a, seq_b))

    language = "TCA"
    score_matrix = [[1, -1, -1, -2],
                    [-1, 1, -1, -2],
                    [-1, -1, 1, -2],
                    [-2, -2, -2, 0]]
    seq_a = "TAATA"
    seq_b = "TACTAA"
    print(heuralign(language, score_matrix, seq_a, seq_b))

    language = "ABC"
    score_matrix = [[1, -1, -2, -1],
                    [-1, 2, -4, -1],
                    [-2, -4, 3, -2],
                    [-1, -1, -2, 0]]
    seq_a = "ABCACA"
    seq_b = "BAACB"
    print(heuralign(language, score_matrix, seq_a, seq_b))

    language = "CTGA"
    score_matrix = [[3, -3, -3, -3, -2],
                    [-3, 3, -3, -3, -2],
                    [-3, -3, 3, -3, -2],
                    [-3, -3, -3, 3, -2],
                    [-2, -2, -2, -2, 0]]
    seq_a = "GGTTGACTA"
    seq_b = "TGTTACGG"
    print(heuralign(language, score_matrix, seq_a, seq_b))

    language = "ABC"
    score_matrix = [[1, -1, -2, -1],
                    [-1, 2, -4, -1],
                    [-2, -4, 3, -2],
                    [-1, -1, -2, 0]]
    seq_a = "AABBAACA"
    seq_b = "CBACCCBA"
    print(heuralign(language, score_matrix, seq_a, seq_b))

    language = "ABCD"
    score_matrix = [[1, -5, -5, -5, -1],
                    [-5, 1, -5, -5, -1],
                    [-5, -5, 5, -5, -4],
                    [-5, -5, -5, 6, -4],
                    [-1, -1, -4, -4, -9]]

    seq_a = "AAAAACCDDCCDDAAAAACC"
    seq_b = "CCAAADDAAAACCAAADDCCAAAA"
    print([39, [5, 6, 7, 8, 9, 10, 11, 12, 18, 19], [0, 1, 5, 6, 11, 12, 16, 17, 18, 19]] == heuralign(language,
                                                                                                       score_matrix,
                                                                                                       seq_a, seq_b))

    seq_a = "AACAAADAAAACAADAADAAA"
    seq_b = "CDCDDD"
    print([17, [2, 6, 11, 14, 17], [0, 1, 2, 3, 4]] == heuralign(language, score_matrix, seq_a, seq_b))

    seq_a = "DDCDDCCCDCAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACCCCDDDCDADCDCDCDCD"
    seq_b = "DDCDDCCCDCBCCCCDDDCDBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBDCDCDCDCD"
    print([81, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 40, 41, 42, 43, 44, 45, 46, 47, 48, 50, 51, 52, 53, 54, 55, 56, 57, 58],
           [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 61, 62, 63, 64, 65, 66, 67, 68, 69]
           ] == heuralign(language, score_matrix, seq_a, seq_b))
