import numpy as np


def dynproglin(lang, s_mat, a, b):
    def score(i, j):
        if i == "-":
            return s_mat[-1][lang.index(j)]
        elif j == "-":
            return s_mat[lang.index(i)][-1]
        else:
            return s_mat[lang.index(i)][lang.index(j)]

    def align(b_mat):
        max_res = 0
        end_pos = (0, 0)
        start_pos = (0, 0)
        for i in range(1, len(a) + 1):
            for j in range(1, len(b) + 1):
                b_mat[1, j][0] = max(0,
                                     b_mat[0][j - 1][0] + score(a[i - 1], b[j - 1]),
                                     b_mat[0][j][0] + score(a[i - 1], "-"),
                                     b_mat[1][j - 1][0] + score("-", b[j - 1])
                                     )

                if (b_mat[1, j][0] == b_mat[0][j - 1][0] + score(a[i - 1], b[j - 1])) and \
                        (b_mat[0, j - 1][1] != 0 and b_mat[0, j - 1][2] != 0):
                    b_mat[1, j][1], b_mat[1, j][2] = b_mat[0, j - 1][1], b_mat[0, j - 1][2]
                elif b_mat[1, j][0] == b_mat[0][j][0] + score(a[i - 1], "-") and \
                        (b_mat[0, j][1] != 0 and b_mat[0, j][2] != 0):
                    b_mat[1, j][1], b_mat[1, j][2] = b_mat[0, j][1], b_mat[0, j][2]
                elif b_mat[1, j][0] == b_mat[1][j - 1][0] + score("-", b[j - 1]) and \
                        (b_mat[1, j - 1][1] != 0 and b_mat[1, j - 1][2] != 0):
                    b_mat[1, j][1], b_mat[1, j][2] = b_mat[1, j - 1][1], b_mat[1, j - 1][2]
                elif b_mat[1, j][0] == 0:
                    pass
                else:
                    b_mat[1, j][1], b_mat[1, j][2] = i, j

                if b_mat[1, j][0] > max_res:
                    max_res = b_mat[1, j][0]
                    end_pos = (i, j)
                    start_pos = (b_mat[1, j][1], b_mat[1, j][2])

            # print("Before")
            # print(b_mat)
            b_mat = np.delete(b_mat, 0, 0)
            b_mat = np.row_stack((b_mat, np.zeros((1, len(b) + 1), dtype=(np.int, 3))))
            # print("After")
            # print(b_mat)
        return max_res, start_pos, end_pos

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

    backtrack = np.zeros((2, len(b) + 1), dtype=(np.int, 3))
    # print(backtrack)
    final_score, start_point, end_point = align(backtrack)
    # print(final_score, start_point, end_point)
    return [final_score, start_point, end_point]
    # out_a, out_b = create_alignment(backtrack, ind)
    # return [final_score, out_a, out_b]


if __name__ == '__main__':
    language = "CTGA"
    score_matrix = [[10, -5, -5, -5, -7],
                    [-5, 10, -5, -5, -7],
                    [-5, -5, 10, -5, -7],
                    [-5, -5, -5, 10, -7],
                    [-7, -7, -7, -7, 0]]
    seq_a = "CTCGTC"
    seq_b = "AGCGTAG"
    print(dynproglin(language, score_matrix, seq_a, seq_b))

    language = "TCA"
    score_matrix = [[1, -1, -1, -2],
                    [-1, 1, -1, -2],
                    [-1, -1, 1, -2],
                    [-2, -2, -2, 0]]
    seq_a = "TAATA"
    seq_b = "TACTAA"
    print(dynproglin(language, score_matrix, seq_a, seq_b))

    language = "ABC"
    score_matrix = [[1, -1, -2, -1],
                    [-1, 2, -4, -1],
                    [-2, -4, 3, -2],
                    [-1, -1, -2, 0]]
    seq_a = "ABCACA"
    seq_b = "BAACB"
    print(dynproglin(language, score_matrix, seq_a, seq_b))

    language = "CTGA"
    score_matrix = [[3, -3, -3, -3, -2],
                    [-3, 3, -3, -3, -2],
                    [-3, -3, 3, -3, -2],
                    [-3, -3, -3, 3, -2],
                    [-2, -2, -2, -2, 0]]
    seq_a = "GGTTGACTA"
    seq_b = "TGTTACGG"
    print(dynproglin(language, score_matrix, seq_a, seq_b))

    language = "ABC"
    score_matrix = [[1, -1, -2, -1],
                    [-1, 2, -4, -1],
                    [-2, -4, 3, -2],
                    [-1, -1, -2, 0]]
    seq_a = "AABBAACA"
    seq_b = "CBACCCBA"
    print(dynproglin(language, score_matrix, seq_a, seq_b))

    language = "ABCD"
    score_matrix = [[1, -5, -5, -5, -1],
                    [-5, 1, -5, -5, -1],
                    [-5, -5, 5, -5, -4],
                    [-5, -5, -5, 6, -4],
                    [-1, -1, -4, -4, -9]]
    seq_a = "AAAAACCDDCCDDAAAAACC"
    seq_b = "CCAAADDAAAACCAAADDCCAAAA"
    print([39, [5, 6, 7, 8, 9, 10, 11, 12, 18, 19], [0, 1, 5, 6, 11, 12, 16, 17, 18, 19]] == dynproglin(language,
                                                                                                        score_matrix,
                                                                                                        seq_a, seq_b))

    seq_a = "AACAAADAAAACAADAADAAA"
    seq_b = "CDCDDD"
    print([17, [2, 6, 11, 14, 17], [0, 1, 2, 3, 4]] == dynproglin(language, score_matrix, seq_a, seq_b))

    seq_a = "DDCDDCCCDCAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACCCCDDDCDADCDCDCDCD"
    seq_b = "DDCDDCCCDCBCCCCDDDCDBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBDCDCDCDCD"
    print([81, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 40, 41, 42, 43, 44, 45, 46, 47, 48, 50, 51, 52, 53, 54, 55, 56, 57, 58],
           [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 61, 62, 63, 64, 65, 66, 67, 68, 69]
           ] == dynproglin(language, score_matrix, seq_a, seq_b))
