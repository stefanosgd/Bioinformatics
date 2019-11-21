import numpy as np


def dynproglin(lang, s_mat, a, b):

    def score(i, j):
        if i == "-":
            return s_mat[-1][lang.index(j)]
        elif j == "-":
            return s_mat[lang.index(i)][-1]
        else:
            return s_mat[lang.index(i)][lang.index(j)]

    def align():
        b_mat = np.zeros((2, len(b) + 1), dtype=(np.int, 3))
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
            b_mat = np.delete(b_mat, 0, 0)
            b_mat = np.row_stack((b_mat, np.zeros((1, len(b) + 1), dtype=(np.int, 3))))
        return max_res, start_pos, end_pos

    def path(match_a, match_b):
        output = [[], []]
        if len(match_a) == 0 or len(match_b) == 0:
            return output
        b_mat = np.zeros((len(match_a) + 1, len(match_b) + 1), dtype=np.int)
        for i in range(1, len(match_a) + 1):
            b_mat[i, 0] = b_mat[i-1, 0] + score(match_a[i-1], "-")
        for i in range(1, len(match_b) + 1):
            b_mat[0, i] = b_mat[0, i-1] + score("-", match_b[i-1])
        for i in range(1, len(match_a) + 1):
            for j in range(1, len(match_b) + 1):
                b_mat[i, j] = max(b_mat[i-1][j-1] + score(match_a[i-1], match_b[j-1]),
                                  b_mat[i-1][j] + score(match_a[i-1], "-"),
                                  b_mat[i][j-1] + score("-", match_b[j-1])
                                  )
        prev_a, prev_b = b_mat.shape[0] - 1, b_mat.shape[1] - 1

        while True:
            if prev_a == 0 and prev_b == 0:
                break
            if b_mat[prev_a][prev_b] == b_mat[prev_a-1][prev_b-1] + score(match_a[prev_a-1], match_b[prev_b-1]):
                prev_a -= 1
                prev_b -= 1
                output[0].insert(0, prev_a)
                output[1].insert(0, prev_b)
            elif b_mat[prev_a][prev_b] == b_mat[prev_a][prev_b-1] + score("-", match_b[prev_b-1]):
                prev_b -= 1
            elif b_mat[prev_a][prev_b] == b_mat[prev_a-1][prev_b] + score(match_a[prev_a-1], "-"):
                prev_a -= 1
        return output

    def prefix_suffix(string_a, string_b):
        b_mat = np.zeros((2, len(string_b) + 1), dtype=np.int)
        for i in range(1, len(string_b) + 1):
            b_mat[0, i] = b_mat[0][i - 1] + score("-", string_b[i - 1])
        for i in range(1, len(string_a) + 1):
            b_mat[1, 0] = b_mat[0][0] + score(string_a[i-1], "-")
            for j in range(1, len(string_b) + 1):
                b_mat[1, j] = max(b_mat[0][j - 1] + score(string_a[i - 1], string_b[j - 1]),
                                  b_mat[0][j] + score(string_a[i - 1], "-"),
                                  b_mat[1][j - 1] + score("-", string_b[j - 1])
                                  )
            b_mat = np.delete(b_mat, 0, 0)
            b_mat = np.row_stack((b_mat, np.zeros((1, len(string_b) + 1), dtype=np.int)))
        return b_mat[0]

    def create_alignment(string_a, string_b):
        output = [[], []]
        start_i = 0
        start_j = 0
        end_i = len(string_a)
        end_j = len(string_b)
        if start_i + 2 >= end_i or start_j + 2 >= end_j:
            out_1, out_2 = path(string_a, string_b)
            output[0].extend(out_1)
            output[1].extend(out_2)
        else:
            mid = (start_i + end_i) // 2
            pref = prefix_suffix(string_a[:mid], string_b)
            suff = prefix_suffix(string_a[:mid-1:-1], string_b[::-1])[::-1]
            j = 0
            maximum = np.NINF
            for n in range(len(suff)):
                if maximum < suff[n] + pref[n]:
                    maximum = suff[n] + pref[n]
                    j = n

            prev_out_a, prev_out_b = create_alignment(string_a[:mid], string_b[:j])
            output[0].extend(prev_out_a)
            output[1].extend(prev_out_b)
            prev_out_a, prev_out_b = create_alignment(string_a[mid:], string_b[j:])
            output[0].extend([x+mid for x in prev_out_a])
            output[1].extend([x+j for x in prev_out_b])
        return output

    final_score, start_point, end_point = align()
    out_a, out_b = create_alignment(a[start_point[0]-1:end_point[0]], b[start_point[1]-1:end_point[1]])
    return [final_score, [x+start_point[0]-1 for x in out_a], [x+start_point[1]-1 for x in out_b]]


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
    # print(dynproglin(language, score_matrix, seq_a, seq_b))
    print([39, [5, 6, 7, 8, 9, 10, 11, 12, 18, 19], [0, 1, 5, 6, 11, 12, 16, 17, 18, 19]] == dynproglin(language,
                                                                                                        score_matrix,
                                                                                                        seq_a, seq_b))
    #
    seq_a = "AACAAADAAAACAADAADAAA"
    seq_b = "CDCDDD"
    # print(dynproglin(language, score_matrix, seq_a, seq_b))
    print([17, [2, 6, 11, 14, 17], [0, 1, 2, 3, 4]] == dynproglin(language, score_matrix, seq_a, seq_b))

    seq_a = "DDCDDCCCDCAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACCCCDDDCDADCDCDCDCD"
    seq_b = "DDCDDCCCDCBCCCCDDDCDBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBDCDCDCDCD"
    # print(dynproglin(language, score_matrix, seq_a, seq_b))
    print([81, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 40, 41, 42, 43, 44, 45, 46, 47, 48, 50, 51, 52, 53, 54, 55, 56, 57, 58],
           [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 61, 62, 63, 64, 65, 66, 67, 68, 69]
           ] == dynproglin(language, score_matrix, seq_a, seq_b))
