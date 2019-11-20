import numpy as np


def dynproglin(lang, s_mat, a, b):
    s1 = np.zeros(len(b) + 1, dtype=np.int)
    s2 = np.zeros(len(b) + 1, dtype=np.int)

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
            b_mat = np.delete(b_mat, 0, 0)
            b_mat = np.row_stack((b_mat, np.zeros((1, len(b) + 1), dtype=(np.int, 3))))
        return max_res, start_pos, end_pos

    def path(start_a, end_a, start_b, end_b):
        output = [[], []]
        if start_a == end_a or start_b == end_b:
            return output
        b_mat = np.zeros((2, 2), dtype=np.int)
        b_mat[1,0] = b_mat[0, 0] + score(a[start_a], "-")
        b_mat[0,1] = b_mat[0, 0] + score("-", b[start_b])
        b_mat[1, 1] = max(b_mat[0][0] + score(a[start_a], b[start_b]),
                          b_mat[0][1] + score(a[start_a], "-"),
                          b_mat[1][0] + score("-", b[start_b])
                          )
        prev_a, prev_b = np.unravel_index(np.argmax(b_mat, axis=None), b_mat.shape)
        print(b_mat)
        print(prev_a, prev_b)
        while True:
            if prev_a == 0 and prev_b == 0:
                break
            if b_mat[prev_a][prev_b] == b_mat[prev_a-1][prev_b-1] + score(a[start_a +prev_a-1], b[start_b +prev_b-1]):
                prev_a -= 1
                prev_b -= 1
                output[0].insert(0, prev_a+start_a)
                output[1].insert(0, prev_b+start_b)
            elif b_mat[prev_a][prev_b] == b_mat[prev_a][prev_b-1] + score("-", b[start_b +prev_b-1]):
                prev_b -= 1
            elif b_mat[prev_a][prev_b] == b_mat[prev_a-1][prev_b] + score(a[start_a +prev_a-1], "-"):
                prev_a -= 1
        print(output)
        return output

    def create_alignment(start_i, start_j, end_i, end_j):
        output = [[], []]
        print(start_i, start_j, end_i, end_j)
        string_a = a[start_i:end_i]
        string_b = b[start_j:end_j]
        print(string_a, string_b)
        if start_i + 1 == end_i or start_j + 1 == end_j:
            out_1, out_2 = path(start_i, end_i, start_j, end_j)
            output[0].extend(out_1)
            output[1].extend(out_2)
            # print(output)
        else:
            mid = (start_i + end_i) // 2
            s1[start_j] = 0
            # print("Mid", mid)
            for j in range(start_j + 1, end_j):
                s1[j] = s1[j - 1] + score("-", b[j])
            for i in range(start_i + 1, mid):
                s = s1[start_j]
                c = s1[start_j] + score(a[i], "-")
                s1[start_j] = c
                for j in range(start_j + 1, end_j):
                    c = max(s1[j] + score(a[i], "-"),
                            s + score(a[i], b[j]),
                            c + score("-", b[j]))
                    s = s1[j]
                    s1[j] = c
            s2[end_j] = 0
            for j in range(end_j - 1, start_j, -1):
                s2[j] = s2[j + 1] + score("-", b[j+1])
            for i in range(end_i - 1, mid, -1):
                # print("i = ", i)
                s = s2[end_j]
                c = s2[end_j] + score(a[i+1], "-")
                s2[end_j] = s2[end_j] + score(a[i+1], "-")
                for j in range(end_j - 1, start_j, -1):
                    c = max(s2[j] + score(a[i+1], "-"),
                            s + score(a[i+1], b[j+1]),
                            c + score("-", b[j+1]))
                    s = s2[j]
                    s2[j] = c
            print(s1)
            print(s2)

            j = mid
            maximised = False
            # maximised = max(s1[start_j] + s2[start_j], s1[end_j] + s2[end_j])
            # if maximised == s1[start_j] + s2[start_j]:
            #     j = start_j
            # else:
            #     j = end_j
            # for n in range(start_j, end_j):
            #     if maximised == False:
            #         j = n
            #         maximised = s1[n] + s2[n]
            #     elif s1[n] + s2[n] > maximised:
            #         j = n
            #         maximised = s1[n] + s2[n]

            prev_out_a, prev_out_b = create_alignment(start_i, start_j, mid, j)
            output[0].extend(prev_out_a)
            output[1].extend(prev_out_b)
            # print("First", prev_out_a, prev_out_b)
            prev_out_a, prev_out_b = create_alignment(mid, j, end_i, end_j)
            output[0].extend(prev_out_a)
            output[1].extend(prev_out_b)
            # print("Second", prev_out_a, prev_out_b)
            # print(output)
        return output

    backtrack = np.zeros((2, len(b) + 1), dtype=(np.int, 3))
    final_score, start_point, end_point = align(backtrack)
    # print(final_score, start_point, end_point)
    # print(start_point[0], start_point[1], end_point[0], end_point[1])
    # print(a[start_point[0]-1:end_point[0]])
    # print(b[start_point[1]-1:end_point[1]])
    out_a, out_b = create_alignment(start_point[0]-1, start_point[1]-1, end_point[0], end_point[1])
    # out_a.append(end_point[0]-1)
    # out_b.append(end_point[1]-1)
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
    print(dynproglin(language, score_matrix, seq_a, seq_b))
    exit()
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
    # print(dynproglin(language, score_matrix, seq_a, seq_b))
    #
    # language = "CTGA"
    # score_matrix = [[3, -3, -3, -3, -2],
    #                 [-3, 3, -3, -3, -2],
    #                 [-3, -3, 3, -3, -2],
    #                 [-3, -3, -3, 3, -2],
    #                 [-2, -2, -2, -2, 0]]
    # seq_a = "GGTTGACTA"
    # seq_b = "TGTTACGG"
    # print(dynproglin(language, score_matrix, seq_a, seq_b))
    #
    # language = "ABC"
    # score_matrix = [[1, -1, -2, -1],
    #                 [-1, 2, -4, -1],
    #                 [-2, -4, 3, -2],
    #                 [-1, -1, -2, 0]]
    # seq_a = "AABBAACA"
    # seq_b = "CBACCCBA"
    # print(dynproglin(language, score_matrix, seq_a, seq_b))
    #
    # language = "ABCD"
    # score_matrix = [[1, -5, -5, -5, -1],
    #                 [-5, 1, -5, -5, -1],
    #                 [-5, -5, 5, -5, -4],
    #                 [-5, -5, -5, 6, -4],
    #                 [-1, -1, -4, -4, -9]]
    #
    # seq_a = "AAAAACCDDCCDDAAAAACC"
    # seq_b = "CCAAADDAAAACCAAADDCCAAAA"
    # print(dynproglin(language, score_matrix, seq_a, seq_b))
    # print([39, [5, 6, 7, 8, 9, 10, 11, 12, 18, 19], [0, 1, 5, 6, 11, 12, 16, 17, 18, 19]] == dynproglin(language,
    #                                                                                                     score_matrix,
    #                                                                                                     seq_a, seq_b))
    #
    # seq_a = "AACAAADAAAACAADAADAAA"
    # seq_b = "CDCDDD"
    # print(dynproglin(language, score_matrix, seq_a, seq_b))
    # print([17, [2, 6, 11, 14, 17], [0, 1, 2, 3, 4]] == dynproglin(language, score_matrix, seq_a, seq_b))
    #
    # seq_a = "DDCDDCCCDCAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACCCCDDDCDADCDCDCDCD"
    # seq_b = "DDCDDCCCDCBCCCCDDDCDBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBDCDCDCDCD"
    # print(dynproglin(language, score_matrix, seq_a, seq_b))
    # print([81, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 40, 41, 42, 43, 44, 45, 46, 47, 48, 50, 51, 52, 53, 54, 55, 56, 57, 58],
    #        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 61, 62, 63, 64, 65, 66, 67, 68, 69]
    #        ] == dynproglin(language, score_matrix, seq_a, seq_b))
