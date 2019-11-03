import numpy as np


def dynprog(lang, s_mat, a, b):
    final_score = 0
    a = a[::-1]
    b = b[::-1]
    backtrack = np.zeros((len(a)+1, len(b)+1), dtype=np.int)
    print(backtrack)
    out_a = np.arange(len(a), dtype=np.int).tolist()
    out_b = np.arange(len(b), dtype=np.int).tolist()
    # print(backtrack)
    backtrack = align(lang, s_mat, backtrack, a, b)
    # score(lang, s_mat, a, b)
    return backtrack
    # return [final_score, out_a, out_b]


def score(lang, s_mat, a, b):
    if a == "-":
        return s_mat[-1][lang.index(b)]
    elif b == "-":
        return s_mat[lang.index(a)][-1]
    else:
        return s_mat[lang.index(a)][lang.index(b)]


def align(lang, s_mat, b_mat, a, b):
    # if len(a) == 2 and len(b) == 2:
    #     b_mat[-1][-1] = score(lang, s_mat, a[0], b[0])
    #     return b_mat
    print(b_mat)
    if (len(a)==0) and (len(b)==0):
        return b_mat
        # return 0
    elif len(a)==0:
        b_mat[0][len(b)] = score(lang, s_mat, "-", b[0]) + align(lang, s_mat, b_mat, a, b[1:])[0][len(b)-1]
        return b_mat
        # return score(lang, s_mat, a[0], b[0]) + align(lang, s_mat, b_mat, a, b[1:])
    elif len(b)==1:
        b_mat[len(a)][0] = score(lang, s_mat, a[0], "-") + align(lang, s_mat, b_mat, a[1:], b)[len(a)-1][0]
        return b_mat
        # return score(lang, s_mat, a[0], b[0]) + align(lang, s_mat, b_mat, a[1:], b)
    else:
        b_mat[len(a)][len(b)] = score(lang, s_mat, a[0], b[0]) + align(lang, s_mat, b_mat, a[1:], b[1:])[len(a)-1][len(b)-1]
        b_mat[len(a)][len(b) - 1] = score(lang, s_mat, "-", b[0]) + align(lang, s_mat, b_mat, a, b[1:])[len(a) - 1][len(b)]
        b_mat[len(a) - 1][len(b)] = score(lang, s_mat, a[0], "-") + align(lang, s_mat, b_mat, a[1:], b)[len(a)][len(b)-1]
        return b_mat
        # return max(
        #     score(lang, s_mat, a[0], b[0]) + align(lang, s_mat, b_mat, a[1:], b[1:]),
        #     score(lang, s_mat, "-", b[0]) + align(lang, s_mat, b_mat, a, b[1:]),
        #     score(lang, s_mat, a[0], "-") + align(lang, s_mat, b_mat, a[1:], b)
        # )


if __name__ == '__main__':
    language = "ACG"
    score_matrix = [[1, -1, -1, -2],
                    [-1, 1, -1, -2],
                    [-1, -1, 1, -2],
                    [-2, -2, -2, 0]]
    seq_a = "AAAC"
    seq_b = "AGC"

    # language = "ABC"
    # score_matrix = [[1, -1, -2, -1],
    #                 [-1, 2, -4, -1],
    #                 [-2, -4, 3, -2],
    #                 [-1, -1, -2, 0]]
    # seq_a = "ABAAC"
    # seq_b = "ABAC"

    print(dynprog(language, score_matrix, seq_a, seq_b))
