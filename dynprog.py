import numpy as np


def dynprog(lang, s_mat, a, b):
    final_score = 0
    a = a + "-"
    b = b + "-"
    backtrack = np.zeros((len(a), len(b)), dtype=np.int)
    out_a = np.arange(len(a), dtype=np.int).tolist()
    out_b = np.arange(len(b), dtype=np.int).tolist()
    # print(backtrack)
    final_score = align(lang, s_mat, backtrack, a, b)
    # score(lang, s_mat, a, b)
    return [final_score, out_a, out_b]


def score(lang, s_mat, a, b):
    if a == "-":
        return s_mat[-1][lang.index(b)]
    elif b == "-":
        return s_mat[lang.index(a)][-1]
    else:
        return s_mat[lang.index(a)][lang.index(b)]


def align(lang, s_mat, b_mat, a, b):
    # print(len(a), a, len(b), b)
    # print(b_mat)
    if (a == "-") and (b == "-"):
        return 0
    elif a == "-":
        return score(lang, s_mat, a[0], b[0]) + align(lang, s_mat, b_mat, a, b[1:])
    elif b == "-":
        return score(lang, s_mat, a[0], b[0]) + align(lang, s_mat, b_mat, a[1:], b)
    else:
        return max(
            score(lang, s_mat, a[0], b[0]) + align(lang, s_mat, b_mat, a[1:], b[1:]),
            score(lang, s_mat, "-", b[0]) + align(lang, s_mat, b_mat, a, b[1:]),
            score(lang, s_mat, a[0], "-") + align(lang, s_mat, b_mat, a[1:], b)
        )


if __name__ == '__main__':
    language = "ABC"
    score_matrix = [[1, -1, -2, -1],
                    [-1, 2, -4, -1],
                    [-2, -4, 3, -2],
                    [-1, -1, -2, 0]]
    seq_a = "AAAAC"
    seq_b = "ABAC"

    print(dynprog(language, score_matrix, seq_a, seq_b))
