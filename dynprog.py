import numpy as np


def dynprog(lang, s_mat, a, b):
    a = "-" + a
    b = "-" + b
    backtrack = np.zeros((len(a), len(b)), dtype=np.int)
    direction = np.zeros((len(a), len(b)), dtype=np.str)
    direction[:] = "E"
    backtrack, direction = align(lang, s_mat, backtrack, direction, a, b)
    ind = np.unravel_index(np.argmax(backtrack, axis=None), backtrack.shape)
    final_score = np.amax(backtrack)
    out_a, out_b = create_alignment(backtrack, direction, ind, a, b)
    return [final_score, out_a, out_b]


def score(lang, s_mat, a, b):
    if a == "-":
        return s_mat[-1][lang.index(b)]
    elif b == "-":
        return s_mat[lang.index(a)][-1]
    else:
        return s_mat[lang.index(a)][lang.index(b)]


def align(lang, s_mat, b_mat, d_mat, a, b):
    for i in range(1, len(a)):
        for j in range(1, len(b)):
            b_mat[i, j] = max(0,
                              b_mat[i - 1][j - 1] + score(lang, s_mat, a[i], b[j]),
                              b_mat[i-1][j] + score(lang, s_mat, a[i], "-"),
                              b_mat[i][j-1] + score(lang, s_mat, "-", b[j])
                              )
            if b_mat[i, j] == b_mat[i - 1][j - 1] + score(lang, s_mat, a[i], b[j]):
                d_mat[i, j] = "D"
            elif b_mat[i, j] == b_mat[i-1][j] + score(lang, s_mat, a[i], "-"):
                d_mat[i, j] = "U"
            else:
                d_mat[i, j] = "L"
                              
    return b_mat, d_mat


def create_alignment(b_mat, d_mat, start, a, b):
    previous_a = start[0]
    previous_b = start[1]
    output = [[], []]
    while previous_a > 0 and previous_b > 0:
        if d_mat[previous_a][previous_b] == "D":
            previous_a -= 1
            previous_b -= 1
            output[0].insert(0, previous_a)
            output[1].insert(0, previous_b)
        elif d_mat[previous_a][previous_b] == "U":
            previous_a -= 1
        elif d_mat[previous_a][previous_b] == "L":
            previous_b -= 1
        elif d_mat[previous_a][previous_b] == "E":
            break

    return output


if __name__ == '__main__':
    language = "CTGA"
    score_matrix = [[10, -5, -5, -5, -7],
                    [-5, 10, -5, -5, -7],
                    [-5, -5, 10, -5, -7],
                    [-5, -5, -5, 10, -7],
                    [-7, -7, -7, -7, 0]]
    seq_a = "CTCGTC"
    seq_b = "AGCGTAG"
    print(dynprog(language, score_matrix, seq_a, seq_b))

    language = "TCA"
    score_matrix = [[1, -1, -1, -2],
                   [-1, 1, -1, -2],
                   [-1, -1, 1, -2],
                   [-2, -2, -2, 0]]
    seq_a = "TAATA"
    seq_b = "TACTAA"
    print(dynprog(language, score_matrix, seq_a, seq_b))

    language = "ABC"
    score_matrix = [[1, -1, -2, -1],
                   [-1, 2, -4, -1],
                   [-2, -4, 3, -2],
                   [-1, -1, -2, 0]]
    seq_a = "ABCACA"
    seq_b = "BAACB"
    print(dynprog(language, score_matrix, seq_a, seq_b))

    language = "CTGA"
    score_matrix = [[3, -3, -3, -3, -2],
                    [-3, 3, -3, -3, -2],
                    [-3, -3, 3, -3, -2],
                    [-3, -3, -3, 3, -2],
                    [-2, -2, -2, -2, 0]]
    seq_a = "GGTTGACTA"
    seq_b = "TGTTACGG"
    print(dynprog(language, score_matrix, seq_a, seq_b))

    language = "ABC"
    score_matrix = [[1,-1,-2,-1],
                    [-1,2,-4,-1],
                    [-2,-4,3,-2],
                    [-1,-1,-2,0]]
    seq_a = "AABBAACA"
    seq_b = "CBACCCBA"
    print(dynprog(language, score_matrix, seq_a, seq_b))
