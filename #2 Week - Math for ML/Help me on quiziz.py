from cmath import sqrt
from threading import main_thread
import numpy as np


# Solving x, y, z
def solve_linear_eq(problems, equal_to):
    result = np.linalg.solve(problems, equal_to)
    return result

def matrix_multiplication(matrix_1, matrix_2):
    return np.array(matrix_1) * np.array(matrix_2)

def matrix_division(matrix_1, matrix_2):
    return np.array(matrix_1) / np.array(matrix_2)

def matrix_addition(matrix_1, matrix_2):
    return np.array(matrix_1) + np.array(matrix_2)

def matrix_substraction(matrix_1, matrix_2):
    return np.array(matrix_1) - np.array(matrix_2)

def vector_projection(vector_a, vector_b):
    vector_a = np.array(vector_a)
    vector_b = np.array(vector_b)
    projb_a = (np.dot(vector_a, vector_b) / sum(vector_a**2)) * vector_a
    return projb_a

def find_new_basis(vector_old, standard):
    """
        standard : [a, b, c]
    """

    return np.linalg.inv(np.array(standard)).dot(vector_old)

def checking_ortogonal(matrix):
    vector = np.array(matrix)
    return np.transpose(vector) == np.linalg.inv(vector)

def checking_ortonormal(matrices):
    matrix = np.array(matrices)

    cond_a = True
    cond_b = True

    for row in matrix:
        for other_row in matrix:
            if list(row) != list(other_row):
                if (round(np.dot(row, other_row))) != 0:
                    cond_a = False
                    print("masuk sini", np.dot(row, other_row))
                    break

    if len(matrices) != round(sum(sum(matrix ** 2))) : 
       cond_b = False

    return cond_a and cond_b


def matrix_determinant(matrix_a):
    matrix = np.array(matrix_a)
    return np.linalg.det(matrix)

# A = [[1, 3, 5], [7, 12, 21], [5, 18, 3]]
# B = [13, 123, 51]
# print(f"result : {solve_linear_eq(A, B)}")

# vector_a = [0, 1, 0]
# vector_b = [1/(2**0.5), 0,  1/(2**0.5)]
# vector_c = [-1/(2**0.5), 0,  1/(2**0.5)]
# matrix = [vector_a, vector_b, vector_c]
# print(f"result : {checking_ortonormal(matrix)}")

a = np.array([ [3, 2], [4, -2, 5], [2, 8, 7] ])
b = 
print("INVERSE", matrix_multiplication(a))