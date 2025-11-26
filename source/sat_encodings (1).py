from itertools import combinations
import math
from z3 import *


# Naive pairwise encoding
def at_least_one_np(bool_vars):
    return Or(bool_vars)

def at_most_one_np(bool_vars, name = ""):
    n = len(bool_vars)
    return And([Not(And(bool_vars[i], bool_vars[j])) for i in range(n) for j in range(i+1, n)])

def exactly_one_np(bool_vars, name = ""):
    return And(at_least_one_np(bool_vars), at_most_one_np(bool_vars))


# Sequential encoding
def at_least_one_seq(bool_vars):
    return Or(bool_vars)

def at_most_one_seq(bool_vars, name=""):
    x = bool_vars
    n = len(bool_vars)
    sums = [Bool(f"s_{i}_{name}") for i in range(n-1)]
    first = Or(Not(x[0]), sums[0])

    # Middle clauses for i=1 to n-2:
    # if x[i] is true, then s[i] must be true
    # if s[i-1] is true, then s[i] must be true
    # if s[i-1] is true, then x[i] must be false
    middle = And([And(
        Or(Not(x[i]), sums[i]),
        Or(Not(sums[i-1]), sums[i]),
        Or(Not(sums[i-1]), Not(x[i]))
    ) for i in range(1, n-1)])

    last = Or(Not(sums[n-2]), Not(x[n-1]))
    
    return And(first, middle, last)

def exactly_one_seq(bool_vars, name=""):
    return And(at_least_one_seq(bool_vars), at_most_one_seq(bool_vars, name))


# Bitwise encoding
def toBinary(num, length, name = ""):
    bits = []
    for i in range(length):
        bit = num % 2
        num //= 2
        bit_one = Bool(f"r_{i}_{name}")
        bits.append(bit_one if bit == 1 else Not(bit_one))
    return bits

def at_least_one_bw(bool_vars):
    return Or(bool_vars)

def at_most_one_bw(bool_vars, name=""):
    x = bool_vars
    n = len(bool_vars)
    length = math.ceil(math.log2(n))
    r = [toBinary(i, length, name) for i in range(n)]
    return And([And([Or(Not(x[i]), r_i_j) for r_i_j in r[i]]) for i in range(n)])

def exactly_one_bw(bool_vars, name=""):
    return And(at_least_one_bw(bool_vars), at_most_one_bw(bool_vars, name))


# Heule encoding
def at_least_one_he(bool_vars):
    return Or(bool_vars)

count = 0

def at_most_one_he(bool_vars, name=""):
    global count
    x = bool_vars
    n = len(bool_vars)
    if n > 4:
        y = Bool(f"y_{name}")
        count += 2
        return And(
            at_most_one_he(x[:3] + [y], name=f"{name}_{count}"),
            at_most_one_he([Not(y)] + x[3:], name=f"{name}_{count + 1}")
        )
    else:
        return at_most_one_np(x)

def exactly_one_he(bool_vars, name=""):
    return And(at_least_one_he(bool_vars), at_most_one_he(bool_vars, name))


# Naive pairwise encoding with k greater than 1
def at_least_k_np(bool_vars, k, name = ""):
    not_bool_vars = [Not(b) for b in bool_vars]
    n = len(bool_vars)
    return at_most_k_np(not_bool_vars, n - k, name)

def at_most_k_np(bool_vars, k, name = ""):
    x = bool_vars
    return And([Or([Not(xi) for xi in subset]) for subset in combinations(x, k + 1)])

def exactly_k_np(bool_vars, k, name = ""):
    return And(at_least_k_np(bool_vars, k, name), at_most_k_np(bool_vars, k, name))


# Sequential encoding with k greater than 1
def at_least_k_seq(bool_vars, k, name):
    not_bool_vars = [Not(b) for b in bool_vars]
    n = len(bool_vars)
    return at_most_k_seq(not_bool_vars, n - k, f'{name}_not')

def at_most_k_seq(bool_vars, k, name):
    constraints = []
    n = len(bool_vars)
    s = [[Bool(f"s_{name}_{i}_{j}") for j in range(k)] for i in range(n - 1)]
    constraints.append(Or(Not(bool_vars[0]), s[0][0]))
    constraints += [Not(s[0][j]) for j in range(1, k)]

    for i in range(1, n-1):
        constraints.append(Or(Not(bool_vars[i]), s[i][0]))
        constraints.append(Or(Not(s[i-1][0]), s[i][0]))
        constraints.append(Or(Not(bool_vars[i]), Not(s[i-1][k-1])))
        for j in range(1, k):
            constraints.append(Or(Not(bool_vars[i]), Not(s[i-1][j-1]), s[i][j]))
            constraints.append(Or(Not(s[i-1][j]), s[i][j]))

    constraints.append(Or(Not(bool_vars[n-1]), Not(s[n-2][k-1]))) 
      
    return And(constraints)


def exactly_k_seq(bool_vars, k, name):
    return And(at_least_k_seq(bool_vars, k, name), at_most_k_seq(bool_vars, k, name))