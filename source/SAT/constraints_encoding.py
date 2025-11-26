from z3 import *


# Naive pairwise encoding
def at_least_one(bool_vars):
    return Or(bool_vars)

def at_most_one(bool_vars, name = ""):
    n = len(bool_vars)
    return And([Not(And(bool_vars[i], bool_vars[j])) for i in range(n) for j in range(i+1, n)])

def exactly_one(bool_vars, name = ""):
    return And(at_least_one(bool_vars), at_most_one(bool_vars))


# Sequential encoding with k greater than 1
def at_least_k(bool_vars, k, name):
    not_bool_vars = [Not(b) for b in bool_vars]
    n = len(bool_vars)
    return at_most_k(not_bool_vars, n - k, f'{name}_not')

def at_most_k(bool_vars, k, name):
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


def exactly_k(bool_vars, k, name):
    return And(at_least_k(bool_vars, k, name), at_most_k(bool_vars, k, name))