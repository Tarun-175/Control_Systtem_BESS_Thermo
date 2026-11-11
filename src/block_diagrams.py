def series(*g):
    from functools import reduce
    import operator
    return reduce(operator.mul,g,1)
def parallel(*g): return sum(g)
def feedback(g,h=1): return g/(1+g*h)
