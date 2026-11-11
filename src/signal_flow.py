import itertools

def mason(forward_paths,loops):
    # Each path/loop is (gain, set(nodes)); loops touching if node sets intersect.
    delta=1-sum(g for g,n in loops)
    for r in range(2,len(loops)+1):
        for combo in itertools.combinations(loops,r):
            if all(not (a[1]&b[1]) for a,b in itertools.combinations(combo,2)):
                delta+=(-1)**r*__import__('math').prod(x[0] for x in combo)
    numer=0.; deltas=[]
    for p,nodes in forward_paths:
        d=1-sum(g for g,ns in loops if not nodes&ns)
        for r in range(2,len(loops)+1):
            for combo in itertools.combinations(loops,r):
                if all(not(a[1]&b[1]) for a,b in itertools.combinations(combo,2)) and all(not nodes&x[1] for x in combo): d+=(-1)**r*__import__('math').prod(x[0] for x in combo)
        deltas.append(d); numer+=p*d
    return {'delta':delta,'delta_k':deltas,'transfer':numer/delta}
