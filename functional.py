def joint(*functions):
    return lambda x: map(lambda f: f(x), functions)

def mapcan(f, xs):
    return (y for x in xs
            for y in f(x))

def take(n, seq):
    return list(itertools.islice(seq, n))
