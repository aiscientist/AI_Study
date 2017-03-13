def rd(x):
    b = []
    for a in x:
        if a not in b:
            b.append(a)
    return b


print(rd([1, 2, 3, 4, 3, 2, 1]))