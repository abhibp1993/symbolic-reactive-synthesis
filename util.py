def id2binary(id, num_bits):
    return format(id, f"0{num_bits}b")


def id2expr(id, num_bits, varname="u"):
    uid_binary = id2binary(id, num_bits)
    expr = [f"{varname}{i}" if uid_binary[-1 - i] == "1" else f"~{varname}{i}" for i in range(0, num_bits)]
    expr.reverse()
    expr = " & ".join(expr)
    return expr


def id2dict(id, num_bits, varname="u"):
    id_binary = id2binary(id, num_bits)
    dict_ = dict()
    for i in range(0, num_bits):
        if id_binary[-1 - i] == "1":
            dict_[f"{varname}{i}"] = True
        else:
            dict_[f"{varname}{i}"] = False
    return dict_


def uid2cell(uid, gw_dim):
    p, residue = divmod(uid, gw_dim[0] * gw_dim[1])
    c, r = divmod(residue, gw_dim[0])
    return r, c, p


def cell2uid(r, c, p, gw_dim):
    return (p * gw_dim[0] * gw_dim[1]) + (c * gw_dim[0]) + r


if __name__ == '__main__':
    idx = list()
    for r in range(3):
        for c in range(3):
            for p in range(2):
                print(cell2uid(r, c, p, (3, 3)))
                idx.append(cell2uid(r, c, p, (3, 3)))
    idx.sort()
    print(idx)

    cells = list()
    for i in range(18):
        print(uid2cell(i, (3, 3)))
        cells.append(uid2cell(i, (3, 3)))
    cells.sort()
    print(cells)

