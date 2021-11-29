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
    return divmod(uid, gw_dim[1])


def cell2uid(r, c, gw_dim):
    return r * gw_dim[1] + c