def consolidate(contribs):
    return max((c.value for c in contribs), default=0)


def score(out):
    return out == 1
