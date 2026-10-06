def check_rectangular_message(lol, width):
    short = [(i, len(r)) for i, r in enumerate(lol) if len(r) < width]
    long_ = [(i, len(r)) for i, r in enumerate(lol) if len(r) > width]
    if not short and not long_:
        return None
    parts = []
    if long_:
        parts.append(f"{len(long_)} row(s) are longer, often from an unquoted comma. First: {long_[:5]}")
    if short:
        parts.append(f"{len(short)} row(s) are shorter, often from a cut line. First: {short[:5]}")
    return f"from_csv_buff: {len(short)+len(long_)} of {len(lol)} data rows do not have the {width} columns of the header. " + " ".join(parts) + " Each entry is (data row index, length)."

lol = [['1','a','x']] + [['2','b','x','EXTRA']]*3 + [['3','c']]*2 + [['4','d','x']] + [['5','e','x','EXTRA','MORE']]*40
print(check_rectangular_message(lol, 3))
print()
print(check_rectangular_message([['1','Ann','30'], ['2','Bob'], ['3','Cy','40','extra']], 3))
