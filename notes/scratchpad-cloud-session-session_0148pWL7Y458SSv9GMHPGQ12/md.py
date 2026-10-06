from daffodil.daf import Daf
cases = {
 'normal (header + separator)':          "| id | v |\n| -- | - |\n| 1 | a |\n| 2 | b |\n",
 'rows only, no separator':              "| 1 | a |\n| 2 | b |\n",
 'separator first, then rows':           "| -- | - |\n| 1 | a |\n| 2 | b |\n",
 'blank header + separator + rows':      "|  |  |\n| -- | - |\n| 1 | a |\n| 2 | b |\n",
 'header + separator, no rows':          "| id | v |\n| -- | - |\n",
 'empty text':                           "",
}
for k, text in cases.items():
    try:
        d = Daf.from_md(text); print(f'{k:34} cols={d.columns()} lol={d.lol}')
    except Exception as e: print(f'{k:34} {type(e).__name__}: {str(e)[:80]}')
