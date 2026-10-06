import doctest, re, sys, collections, importlib
mods = ['daffodil.daf', 'daffodil.keyedlist', 'daffodil.lib.daf_md', 'daffodil.lib.daf_pandas']
finder = doctest.DocTestFinder(recurse=True); parser = doctest.DocTestParser()
total = 0; failed = []; ok = 0; lol_lines_by_test = collections.Counter()
for m in mods:
    mod = importlib.import_module(m)
    for test in finder.find(mod, mod.__name__):
        globs = dict(test.globs)
        for ex in test.examples:
            src = ex.source
            if re.search(r'\.lol\b', src):
                total += 1
                new = re.sub(r'\.lol\b', '.to_lod()', src)
                try:
                    exec(compile(new, '<x>', 'exec' if ex.source.rstrip().endswith(':') else 'single'), globs)
                    ok += 1
                except Exception as e:
                    failed.append((test.name, src.strip()[:90], type(e).__name__))
                    # keep the state moving with the original source
                    try: exec(compile(src, '<x>', 'single'), globs)
                    except Exception: pass
            else:
                try: exec(compile(src, '<x>', 'exec' if '\n' in src.strip() else 'single'), globs)
                except Exception: pass
print('examples with .lol:', total, '| ran with to_lod():', ok, '| raised:', len(failed))
for n, s, e in failed: print(f'  {e:18} {n.split(".")[-1]:26} {s}')
