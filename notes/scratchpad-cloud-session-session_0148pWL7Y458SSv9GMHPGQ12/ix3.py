from daffodil.daf import Daf
orig_insert_col = Daf.insert_col
def safe_insert_col(self, *a, **k):
    self.lol = [list(r) for r in self.lol]     # own rows first
    return orig_insert_col(self, *a, **k)
Daf.insert_col = safe_insert_col

d = Daf(lol=[[1,'a'],[2,'b'],[3,'c']], cols=['id','v'], keyfield='id')
first_matches_daf = d.select_irows([0,1])
first_matches_daf.insert_idx_col(colname='idx', icol=0)     # return value discarded, as at line 1614
print(first_matches_daf.to_md())
print('original:', d.lol, list(d.hd))
