# test_daf_inverse_key_selection.py
# Inverse key selection gives the same rows after the fix in gkeys_to_idxs(), which made it linear (2026-10-14).

from daffodil.daf import Daf


def make_daf(n: int) -> Daf:
    return Daf(lol=[[f'k{i:06d}', i] for i in range(n)], cols=['id', 'n'], keyfield='id')


def test_inverse_keeps_the_others_in_order():
    data_daf: Daf = make_daf(10)
    assert data_daf.select_krows(['k000007', 'k000002', 'k000005'], inverse=True)[:, 'n'].to_list() == [0, 1, 3, 4, 6, 8, 9]


def test_inverse_with_repeated_keys():
    data_daf: Daf = make_daf(5)
    assert data_daf.select_records_daf(['k000001', 'k000001', 'k000003'], inverse=True)[:, 'n'].to_list() == [0, 2, 4]


def test_inverse_of_half_of_a_large_table():
    data_daf: Daf = make_daf(200_000)
    keys_ls: list = [f'k{i:06d}' for i in range(0, 200_000, 2)]
    kept_daf: Daf = data_daf.select_records_daf(keys_ls, inverse=True)
    assert len(kept_daf) == 100_000
    assert kept_daf[:3, 'n'].to_list() == [1, 3, 5]
    assert kept_daf.lol[0] is data_daf.lol[1]
