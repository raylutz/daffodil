from impl_pydantic import Batch as PB
from impl_marshmallow import Batch as MB
from daffodil.daf import Daf
def p(bs: list[PB]) -> int: return sum(b.num_balots for b in bs)
def m(bs: list[MB]) -> int: return sum(b.num_balots for b in bs)
def d(daf: Daf) -> int:     return sum(daf.col('num_balots'))
