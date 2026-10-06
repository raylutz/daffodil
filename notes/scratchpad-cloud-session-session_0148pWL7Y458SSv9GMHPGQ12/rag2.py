import time
from daffodil.daf import Daf
d = Daf.from_csv_buff("id,name,age\n1,Ann,30\n2,Bob\n3,Cy,40,extra\n")
rows = "\n".join(f"{i},name{i},{i%90},x,y,z,w,v" for i in range(200000))
text = "a,b,c,d,e,f,g,h\n" + rows + "\n"
def best(f, k=3):
    r=[]
    for _ in range(k):
        t=time.perf_counter(); out=f(); r.append(time.perf_counter()-t)
    return min(r), out
t1,big = best(lambda: Daf.from_csv_buff(text))
t2,_ = best(lambda: big.is_rectangular())
print(f'from_csv_buff, 200,000 rows x 8: {t1:.3f}s   is_rectangular(): {t2:.4f}s')
