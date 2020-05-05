
from z3 import *
a = Array('a', BitVecSort(16), RealSort())
b = Array('b', BitVecSort(16), RealSort())
a0 = a[0] == 5
b0 = b[0] == 0
i = BitVec('i', 16)
f_a = ForAll(i, a[i] >= a[0] + BV2Int(i * BitVecVal(1, 16)))
f_b = ForAll(i, b[i] <= b[0] + BV2Int(i * BitVecVal(1, 16))) #
# Change the 4 to 1 and get unsat

n = BitVec('n', 16)
f_n =  n == BitVecVal(3, 16)

a_b = a[n] < b[n]
s = Solver()
s.push()
s.add(a0)
s.add(b0)
s.add(f_a)
s.add(f_b)
s.add(f_n)
s.add(a_b)
print(s.sexpr())
print(s.check())
if (s.check() == sat):
    print(s.model())
