# coding: utf-8

from z3 import *
a = Array('a', BitVecSort(8), RealSort())
b = Array('b', BitVecSort(8), RealSort())
a0 = a[0] == 5
b0 = b[0] == 0
i = BitVec('i', 8)
f_a = ForAll(BitVec('i', 8), a[i] >= a[0] + BV2Int(i * BitVecVal(1, 8)))
f_b = ForAll(BitVec('i', 8), b[i] <= b[0] + BV2Int(i * BitVecVal(4, 8))) #
Change the 4 to 1 and get unsat

n = BitVec('n', 8)
f_n =  n == BitVecVal(3, 8)

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
# print(s.model())
