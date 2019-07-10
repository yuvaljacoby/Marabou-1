from z3 import *
a = BitVecVal(-1, 4)
b = Int('b')

s = Solver()
s.push()
s.add(b == BV2Int(a, is_signed=True))
print(s.sexpr())
