from maraboupy.MarabouRNN import *
from maraboupy import MarabouCore

large = 1000


def minimal_changes_that_works():
    '''
    Marabou finds a valid assignment to this.
    The only difference between this function ans the "minimal_example" that marabou
    doesn't find an assignment is that we added there a variable and constraints it
    to be exactly 1.0
    '''
    query = MarabouCore.InputQuery()
    query.setNumberOfVariables(3)

    # x
    query.setLowerBound(0, 1)
    query.setUpperBound(0, 1.2)

    # s_i-1 f
    query.setLowerBound(1, 0)
    query.setUpperBound(1, large)

    # s_i b
    query.setLowerBound(2, -large)
    query.setUpperBound(2, large)

    # s_i b = x * 1 + s_i-1 f * 1
    update_eq = MarabouCore.Equation()
    update_eq.addAddend(1, 0)
    update_eq.addAddend(1, 1)
    update_eq.addAddend(-1, 2)
    update_eq.setScalar(0)
    query.addEquation(update_eq)

    # # i
    # query.setLowerBound(3, 0)
    # query.setLowerBound(3, 5)

    # i == 1
    # loop_eq = MarabouCore.Equation()
    # loop_eq.addAddend(1, 3)
    # loop_eq.setScalar(1)
    # query.addEquation(loop_eq)

    vars1, stats1 = MarabouCore.solve(query, "", 0)
    if len(vars1) > 0:
        print("SAT")
        print(vars1)
        assert True
    else:
        update_eq.dump()
        # loop_eq.dump()
        print("UNSAT")
        assert False


def minimal_example():
    '''
    Marabou can't find an assignment to this query
    here is a valid one: {0: 1.0, 1: 0.0, 2: 1.0, 3:1.0}
    it's exactly the same as minimal_changes_that_works just with the extra variable
    '''
    query = MarabouCore.InputQuery()
    query.setNumberOfVariables(4)

    # x
    query.setLowerBound(0, 1)
    query.setUpperBound(0, 1.2)

    # s_i-1 f
    query.setLowerBound(1, 0)
    query.setUpperBound(1, large)

    # s_i b
    query.setLowerBound(2, -large)
    query.setUpperBound(2, large)

    # i
    query.setLowerBound(3, 0)
    query.setUpperBound(3, large)

    # s_i b = x * 1 + s_i-1 f * 1
    update_eq = MarabouCore.Equation()
    update_eq.addAddend(1, 0)
    update_eq.addAddend(1, 1)
    update_eq.addAddend(-1, 2)
    update_eq.setScalar(0)
    query.addEquation(update_eq)

    # i == 1
    loop_eq = MarabouCore.Equation()
    loop_eq.addAddend(1, 3)
    loop_eq.setScalar(1)
    query.addEquation(loop_eq)

    vars1, stats1 = MarabouCore.solve(query, "", 0)
    if len(vars1) > 0:
        print("SAT")
        print(vars1)
        assert True
    else:
        update_eq.dump()
        loop_eq.dump()
        print("UNSAT")
        assert False


def full_example():
    '''
    This is what I really need (the API is much cleaner of course, for debugging
    everything is here)
    '''
    query = MarabouCore.InputQuery()
    query.setNumberOfVariables(5)

    # x
    query.setLowerBound(0, 1)
    query.setUpperBound(0, 1.2)

    # i
    query.setLowerBound(1, 0)
    query.setLowerBound(1, 5)

    # s_i-1 f
    query.setLowerBound(2, 0)
    query.setUpperBound(2, large)

    # s_i b
    query.setLowerBound(3, -large)
    query.setUpperBound(3, large)

    # s_i f
    query.setLowerBound(4, 0)
    query.setUpperBound(4, large)

    # s_i f = ReLu(s_i b)
    MarabouCore.addReluConstraint(query, 3, 4)

    # s_i b = x * 1 + s_i-1 f * 1
    update_eq = MarabouCore.Equation()
    update_eq.addAddend(1, 0)
    update_eq.addAddend(1, 2)
    update_eq.addAddend(-1, 3)
    update_eq.setScalar(0)
    query.addEquation(update_eq)

    # s_i-1 f <= i-1 <--> s_i-1 f - i <= -1 <--> i - s_i-1 f >= 1
    induction_step = MarabouCore.Equation(MarabouCore.Equation.GE)
    induction_step.addAddend(1, 1)  # i
    induction_step.addAddend(-1, 2)  # s_i f
    induction_step.setScalar(1)
    query.addEquation(induction_step)

    # i == 1
    loop_eq = MarabouCore.Equation()
    loop_eq.addAddend(1, 1)
    loop_eq.setScalar(1)
    query.addEquation(loop_eq)

    # s_i-1 f == 0
    loop_eq2 = MarabouCore.Equation()
    loop_eq2.addAddend(1, 2)
    loop_eq2.setScalar(0)
    query.addEquation(loop_eq2)

    vars1, stats1 = MarabouCore.solve(query, "", 0)
    if len(vars1) > 0:
        print("SAT")
        print(vars1)
        assert True
    else:
        update_eq.dump()
        induction_step.dump()
        loop_eq.dump()
        loop_eq2.dump()
        print("UNSAT")
        assert False


if __name__ == "__main__":
    # I except that all functions will have a valid assignment
    # minimal_changes_that_works()

    # valid assignment: {0: 1.0, 1: 0.0, 2: 1.0, 3:1.0}
    minimal_example()

    # valid assignment: {0: 1.1, 1: 1.0, 2:0.0, 3:1.1, 4:1.1}
    # full_example()
