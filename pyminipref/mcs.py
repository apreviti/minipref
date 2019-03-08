import sys
import pyminipref

def addVariable(S):
    return S.new_var()+1

if len(sys.argv) != 2:
    raise Exception("Usage: python3 %s instance" % sys.argv[0])

D = pyminipref.Dimacs()
D.parse(sys.argv[1])

S = pyminipref.MinisatSolver()
S.new_vars(D.n_vars())

orig = D.n_vars()

if D.maxWeight == None:
    for clause in D.clauses:        
        clause.append(addVariable(S))
        res = S.add_clause(clause)
        assert res
else:
    for clause in D.clauses:
        S.add_clause(clause)
    for soft in D.softClauses:
        soft.append(addVariable(S))
        res = S.add_clause(soft)
        assert res

for i in range(orig+1,S.nvars()+1):
    S.set_preference(-i,1)

res = S.solve()
if not res:
    print ("Unsatisfiable")
    exit(20)
else:
    trues = []
    for i in range(orig+1,S.nvars()+1):
        if S.model_value(i) == 1:
            if D.maxWeight == None:
                trues.append("Clause %s, n°: %s, selector: %s" % (D.clauses[i-orig-1][:-1],i-orig,i))
            else:
                trues.append("Soft clause %s, n°: %s, selector: %s" % (D.softClauses[i-orig-1][:-1],i-orig,i))
    print(trues)

exit(10)
