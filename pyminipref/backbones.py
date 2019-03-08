import sys
import pyminipref

if len(sys.argv) != 2:
    raise Exception("Usage: python3 %s instance" % sys.argv[0])

D = pyminipref.Dimacs()
D.parse(sys.argv[1])

S = pyminipref.MinisatSolver()
S.new_vars(D.n_vars())

for clause in D.clauses:
    res = S.add_clause(clause)
    if not res:
        print ("Unsatisfiable")
        exit(20)

res = S.solve()
if not res:
    print ("Unsatisfiable")
    exit(20)

model=list(S.get_model())
candidates=[]
j = 0
for i in model:
    j += 1
    if i == 0:
        S.set_preference(j, 1)
        candidates.append(-j)
    else:
        S.set_preference(-j, 1)
        candidates.append(j)

print ("bbc current size: %s" % len(candidates))
while len(candidates) > 0:
    res = S.solve()
    assert res

    newCandidates = []
    for i in candidates:
        if i > 0 and S.model_value(i) == 1 or i < 0 and S.model_value(-i) == 0:
            newCandidates.append(i)
        else:
            S.remove_preference(abs(i))

    if len(candidates) == len(newCandidates):
        break
    candidates = newCandidates
    print ("bbc current size: %s" % len(candidates))

print ("Backbones: %s" % candidates)
print("Number of backbones: %s" % len(candidates));
exit(10)
