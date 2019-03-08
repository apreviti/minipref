"""A Python API for the MiniSat_ solver.

.. _MiniSat: http://minisat.se/
Classes:
  `MinisatSolver`
    Solve CNF instances using MiniSat.

  Solver
    An abstract base class for the other classes.
"""

import array
import os
import ctypes  # type: ignore
from abc import ABCMeta, abstractmethod
from ctypes import c_void_p, c_ubyte, c_bool, c_int, c_double  # type: ignore
import sys

try:
    import typing  # noqa: for mypy-lang type-checking
    from typing import Iterable, Sequence, Tuple  # noqa: for mypy-lang type-checking
except ImportError:
    # not needed at runtime, so no error
    pass


class Solver(object):
    """The Solver class is an abstract base class for MiniSat.
    It provides the basic methods that both
    contain, closely following the methods in MiniSat Solver class.

    Solver should not be instantiated directly.  Instead, use its
    subclass MinisatSolver.
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, libfilename):  # type: (str) -> None
        self._setup_lib(libfilename)
        self.s = self.lib.Solver_new()

    def _setup_lib(self, libfilename):  # type: (str) -> None
        """Load the minisat library with ctypes and create a Solver
           object.  Correct return types (if not int as assumed by
           ctypes) and set argtypes for functions from the minisat
           library.
        """
        dirname = os.path.dirname(os.path.abspath(__file__))
        libfile = os.path.join(dirname, libfilename)
        if not os.path.exists(libfile):
            raise IOError("Specified library file not found.  Did you run 'make' to build the solver libraries?\nFile not found: %s" % libfile)

        self.lib = ctypes.cdll.LoadLibrary(dirname+'/'+libfilename)

        l = self.lib

        l.Solver_new.restype = c_void_p
        l.Solver_new.argtypes = []
        l.Solver_delete.argtypes = [c_void_p]

        l.nVars.argtypes = [c_void_p]
        l.nClauses.argtypes = [c_void_p]
        l.setPhaseSaving.argtypes = [c_void_p, c_int]
        l.setRndPol.argtypes = [c_void_p, c_bool]
        l.setRndInitAct.argtypes = [c_void_p, c_bool]
        l.setRndSeed.argtypes = [c_void_p, c_double]

        l.newVar.argtypes = [c_void_p, c_ubyte, c_bool]

        l.addClause.restype = c_bool
        l.addClause.argtypes = [c_void_p, c_int, c_void_p]
        l.addUnit.restype = c_bool
        l.addUnit.argtypes = [c_void_p, c_int]

        l.solve.restype = c_bool
        l.solve.argtypes = [c_void_p]
        l.solve_assumptions.restype = c_bool
        l.solve_assumptions.argtypes = [c_void_p, c_int, c_void_p]
        l.check_complete.restype = c_bool
        l.check_complete.argtypes = [c_void_p, c_int, c_void_p, c_bool]
        l.simplify.restype = c_bool
        l.simplify.argtypes = [c_void_p]

        l.conflictSize.argtypes = [c_void_p]
        l.conflictSize.restype = c_int
        l.unsatCore.argtypes = [c_void_p, c_int, c_void_p, c_int]
        l.unsatCore.restype = c_int
        l.modelValue.argtypes = [c_void_p, c_int]
        l.modelValue.restype = c_int
        l.fillModel.argtypes = [c_void_p, c_void_p, c_int, c_int]
        l.getModelTrues.argtypes = [c_void_p, c_void_p, c_int, c_int, c_int]
        l.getModelTrues.restype = c_int

        l.getImplies.argtypes = [c_void_p, c_void_p]
        l.getImplies.restype = c_int
        l.getImplies_assumptions.argtypes = [c_void_p, c_void_p, c_void_p, c_int]
        l.getImplies_assumptions.restype = c_int

        l.setPreference.argtypes = [c_void_p, c_int, c_int]
        l.removePreference.argtypes = [c_void_p, c_int]

    def __del__(self):  # type: () -> None
        """Delete the Solver object"""
        self.lib.Solver_delete(self.s)

    @staticmethod
    def _to_intptr(a):  # type: (array.array) -> Tuple[int, int]
        """Helper function to get a ctypes POINTER(c_int) for an array"""
        addr, size = a.buffer_info()
        return ctypes.cast(addr, ctypes.POINTER(c_int)), size

    @staticmethod
    def _get_array(seq):  # type: (Iterable[int]) -> array.array
        """Helper function to turn any iterable into an array (unless it already is one)"""
        if isinstance(seq, array.array):
            return seq
        else:
            return array.array('i', seq)

    def new_vars(self, n):
        while n > self.nvars():
            self.new_var()

    def new_var(self, polarity=None, dvar=True):  # type: (bool, bool) -> int
        """Create a new variable in the solver.

        Args:
            polarity (bool):
              The default polarity for this variable.  True = variable's
              default is True, etc.  Note that this is the reverse of the 'user
              polarity' in MiniSat, where True indicates the *sign* is True.
              The default, None, creates the variable using Minisat's default,
              which assigns a variable False at first, but then may change that
              based on the phase-saving setting.
            dvar (bool):
              Whether this variable will be used as a decision variable.

        Returns:
            The new variable's index (0-based counting).
        """

        if polarity is None:
            pol_int = 2  # lbool l_Undef
        elif polarity is True:
            pol_int = 1  # lbool l_False (hence, the *sign* is false, so the literal is true)
        elif polarity is False:
            pol_int = 0  # lbool l_True (hence the literal is false)
        return self.lib.newVar(self.s, pol_int, dvar)

    def nvars(self):  # type: () -> int
        '''Get the number of variables created in the solver.'''
        return self.lib.nVars(self.s)

    def nclauses(self):  # type: () -> int
        '''Get the number of clauses or constraints added to the solver.'''
        return self.lib.nClauses(self.s)

    def set_phase_saving(self, ps):  # type: (int) -> None
        '''Set the level of phase saving (0=none, 1=limited, 2=full (default)).'''
        self.lib.setPhaseSaving(self.s, ps)

    def set_rnd_pol(self, val):  # type: (bool) -> None
        '''Set whether random polarities are used for decisions (overridden if vars are created with a user polarity other than None)'''
        self.lib.setRndPol(self.s, val)

    def set_rnd_init_act(self, val):  # type: (bool) -> None
        '''Set whether variables are intialized with a random initial activity.
           (default: False)'''
        self.lib.setRndInitAct(self.s, val)

    def set_rnd_seed(self, seed):  # type: (float) -> None
        '''Set the solver's random seed to the given double value.  Cannot be 0.0.'''
        assert(seed != 0.0)
        self.lib.setRndSeed(self.s, seed)

    def add_clause(self, lits):  # type: (Sequence[int]) -> bool
        """Add a clause to the solver.

        Args:
            lits:
              A sequence of literals as integers.  Each integer specifies a
              variable with *1*-based counting and a sign via the sign of the
              integer.  Ex.: [-1, 2, -3] is (!x0 + x1 + !x2)

        Returns:
            A boolean value returned from MiniSat's ``addClause()`` function,
            indicating success (True) or conflict (False).
        """
        if not all(abs(x) <= self.nvars() for x in lits):
            raise Exception("Not all variables in %s are created yet.  Call new_var() first." % lits)
        if len(lits) > 1:
            a = self._get_array(lits)
            a_ptr, size = self._to_intptr(a)
            return self.lib.addClause(self.s, size, a_ptr)
        elif len(lits) == 1:
            (lit,) = lits   # extract one item whether list or set
            return self.lib.addUnit(self.s, lit)
        else:
            return self.lib.addClause(self.s, 0, None)

    def check_complete(self, positive_lits=None, negative_lits=None):  # type: (Sequence[int], Sequence[int]) -> bool
        """Check whether a given complete assignment satisfies the current set
        of clauses.  For efficiency, it may be given just the positive literals
        or just the negative literals.

        Args:
            positive_lits, negative_lits:
              Optional sequences (exactly one must be specified) containing
              literals as integers, specified as in `add_clause()`.  If
              positive literals are given, the assignment will be completed
              assuming all other variables are negative, and vice-versa if
              negative literals are given.

        Returns:
            True if the assignment satisfies the current clauses, False otherwise.
        """
        if positive_lits is not None:
            a = self._get_array(positive_lits)
            a_ptr, size = self._to_intptr(a)
            return self.lib.check_complete(self.s, size, a_ptr, True)
        elif negative_lits is not None:
            a = self._get_array(negative_lits)
            a_ptr, size = self._to_intptr(a)
            return self.lib.check_complete(self.s, size, a_ptr, False)
        else:
            raise Exception("Either positive_lits or negative_lits must be specified in check_complete().")

    def solve(self, assumptions=None):  # type: (Sequence[int]) -> bool
        """Solve the current set of clauses, optionally with a set of assumptions.

        Args:
            assumptions:
              An optional sequence of literals as integers, specified as in
              `add_clause()`.

        Returns:
            True if the clauses (and assumptions) are satisfiable, False otherwise.
        """
        if assumptions is None:
            return self.lib.solve(self.s)
        else:
            a = self._get_array(assumptions)
            a_ptr, size = self._to_intptr(a)
            return self.lib.solve_assumptions(self.s, size, a_ptr)

    def simplify(self):  # type: () -> bool
        '''Call Solver.simplify().'''
        return self.lib.simplify(self.s)

    def get_model(self, start=0, end=-1):  # type: (int, int) -> array.array
        """Get the current model from the solver, optionally retrieving only a slice.

        Args:
            start, end (int):
              Optional start and end indices, interpreted as in ``range()``.

        Returns:
            An array of booleans indexed to each variable (from 0).  If a start
            index was given, the returned list starts at that index (i.e.,
            ``get_model(10)[0]`` is index 10 from the solver's model.
        """
        if end == -1:
            end = self.nvars()
        a = array.array('i', [-1] * (end-start))
        a_ptr, size = self._to_intptr(a)
        self.lib.fillModel(self.s, a_ptr, start, end)
        return a

    def get_model_trues(self, start=0, end=-1, offset=0):  # type: (int, int, int) -> array.array
        """Get variables assigned true in the current model from the solver.

        Args:
            start, end (int):
              Optional start and end indices, interpreted as in ``range()``.
            offset (int):
              Optional offset to be added to the zero-based variable numbers
              from MiniSat.

        Returns:
            An array of true variables in the solver's current model.  If a
            start index was given, the variables are indexed from that value.
            """
        if end == -1:
            end = self.nvars()
        a = array.array('i', [-1] * (end-start))
        a_ptr, size = self._to_intptr(a)
        count = self.lib.getModelTrues(self.s, a_ptr, start, end, offset)
        # reduce the array down to just the valid indexes
        return a[:count]

    def block_model(self):
        """Block the current model from the solver."""
        model = self.get_model()
        self.add_clause([-(x+1) if model[x] > 0 else x+1 for x in range(1, len(model))])

    def model_value(self, i):  # type: (int) -> bool
        '''Get the value of a given variable in the current model.'''
        return self.lib.modelValue(self.s, i)

    def implies(self, assumptions=None):  # type(Sequence[int]) -> array.array
        """Get literals known to be implied by the current formula.  (I.e., all
        assignments made at level 0.)

        Args:
            assumptions:
              An optional sequence of literals as integers, specified as
              in `add_clause()`.

        Returns:
            An array of literals implied by the current formula (and optionally
            the given assumptions).
        """
        res = array.array('i', [-1] * self.nvars())
        res_ptr, _ = self._to_intptr(res)

        if assumptions is None:
            count = self.lib.getImplies(self.s, res_ptr)
        else:
            assumps = self._get_array(assumptions)
            assumps_ptr, assumps_size = self._to_intptr(assumps)
            count = self.lib.getImplies_assumptions(self.s, res_ptr, assumps_ptr, assumps_size)

        # reduce the array down to just the valid indexes
        return res[:count]

    def set_preference(self, l, level):
        """Set the preference for literal l

        Args:
            l (int):
              A literal.

            level (int):
                The importance of the literal (the highest level is chosen before any other level).
        Returns:
            Nothing.
        """
        if abs(l) > self.nvars():
            raise Exception("Variable %s is not created yet.  Call new_var() first." % abs(l))
        self.lib.setPreference(self.s,l,level)

    def remove_preference(self, v):
        """Remove the preference for variable v

        Args:
            v (int):
              A variable.
        Returns:
            Nothing.
        """
        if abs(v) > self.nvars():
            raise Exception("Variable %s is not created yet.  Call new_var() first." % v)
        self.lib.removePreference(self.s,v)

class MinisatSolver(Solver):
    """A Python analog to MiniSat's Solver class.

    >>> S = MinisatSolver()

    Create variables using `new_var()`.  Add clauses as sequences of literals
    with `add_clause()`, analogous to MiniSat's ``addClause()``.  Literals are
    specified as integers, with the magnitude indicating the variable index
    (with 1-based counting) and the sign indicating True/False.  For example,
    to add clauses (x0), (!x1), (!x0 + x1 + !x2), and (x2 + x3):

    >>> for i in range(4):
    ...     S.new_var()  # doctest: +ELLIPSIS
    0
    1
    2
    3
    >>> for clause in [1], [-2], [-1, 2, -3], [3, 4]:
    ...     S.add_clause(clause)  # doctest: +ELLIPSIS
    True
    True
    True
    True

    The `solve()` method returns True or False just like MiniSat's.

    >>> S.solve()
    True

    Models are returned as arrays of Booleans, indexed by var.
    So the following represents x0=True, x1=False, x2=False, x3=True.

    >>> list(S.get_model())
    [1, 0, 0, 1]

    The `add_clause()` method may return False if a conflict is detected
    when adding the clause, even without search.

    >>> S.add_clause([-4])
    False
    >>> S.solve()
    False
    """
    def __init__(self):  # type: () -> None
        super(MinisatSolver, self).__init__("libminisat.so")

class Dimacs:
    def __init__(self):
        self.clauses = []
        self.softClauses = []
        self.weights = []
        self.nVars = 0
        self.nClauses = 0
        self.nSoftClauses = 0        
        self.maxWeight = None

    def parse(self, filename):
        lines = open(filename,'r').readlines()
        maxVar = 0
        for line in lines:
            l = line.rstrip().split()
            if len(l) == 0:
                pass
            elif l[0] == 'p':
                assert l[1]== "cnf" or l[1]== "wcnf"
                self.nVars=int(l[2])
                self.nClauses=int(l[3])
                if l[1]== "wcnf":
                    self.maxWeight=int(l[4])                    
            elif l[0] == 'c':
                pass
            else:
                assert l[-1]=='0'
                if self.maxWeight == None:
                    clause = []
                    for i in l[:-1]:
                        maxVar = max(maxVar, abs(int(i)))
                        clause.append(int(i))
                    self.clauses.append(clause)
                else:
                    clause = []
                    for i in l[1:-1]:
                        maxVar = max(maxVar, abs(int(i)))
                        clause.append(int(i))                        
                    if self.maxWeight == int(l[0]):
                        self.clauses.append(clause)                        
                    else:
                        self.softClauses.append(clause)
                        self.weights.append(int(l[0]))
        if maxVar != self.nVars:
            sys.stderr.write("Warning: Mismatch of variables: read %s expected %s\n" % (maxVar,self.nVars))
        if len(self.clauses)+len(self.softClauses) != self.nClauses:
            sys.stderr.write("Warning: Mismatch of clauses: read %s expected %s\n" % (len(self.clauses),self.nClauses))

        self.nVars = maxVar
        self.nClauses = len(self.clauses)
        self.nSoftClauses = len(self.softClauses)

    def all_clauses(self):
        for clause in self.clauses:
            yield clause

    def n_vars(self):
        return self.nVars

    def n_clauses(self):
        return self.nClauses

    def print(self):
        print ("p cnf %s %s" % (self.nVars, self.nClauses))
        for clause in self.clauses:
            output = str(clause[0])
            for i in range(1,len(clause)):
                output += " " + str(clause[i])
            print ("%s 0" % output)
