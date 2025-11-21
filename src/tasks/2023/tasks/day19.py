"""
--- Day 19: Aplenty  ---
https://adventofcode.com/2023/day/19
"""
import dataclasses
from copy import deepcopy
from operator import gt, lt
from typing import Callable

from src.utils.test_and_run import run, test


@dataclasses.dataclass
class Clause:
    attr: str
    comparator: Callable
    arg: int
    output: str


@dataclasses.dataclass
class Workflow:
    name: str
    clauses: list[Clause]
    fallback: str
    workflow_by_name: dict[str, "Workflow"] = None
    constraints: dict[str, list[list[int, int]]] = None
    min_arg = 1
    max_arg = 4000

    @classmethod
    def from_line(cls, line):
        name, _schema = line.split("{")
        assert _schema[-1] == "}"
        schemas = _schema[:-1].split(",")
        fallback = schemas[-1]

        clauses = []
        for schema in schemas[:-1]:
            raw_clause, output = schema.split(":")
            if ">" in raw_clause:
                comparator = gt
                comparator_str = ">"
            elif "<" in raw_clause:
                comparator = lt
                comparator_str = "<"
            else:
                raise ValueError(f"no comparator found in {raw_clause}")

            attr, arg = raw_clause.split(comparator_str)
            clause = Clause(attr, comparator, int(arg), output)
            clauses.append(clause)

        return cls(name, clauses, fallback)

    def run(self, part):
        for clause in self.clauses:
            value = part[clause.attr]
            comparator_result = clause.comparator(value, clause.arg)
            if comparator_result:
                return clause.output
        return self.fallback

    def _cut_intervals(self, constraints, attr, comparator_str, arg):
        intervals = constraints[attr]

        intervals_to_exclude = []
        intervals_to_add = []
        for interval in intervals:
            s, e = interval
            # todo consider >

            if s < arg < e:
                if comparator_str == "<":
                    # intervals_to_exclude.append(interval)
                    #
                    # new_interval = [s, arg]
                    #
                    # intervals_to_add.append()

                    interval[1] = arg - 1
                else:
                    interval[0] = arg + 1
            else:
                a = 0

        return intervals

    def _merge(self, constraints_orig, constraints2, operator):
        assert operator in ("and", "or")

        if constraints2 == constraints_orig:
            return constraints_orig

        constraints = deepcopy(constraints_orig)
        if operator == "or":
            # check if we can squash smth here
            # find if smth overlaps.
            # if absolutely nothing - add absolutely new constraint
            found = False
            for c1 in constraints:
                for c2 in constraints2:
                    for k, intervals2 in c2.items():
                        intervals1 = c1[k]
                        if intervals1 == intervals2:
                            found = True
                            continue

                        for interval2 in intervals2:
                            found = False  # interval to cut
                            s2, e2 = interval2
                            for interval1 in intervals1:
                                s1, e1 = interval1
                                if s1 < s2 < e1:
                                    s1 = s2
                                    interval1[0] = s1
                                    found = True
                                if s1 < e2 < e1:
                                    e1 = e2
                                    interval1[1] = e1
                                    found = True

                            if not found:
                                intervals1.append(interval2)

        else:
            for c1 in constraints:
                for c2 in constraints2:
                    for k, intervals2 in c2.items():
                        intervals1 = c1[k]
                        if intervals1 == intervals2:
                            continue

                        for interval2 in intervals2:
                            found = False  # interval to cut
                            s2, e2 = interval2
                            for interval1 in intervals1:
                                s1, e1 = interval1
                                if s1 < s2 < e1:
                                #     s1 = s2
                                    interval1[0] = s1
                                    found = True
                                if s1 < e2 < e1:
                                #     e1 = e2
                                    interval1[1] = e1
                                    found = True
                                # interval1[0] = max(s1, s2)
                                # interval1[1] = min(e1, e2)

                            if not found:
                            #     # intervals1.append(interval2)
                                a = 0

        return constraints

    def get_null_constraints(self):
        return deepcopy([{
            k: [[self.min_arg, self.max_arg]] for k in "xmas"
        }])

    def get_acceptance_clause(self):
        if self.constraints:
            return self.constraints

        name = self.name
        # attr name to intervals of possible values (inclusive)
        constraints = self.get_null_constraints()

        comparator_to_str = {
            gt: ">",
            lt: "<",
        }

        full_acceptance_clause = ""
        next_operator = "and"
        for clause in self.clauses:

            attr = clause.attr
            comparator_str = comparator_to_str[clause.comparator]
            arg = clause.arg
            clause_str = f"part[{attr!r}] {comparator_str} {arg}"
            output = clause.output

            if comparator_str == ">":
                clause_interval = [arg + 1, self.max_arg]
            else:
                clause_interval = [self.min_arg, arg - 1]
            clause_constraints = self.get_null_constraints()
            clause_constraints[0][attr] = [clause_interval]

            match output:
                case "A":
                    # self._cut_intervals(constraints, attr, comparator_str, arg)
                    constraints = self._merge(constraints, clause_constraints, next_operator)
                    next_operator = "or"
                case "R":
                    clause_str = f"not {clause_str}"
                    full_acceptance_clause = self._cut_intervals(full_acceptance_clause, clause_str, next_operator)
                    next_operator = "or"
                case _:
                    output_constraints = self.workflow_by_name[output].get_acceptance_clause()
                    case_constraints = self._merge(output_constraints, clause_constraints, "and")
                    constraints = self._merge(constraints, case_constraints, next_operator)

            next_operator = "or"

        # add fallback
        if self.fallback == "R":
            a = 0
            pass
        elif self.fallback == "A":
            a = 0
            pass
        else:
            w_fallback = self.workflow_by_name[self.fallback]
            fallback_constraints = w_fallback.get_acceptance_clause()
            # TODO and / or?
            constraints = self._merge(constraints, fallback_constraints, "or")

        self.constraints = constraints
        return constraints

class Part(dict):
    @classmethod
    def from_line(cls, line):
        data = {}
        for part in line[1:-1].split(","):
            k, v = part.split("=")
            data[k] = int(v)

        return cls(data)


class Aplenty:
    def __init__(self, inp):
        self.raw_inp = inp
        self.workflows, self.parts = self._parse_input(inp)

        self.workflow_by_name = {
            w.name: w for w in self.workflows
        }
        for w in self.workflows:
            w.workflow_by_name = self.workflow_by_name

    @staticmethod
    def _parse_input(inp):
        part = 1
        workflows = []
        parts = []
        for line in inp:
            if not line:
                part = 2
                continue

            if part == 1:
                workflows.append(Workflow.from_line(line))
            if part == 2:
                parts.append(Part.from_line(line))

        return workflows, parts

    def get_sum_of_accepted_parts_ratings(self):
        accepted_parts = []

        for part in self.parts:
            w_name = "in"
            while True:
                w = self.workflow_by_name[w_name]

                w_res = w.run(part)
                match w_res:
                    case "A":
                        accepted_parts.append(part)
                        break
                    case "R":
                        # rejected
                        break
                    case _:
                        w_name = w_res

        ratings = []
        for part in accepted_parts:
            ratings.append(sum(part.values()))

        return sum(ratings)

    def get_distinct_combinations_of_accepted_ratings(self):
        for w in self.workflows:
            clauses = w.get_acceptance_clause()
            a = 0

def aplenty(inp, part=1, **kw):
    if part == 1:
        return Aplenty(inp).get_sum_of_accepted_parts_ratings()
    Aplenty(inp).get_distinct_combinations_of_accepted_ratings()


if __name__ == "__main__":
    test(aplenty, 19114)
    assert run(aplenty) == 487623

    test(aplenty, part=2, expected=167409079868000)
    run(aplenty, part=2)
