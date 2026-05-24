import re
import ast
import textwrap
import inspect

def is_valid_python(line):
    try:
        ast.parse(line)
        return True
    except:
        return False
    

def normalize_whitespace(line: str) -> str:
    return re.sub(r"\s+", " ", line).strip()


def simplify_assert(line: str) -> str:
    """
    Converts sorted(candidate(...)) == sorted([...]) form
    into the simpler candidate(...) == [...] form.
    Falls through unchanged if pattern doesn't match.
    """
    m = re.match(
        r"assert sorted\(candidate\((.*?)\)\)\s*==\s*sorted\(\[(.*?)\]\)$",
        line
    )
    if m:
        args, expected = m.group(1), m.group(2)
        return f"assert candidate({args}) == [{expected}]"
    return line


def extract_input_args(assert_line: str) -> str | None:
    """
    From: assert candidate(s="abcab") == 3
    Returns: candidate(s="abcab")
    Returns None if pattern doesn't match.
    """
    m = re.match(r"assert (candidate\(.*?\)) ==", assert_line)
    if m:
        return m.group(1)
    return None


def extract_signature(solution_code: str) -> str:
    """
    Extracts the function signature(s) from solution code,
    stripping self, type annotations, and return types.
    Example: "def twoSum(self, nums: List[int], target: int) -> List[int]:"
          -> "def twoSum(nums, target):"
    """
    lines = []
    for line in solution_code.split("\n"):
        stripped = line.strip()
        if not stripped.startswith("def "):
            continue
        stripped = re.sub(r"self,\s*", "", stripped)        # remove self
        stripped = re.sub(r"->\s*[\w\[\], ]+\s*:", ":", stripped)  # remove return type
        stripped = re.sub(r":\s*[\w\[\], ]+", "", stripped) # remove param types
        stripped = re.sub(r"\(,\s*", "(", stripped)         # fix leftover (, from self removal
        lines.append(stripped)
    return "\n".join(lines)


def extract_func_name(solution_code: str) -> str | None:
    """
    Extracts the method name from the Solution class.
    Returns None if not found.
    """
    for line in solution_code.split("\n"):
        stripped = line.strip()
        if stripped.startswith("def ") and "self" in stripped:
            m = re.match(r"def (\w+)\(", stripped)
            if m and m.group(1) != "__init__":
                return m.group(1)
    return None


def get_expected_output(solution_code: str, func_name: str, input_call: str):
    """
    Executes the solution on a given input call and returns the result.
    
    solution_code: full class Solution: ... string
    func_name: e.g. "twoSum"
    input_call: e.g. 'candidate(nums=[2,7], target=9)'
    
    Returns the result, or raises an exception if execution fails.
    """
    ns = {}
    exec(textwrap.dedent(solution_code), ns)
    sol = ns["Solution"]()
    func = getattr(sol, func_name)
    # Replace 'candidate' with the actual bound method
    result = eval(input_call.replace("candidate", "func"), {"func": func})
    return result


def build_test_assertion(solution_code: str, func_name: str, input_call: str) -> str | None:
    """
    Combines get_expected_output with input_call to produce a full assert.
    Returns None if execution fails (bad input, exception in solution, etc).
    
    Example output: 'assert candidate(nums=[2,7], target=9) == [0, 1]'
    """
    try:
        result = get_expected_output(solution_code, func_name, input_call)
        return f"assert {input_call} == {repr(result)}"
    except Exception as e:
        print(f"    [exec error] {input_call!r} → {type(e).__name__}: {e}")
        return None


def extract_param_names(solution_code: str, func_name: str) -> set[str]:
    """
    Returns the set of valid parameter names for the given function,
    excluding 'self'.
    Example: "def twoSum(self, nums, target):" -> {"nums", "target"}
    """
    ns = {}
    try:
        exec(textwrap.dedent(solution_code), ns)
        sol_class = ns["Solution"]
        func = getattr(sol_class, func_name)
        params = inspect.signature(func).parameters
        return {p for p in params if p != "self"}
    except Exception:
        return set()


def call_uses_valid_params(input_call: str, valid_params: set[str]) -> bool:
    """
    Checks that every keyword argument in the call is a real parameter.
    Example: candidate(s="abc", k=1) against {"s"} -> False
    """
    # Extract all kwarg names from the call string
    kwarg_names = set(re.findall(r"(\w+)\s*=", input_call))
    return kwarg_names.issubset(valid_params)