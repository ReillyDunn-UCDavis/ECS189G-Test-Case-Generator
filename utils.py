import re
import ast
import textwrap
import inspect


EDGE_CASES = {
    "str":       ['""', '"a"', '"aa"', '"ab"'],
    "int":       ["0", "1", "-1"],
    "List[int]": ["[]", "[0]", "[1, 1]", "[1, -1]"],
    "List[str]": ['[]', '["a"]'],
    "bool":      ["True", "False"],
}


def generate_edge_case_call(solution_code: str, func_name: str) -> str | None:
    """
    Builds one edge-case input call using the simplest known value for
    each parameter type. Used to supplement training data.
    """
    params = extract_params_with_types(solution_code, func_name)
    if not params:
        return None

    parts = []
    for name, type_str in params:
        pool = next(
            (v for k, v in EDGE_CASES.items() if k in type_str),
            ["0"]
        )
        parts.append(f"{name}={pool[0]}")   # always take the simplest value

    call = f"candidate({', '.join(parts)})"
    return call if is_valid_python(call) else None


def extract_params_with_types(solution_code: str, func_name: str) -> list[tuple[str, str]]:
    """
    Returns [(param_name, type_str), ...] for the given function, excluding self.
    Type is taken from the annotation if present, else "unknown".

    Example: def twoSum(self, nums: List[int], target: int)
          -> [("nums", "List[int]"), ("target", "int")]
    """
    ns = {}
    try:
        exec(textwrap.dedent(solution_code), ns)
        func = getattr(ns["Solution"], func_name)
        hints = {}
        try:
            import typing
            hints = typing.get_type_hints(func)
        except Exception:
            pass
        sig = inspect.signature(func)
        result = []
        for name, param in sig.parameters.items():
            if name == "self":
                continue
            if name in hints:
                type_str = str(hints[name])
                type_str = type_str.replace("typing.", "")
            elif param.annotation is not inspect.Parameter.empty:
                type_str = str(param.annotation)
            else:
                type_str = "unknown"
            result.append((name, type_str))
        return result
    except Exception:
        return []
    

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
        stripped = re.sub(r"\s*:\s*$", ":", stripped)       # fix "def f(s) :" -> "def f(s):"
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
    for line in solution_code.split("\n"):
        stripped = line.strip()
        if stripped.startswith("def "):
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
        if "Solution" in ns:
            func = getattr(ns["Solution"], func_name)
        elif func_name and func_name in ns:
            func = ns[func_name]
        else:
            return set()
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

def extract_body(solution_code: str) -> str:
    """
    Extracts the method body from the Solution class, dedented so it
    aligns cleanly in the prompt.
    
    Example:
        d = {}
        for i, x in enumerate(nums):
            ...
    """
    import textwrap
    lines = []
    inside_method = False

    for line in solution_code.split("\n"):
        stripped = line.strip()
        if stripped.startswith("def ") and "self" in stripped:
            inside_method = True
            continue
        if not inside_method:
            continue
        # Stop at next method or class definition
        if stripped.startswith("def ") or stripped.startswith("class "):
            break
        lines.append(line)

    return textwrap.dedent("\n".join(lines)).strip()