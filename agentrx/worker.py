"""Disposable Python-check worker. Not a general-purpose security sandbox."""
import ast
import builtins
import contextlib
import importlib
import io
import json
import math
import sys
from types import SimpleNamespace

MODULE_EXPORTS = {
    'json': 'loads dumps JSONDecodeError',
    're': 'compile search match fullmatch findall finditer split sub subn escape Pattern Match RegexFlag error I M S X A U IGNORECASE MULTILINE DOTALL VERBOSE ASCII UNICODE',
    'math': 'acos acosh asin asinh atan atan2 atanh ceil comb copysign cos cosh degrees dist e erf erfc exp expm1 fabs factorial floor fmod frexp fsum gamma gcd hypot inf isclose isfinite isinf isnan isqrt lcm ldexp lgamma log log10 log1p log2 modf nan nextafter perm pi pow prod radians remainder sin sinh sqrt tan tanh tau trunc ulp',
    'statistics': 'mean fmean geometric_mean harmonic_mean median median_low median_high median_grouped mode multimode quantiles variance pvariance stdev pstdev StatisticsError NormalDist correlation covariance linear_regression',
    'collections': 'Counter defaultdict deque OrderedDict ChainMap namedtuple',
    'datetime': 'date datetime time timedelta timezone MINYEAR MAXYEAR',
    'itertools': 'accumulate batched chain combinations combinations_with_replacement compress count cycle dropwhile filterfalse groupby islice pairwise permutations product repeat starmap takewhile tee zip_longest',
    'functools': 'reduce partial cmp_to_key',
    'decimal': 'Decimal Context DecimalException InvalidOperation DivisionByZero Overflow Underflow getcontext localcontext ROUND_CEILING ROUND_DOWN ROUND_FLOOR ROUND_HALF_DOWN ROUND_HALF_EVEN ROUND_HALF_UP ROUND_UP ROUND_05UP',
    'fractions': 'Fraction',
    'operator': 'add sub mul matmul truediv floordiv mod pow neg pos abs eq ne lt le gt ge not_ truth is_ is_not and_ or_ xor invert lshift rshift concat contains countOf indexOf getitem itemgetter index length_hint',
}
ALLOWED_MODULES = set(MODULE_EXPORTS)
SAFE_BUILTINS = ('abs all any bool dict enumerate filter float int isinstance issubclass '
                 'len list map max min next range reversed round set slice sorted str sum '
                 'tuple zip print Exception ValueError TypeError KeyError IndexError '
                 'RuntimeError AssertionError StopIteration').split()


def check_source(source):
    if len(source) > 100_000:
        raise ValueError('Generated check exceeds source limit')
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(n.name not in ALLOWED_MODULES for n in node.names):
                raise ValueError('Unsupported import')
        elif isinstance(node, ast.ImportFrom):
            if node.level or node.module not in ALLOWED_MODULES or any(
                    n.name not in MODULE_EXPORTS[node.module].split() for n in node.names):
                raise ValueError('Unsupported import')
        elif isinstance(node, ast.Attribute) and (node.attr.startswith(('_', 'f_', 'gi_', 'cr_', 'ag_', 'tb_'))
                                                 or node.attr in ('format', 'format_map')):
            raise ValueError('Private, frame, and dynamic formatting attributes are forbidden')
        elif isinstance(node, ast.Name) and node.id.startswith('__'):
            raise ValueError('Private names are forbidden')
        elif isinstance(node, (ast.ClassDef, ast.AsyncFunctionDef, ast.Global, ast.Nonlocal)):
            raise ValueError('Unsupported generated construct')
    return tree


def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level or name not in ALLOWED_MODULES:
        raise ImportError('Unsupported import')
    module = importlib.import_module(name)
    # Raw modules expose implementation details such as statistics.sys and
    # json.decoder. Only explicitly approved exports enter generated code.
    return SimpleNamespace(**{key: getattr(module, key) for key in MODULE_EXPORTS[name].split()
                              if hasattr(module, key)})


class BoundedLog(io.StringIO):
    def write(self, value):
        if self.tell() + len(value) > 16_384:
            raise ValueError('Generated check output limit exceeded')
        return super().write(value)


def main():
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
        resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
        if sys.platform.startswith('linux'):
            resource.setrlimit(resource.RLIMIT_AS, (512*1024*1024, 512*1024*1024))
    except (ImportError, OSError, ValueError):
        pass
    result = None
    try:
        payload = json.load(sys.stdin)
        tree = check_source(payload['code'])
        safe = {name: getattr(builtins, name) for name in SAFE_BUILTINS}
        safe['__import__'] = safe_import
        env = {'__builtins__': safe}
        with contextlib.redirect_stdout(BoundedLog()), contextlib.redirect_stderr(BoundedLog()):
            exec(compile(tree, '<generated-check>', 'exec'), env)
            fn = env[payload['function_name']]
            value = fn(payload['trajectory'], payload['offset'])
        if type(value) is not bool:
            raise ValueError('Python check must return a boolean')
        result = {'value': value}
    except BaseException as exc:
        result = {'error': f'{type(exc).__name__}: {str(exc)[:1000]}'}
    print(json.dumps(result))


if __name__ == '__main__':
    main()
