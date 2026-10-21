"""The package integrates with the other parts by data contract only.

No module may import another team's package at import time, so this branch
merges and runs independently of ``faultattribution``, ``benchmark`` and
``orchestrator``. The benchmark adapter may import the benchmark's Directive
lazily, inside a function, when the caller did not pass it in.
"""
import ast
import unittest
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]
FOREIGN = ('agentrx', 'benchmark', 'recovery_orchestrator', 'tool_prevention')
LAZY_ALLOWED = {('adapters/benchmark.py', 'benchmark.env.workflow')}


def imports(tree):
    """(module, top_level) for every import; top_level is False inside functions."""
    found = []

    def visit(node, top_level):
        for child in ast.iter_child_nodes(node):
            nested = top_level and not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda))
            if isinstance(child, ast.Import):
                found.extend((alias.name, top_level) for alias in child.names)
            elif isinstance(child, ast.ImportFrom) and child.level == 0 and child.module:
                found.append((child.module, top_level))
            visit(child, nested)

    visit(tree, True)
    return found


class BoundaryTest(unittest.TestCase):
    def test_no_foreign_imports(self):
        for path in sorted(PACKAGE.rglob('*.py')):
            relative = path.relative_to(PACKAGE).as_posix()
            if relative.startswith('tests/'):
                continue
            for module, top_level in imports(ast.parse(path.read_text(encoding='utf-8'))):
                if module.split('.')[0] not in FOREIGN:
                    continue
                self.assertFalse(top_level, f'{relative} imports {module} at import time')
                self.assertIn((relative, module), LAZY_ALLOWED, f'{relative} imports {module}')

    def test_standard_library_only(self):
        import sys
        stdlib = set(sys.stdlib_module_names)
        for path in sorted(PACKAGE.rglob('*.py')):
            if path.relative_to(PACKAGE).as_posix().startswith('tests/'):
                continue
            for module, _ in imports(ast.parse(path.read_text(encoding='utf-8'))):
                root = module.split('.')[0]
                self.assertTrue(root in stdlib or root in FOREIGN or root == '__future__',
                                f'{path.name} depends on third-party package {module}')


if __name__ == '__main__':
    unittest.main()
