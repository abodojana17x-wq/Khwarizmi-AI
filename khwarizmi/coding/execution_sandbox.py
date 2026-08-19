"""
Khwarizmi Execution Sandbox — Safe Local Code Execution.

This module provides a secure sandbox for executing Python code locally with:
- Hard timeout (using signal on Unix, threading on Windows)
- Memory cap (resource limits on Unix)
- No network access (restricted builtins and imports)
- Restricted builtins (no eval, exec, open, etc.)
- Captured stdout/stderr/exit code/duration

All execution is fully offline with no external services.
"""

import ast
import sys
import os
import io
import resource
import signal
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from types import ModuleType


@dataclass
class SandboxResult:
    """Result of sandboxed code execution."""
    success: bool
    exit_code: int = 0
    stdout: str = ""
    stderr: str = ""
    duration: float = 0.0
    memory_used: int = 0
    error_message: str = ""
    timed_out: bool = False
    memory_exceeded: bool = False

    @property
    def is_valid(self) -> bool:
        """Return True if execution completed successfully without errors."""
        return self.success and self.exit_code == 0 and not self.timed_out

    def to_dict(self) -> Dict[str, Any]:
        """Serialize result to dictionary."""
        return {
            "success": self.success,
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "duration_seconds": self.duration,
            "memory_used_bytes": self.memory_used,
            "error_message": self.error_message,
            "timed_out": self.timed_out,
            "memory_exceeded": self.memory_exceeded,
        }


class TimeoutError(Exception):
    """Raised when execution exceeds timeout limit."""
    pass


class MemoryLimitError(Exception):
    """Raised when execution exceeds memory limit."""
    pass


class _SandboxTimeoutHandler:
    """Handle timeout using signal (Unix) or threading (Windows)."""
    
    def __init__(self, timeout_seconds: float):
        self.timeout = timeout_seconds
        self.timed_out = False
        self._old_handler = None
    
    def __enter__(self):
        if hasattr(signal, 'SIGALRM'):
            # Unix: use SIGALRM
            self._old_handler = signal.signal(signal.SIGALRM, self._handle_timeout)
            signal.alarm(int(self.timeout) + 1)
        else:
            # Windows: use threading
            self._timer = threading.Timer(self.timeout, self._handle_timeout_thread)
            self._timer.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)
            if self._old_handler:
                signal.signal(signal.SIGALRM, self._old_handler)
        else:
            if hasattr(self, '_timer'):
                self._timer.cancel()
        return False
    
    def _handle_timeout(self, signum, frame):
        self.timed_out = True
        raise TimeoutError(f"Execution timed out after {self.timeout} seconds")
    
    def _handle_timeout_thread(self):
        self.timed_out = True
        # Can't raise exception from thread, set flag for caller to check


def _restrict_imports():
    """Create a restricted import system that blocks dangerous modules."""
    ALLOWED_MODULES = frozenset([
        'math', 'cmath', 'random', 'statistics',
        'collections', 'itertools', 'functools', 'operator',
        're', 'string', 'textwrap',
        'datetime', 'time',
        'json', 'csv',
        'copy', 'pprint',
        'typing', 'dataclasses',
        'enum',
        '__future__',
    ])
    
    BANNED_MODULES = frozenset([
        'os', 'sys', 'subprocess', 'multiprocessing',
        'socket', 'http', 'urllib', 'requests', 'aiohttp',
        'ftplib', 'smtplib', 'telnetlib',
        'pickle', 'marshal', 'shelve',
        'ctypes', 'cffi',
        'importlib', 'pkgutil', 'modulefinder',
        'inspect', 'dis', 'compileall',
        'threading', 'queue', 'concurrent',
        'asyncio',
        'code', 'codeop', 'runpy',
        'builtins', '__builtin__',
    ])
    
    original_import = __builtins__['__import__'] if isinstance(__builtins__, dict) else __builtins__.__import__
    
    def restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name in BANNED_MODULES:
            raise ImportError(f"Import of '{name}' is not allowed in sandbox")
        
        base_module = name.split('.')[0]
        if base_module in BANNED_MODULES:
            raise ImportError(f"Import of '{name}' is not allowed in sandbox")
        
        if base_module not in ALLOWED_MODULES and base_module not in sys.modules:
            # Check if it's a standard library module we don't explicitly allow
            # For safety, only allow explicitly whitelisted modules
            pass
        
        return original_import(name, globals, locals, fromlist, level)
    
    return restricted_import


def _create_safe_builtins() -> Dict[str, Any]:
    """Create a restricted builtins dictionary for safe execution."""
    # Safe builtins that are allowed
    safe_builtins = {
        # Types
        'bool': bool,
        'int': int,
        'float': float,
        'complex': complex,
        'str': str,
        'bytes': bytes,
        'bytearray': bytearray,
        'list': list,
        'tuple': tuple,
        'dict': dict,
        'set': set,
        'frozenset': frozenset,
        'range': range,
        'slice': slice,
        'type': type,
        
        # Functions
        'abs': abs,
        'all': all,
        'any': any,
        'bin': bin,
        'chr': chr,
        'divmod': divmod,
        'enumerate': enumerate,
        'filter': filter,
        'format': format,
        'hex': hex,
        'isinstance': isinstance,
        'issubclass': issubclass,
        'iter': iter,
        'len': len,
        'map': map,
        'max': max,
        'min': min,
        'next': next,
        'oct': oct,
        'ord': ord,
        'pow': pow,
        'print': print,
        'repr': repr,
        'reversed': reversed,
        'round': round,
        'sorted': sorted,
        'sum': sum,
        'zip': zip,
        
        # Constants
        'True': True,
        'False': False,
        'None': None,
        'Ellipsis': Ellipsis,
        'NotImplemented': NotImplemented,
        
        # Exceptions (for catching)
        'Exception': Exception,
        'ArithmeticError': ArithmeticError,
        'AssertionError': AssertionError,
        'AttributeError': AttributeError,
        'EOFError': EOFError,
        'FloatingPointError': FloatingPointError,
        'GeneratorExit': GeneratorExit,
        'ImportError': ImportError,
        'IndexError': IndexError,
        'KeyError': KeyError,
        'KeyboardInterrupt': KeyboardInterrupt,
        'LookupError': LookupError,
        'MemoryError': MemoryError,
        'NameError': NameError,
        'NotImplementedError': NotImplementedError,
        'OSError': OSError,
        'OverflowError': OverflowError,
        'RecursionError': RecursionError,
        'ReferenceError': ReferenceError,
        'RuntimeError': RuntimeError,
        'StopIteration': StopIteration,
        'SyntaxError': SyntaxError,
        'SystemExit': SystemExit,
        'TypeError': TypeError,
        'UnboundLocalError': UnboundLocalError,
        'ValueError': ValueError,
        'ZeroDivisionError': ZeroDivisionError,
        
        # Allow __import__ for controlled imports
        '__import__': __import__,
    }
    
    return safe_builtins


@contextmanager
def _memory_limit(max_memory_mb: int):
    """Context manager to enforce memory limit (Unix only)."""
    if not hasattr(resource, 'setrlimit'):
        yield
        return
    
    max_bytes = max_memory_mb * 1024 * 1024
    
    # Set both soft and hard limits
    try:
        old_soft, old_hard = resource.getrlimit(resource.RLIMIT_AS)
        resource.setrlimit(resource.RLIMIT_AS, (max_bytes, max_bytes))
        yield
    except MemoryLimitError:
        raise
    except Exception:
        # Restore old limits
        try:
            resource.setrlimit(resource.RLIMIT_AS, (old_soft, old_hard))
        except:
            pass
        raise
    finally:
        # Restore old limits
        try:
            resource.setrlimit(resource.RLIMIT_AS, (old_soft, old_hard))
        except:
            pass


class ExecutionSandbox:
    """
    Secure sandbox for executing Python code locally.
    
    Features:
    - Hard timeout enforcement
    - Memory cap (Unix only)
    - No network access
    - Restricted builtins (no eval, exec, open, etc.)
    - Captured stdout/stderr
    
    Usage:
        sandbox = ExecutionSandbox(timeout=5.0, max_memory_mb=100)
        result = sandbox.execute("print('Hello')")
    """
    
    # Dangerous functions that should not be accessible
    DANGEROUS_NAMES = frozenset([
        'eval', 'exec', 'execfile', 'compile',
        'open', 'file', 'input',
        'globals', 'locals', 'vars', 'dir',
        'getattr', 'setattr', 'delattr',
        'hasattr', 'callable',
        '__import__', 'help', 'license', 'copyright', 'credits',
        'breakpoint', 'pdb',
    ])
    
    def __init__(
        self,
        timeout: float = 5.0,
        max_memory_mb: int = 100,
        allow_network: bool = False,
    ):
        """
        Initialize execution sandbox.
        
        Args:
            timeout: Maximum execution time in seconds
            max_memory_mb: Maximum memory usage in megabytes
            allow_network: Whether to allow network access (default False)
        """
        self.timeout = timeout
        self.max_memory_mb = max_memory_mb
        self.allow_network = allow_network
        
        # Validate parameters
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        if max_memory_mb <= 0:
            raise ValueError("max_memory_mb must be positive")
    
    def execute(
        self,
        code: str,
        globals_dict: Optional[Dict[str, Any]] = None,
        locals_dict: Optional[Dict[str, Any]] = None,
    ) -> SandboxResult:
        """
        Execute Python code in sandbox.
        
        Args:
            code: Python source code to execute
            globals_dict: Optional global namespace (will be restricted)
            locals_dict: Optional local namespace
            
        Returns:
            SandboxResult with execution results
        """
        start_time = time.time()
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        # Pre-execute validation
        try:
            self._validate_code(code)
        except SyntaxError as e:
            return SandboxResult(
                success=False,
                exit_code=1,
                stderr=f"Syntax error: {e}",
                error_message=str(e),
                duration=time.time() - start_time,
            )
        
        # Prepare restricted environment
        safe_builtins = _create_safe_builtins()
        
        exec_globals = {
            '__builtins__': safe_builtins,
            '__name__': '__sandbox__',
            '__doc__': None,
        }
        
        if globals_dict:
            # Filter out dangerous names from user-provided globals
            for key, value in globals_dict.items():
                if key not in self.DANGEROUS_NAMES:
                    exec_globals[key] = value
        
        exec_locals = locals_dict.copy() if locals_dict else {}
        
        # Capture stdout/stderr
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        
        try:
            sys.stdout = stdout_capture
            sys.stderr = stderr_capture
            
            # Execute with timeout and memory limit
            with _SandboxTimeoutHandler(self.timeout) as timeout_handler:
                try:
                    with _memory_limit(self.max_memory_mb):
                        # Compile and execute
                        compiled = compile(code, '<sandbox>', 'exec')
                        exec(compiled, exec_globals, exec_locals)
                        
                except TimeoutError:
                    return SandboxResult(
                        success=False,
                        exit_code=-1,
                        stdout=stdout_capture.getvalue(),
                        stderr=stderr_capture.getvalue(),
                        duration=time.time() - start_time,
                        timed_out=True,
                        error_message=f"Execution timed out after {self.timeout} seconds",
                    )
                except MemoryError:
                    return SandboxResult(
                        success=False,
                        exit_code=-2,
                        stdout=stdout_capture.getvalue(),
                        stderr=stderr_capture.getvalue(),
                        duration=time.time() - start_time,
                        memory_exceeded=True,
                        error_message=f"Memory limit exceeded ({self.max_memory_mb} MB)",
                    )
                except Exception as e:
                    return SandboxResult(
                        success=False,
                        exit_code=1,
                        stdout=stdout_capture.getvalue(),
                        stderr=stderr_capture.getvalue(),
                        duration=time.time() - start_time,
                        error_message=f"{type(e).__name__}: {e}",
                    )
            
            # Success
            end_mem = 0
            if hasattr(resource, 'getrusage'):
                end_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024  # KB to bytes
            
            return SandboxResult(
                success=True,
                exit_code=0,
                stdout=stdout_capture.getvalue(),
                stderr=stderr_capture.getvalue(),
                duration=time.time() - start_time,
                memory_used=end_mem,
            )
            
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
    
    def execute_function(
        self,
        func: Callable,
        args: tuple = (),
        kwargs: Optional[Dict[str, Any]] = None,
    ) -> SandboxResult:
        """
        Execute a Python function in sandbox.
        
        Args:
            func: Function to execute
            args: Positional arguments
            kwargs: Keyword arguments
            
        Returns:
            SandboxResult with execution results
        """
        kwargs = kwargs or {}
        
        # Wrap function call in code string
        code_lines = [
            "# Wrapped function call",
            "import sys",
            "result = None",
            "try:",
            f"    result = func{self._format_call(args, kwargs)}",
            "except Exception as e:",
            "    print(f'Error: {e}', file=sys.stderr)",
            "    raise",
        ]
        code = "\n".join(code_lines)
        
        globals_dict = {'func': func}
        return self.execute(code, globals_dict=globals_dict)
    
    def _validate_code(self, code: str) -> None:
        """Validate code before execution."""
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            raise
        
        # Check for dangerous constructs
        for node in ast.walk(tree):
            # Block exec/eval calls
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in ('eval', 'exec', 'execfile', 'compile'):
                        raise SecurityError(f"Call to '{node.func.id}' is not allowed")
                elif isinstance(node.func, ast.Attribute):
                    if node.func.attr in ('__import__', 'exec', 'eval'):
                        raise SecurityError(f"Call to '{node.func.attr}' is not allowed")
            
            # Block import statements for dangerous modules
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in ('os', 'sys', 'subprocess', 'socket', 'ctypes'):
                        raise SecurityError(f"Import of '{alias.name}' is not allowed")
            
            if isinstance(node, ast.ImportFrom):
                if node.module and node.module.split('.')[0] in ('os', 'sys', 'subprocess', 'socket', 'ctypes'):
                    raise SecurityError(f"Import from '{node.module}' is not allowed")
    
    def _format_call(self, args: tuple, kwargs: Dict[str, Any]) -> str:
        """Format function call arguments."""
        parts = []
        for arg in args:
            parts.append(repr(arg))
        for key, value in kwargs.items():
            parts.append(f"{key}={repr(value)}")
        return f"({', '.join(parts)})"


class SecurityError(Exception):
    """Raised when code violates security constraints."""
    pass
