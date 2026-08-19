"""
Khwarizmi Control Flow Graph — CFG Construction from Python AST.

This module builds a Control Flow Graph (CFG) from Python AST, providing:
- Basic block construction for if/for/while/try/return statements
- Edge creation for control flow paths
- Cyclomatic complexity computation
- Adjacency dict export and text visualization

All analysis is purely static using only the standard library ast module.
"""

import ast
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple, Any


@dataclass
class CFGNode:
    """Represents a basic block in the control flow graph."""
    node_id: int
    node_type: str  # "entry", "exit", "statement", "if", "for", "while", "try", "except", "return"
    start_line: int = 0
    end_line: int = 0
    source_code: str = ""
    successors: List[int] = field(default_factory=list)
    predecessors: List[int] = field(default_factory=list)

    def __str__(self) -> str:
        return f"CFGNode({self.node_id}, type={self.node_type}, lines={self.start_line}-{self.end_line})"


@dataclass
class CFGEdge:
    """Represents an edge between two CFG nodes."""
    source: int
    target: int
    edge_type: str = "flow"  # "flow", "true", "false", "break", "continue", "exception", "return"

    def __str__(self) -> str:
        return f"CFGEdge({self.source} -> {self.target}, type={self.edge_type})"


@dataclass
class CFGReport:
    """Structured report of CFG analysis results."""
    source_code: str
    parse_successful: bool = True
    syntax_error: Optional[str] = None
    
    nodes: Dict[int, CFGNode] = field(default_factory=dict)
    edges: List[CFGEdge] = field(default_factory=list)
    
    entry_node: int = 0
    exit_node: int = 0
    
    cyclomatic_complexity: int = 1
    num_nodes: int = 0
    num_edges: int = 0
    num_decision_points: int = 0

    @property
    def adjacency_dict(self) -> Dict[int, List[int]]:
        """Return adjacency dictionary representation of the CFG."""
        adj: Dict[int, List[int]] = {}
        for node_id, node in self.nodes.items():
            adj[node_id] = list(node.successors)
        return adj

    def to_dict(self) -> Dict[str, Any]:
        """Serialize report to dictionary."""
        return {
            "parse_successful": self.parse_successful,
            "syntax_error": self.syntax_error,
            "cyclomatic_complexity": self.cyclomatic_complexity,
            "num_nodes": self.num_nodes,
            "num_edges": self.num_edges,
            "num_decision_points": self.num_decision_points,
            "entry_node": self.entry_node,
            "exit_node": self.exit_node,
            "nodes": [
                {
                    "id": n.node_id,
                    "type": n.node_type,
                    "start_line": n.start_line,
                    "end_line": n.end_line,
                    "successors": n.successors,
                    "predecessors": n.predecessors,
                }
                for n in self.nodes.values()
            ],
            "edges": [
                {"source": e.source, "target": e.target, "type": e.edge_type}
                for e in self.edges
            ],
        }

    def visualize(self) -> str:
        """Generate simple text visualization of the CFG."""
        lines = []
        lines.append("=" * 60)
        lines.append("CONTROL FLOW GRAPH")
        lines.append("=" * 60)
        lines.append(f"Cyclomatic Complexity: {self.cyclomatic_complexity}")
        lines.append(f"Nodes: {self.num_nodes}, Edges: {self.num_edges}")
        lines.append("")
        
        # Show nodes
        lines.append("NODES:")
        lines.append("-" * 40)
        for node_id in sorted(self.nodes.keys()):
            node = self.nodes[node_id]
            marker = ""
            if node_id == self.entry_node:
                marker = " [ENTRY]"
            elif node_id == self.exit_node:
                marker = " [EXIT]"
            lines.append(f"  [{node_id}] {node.node_type} (lines {node.start_line}-{node.end_line}){marker}")
        
        lines.append("")
        lines.append("EDGES:")
        lines.append("-" * 40)
        for edge in self.edges:
            lines.append(f"  [{edge.source}] --({edge.edge_type})--> [{edge.target}]")
        
        lines.append("")
        lines.append("ADJACENCY LIST:")
        lines.append("-" * 40)
        for node_id in sorted(self.adjacency_dict.keys()):
            succs = self.adjacency_dict[node_id]
            lines.append(f"  [{node_id}] -> {succs}")
        
        lines.append("=" * 60)
        return "\n".join(lines)


class ControlFlowGraph(ast.NodeVisitor):
    """
    Build a Control Flow Graph from Python AST.

    Creates basic blocks and edges representing all possible execution paths
    through the code, including:
    - Sequential flow
    - Conditional branches (if/else)
    - Loop constructs (for/while with break/continue)
    - Exception handling (try/except/finally)
    - Early returns

    Usage:
        cfg_builder = ControlFlowGraph()
        report = cfg_builder.build(source_code)
    """

    def __init__(self):
        self._reset_state()

    def _reset_state(self):
        """Reset internal state for new analysis."""
        self.nodes: Dict[int, CFGNode] = {}
        self.edges: List[CFGEdge] = []
        self.next_node_id = 0
        self.entry_node = 0
        self.exit_node = 0
        self.current_successors: List[int] = []
        self.break_targets: List[int] = []
        self.continue_targets: List[int] = []
        self.exception_handlers: List[Tuple[int, int]] = []  # (handler_start, handler_end)
        self.decision_count = 0

    def build(self, source_code: str) -> CFGReport:
        """
        Build CFG from Python source code.

        Args:
            source_code: Python source code as string

        Returns:
            CFGReport with graph structure and metrics
        """
        self._reset_state()

        try:
            tree = ast.parse(source_code)
        except SyntaxError as e:
            return CFGReport(
                source_code=source_code,
                parse_successful=False,
                syntax_error=f"Line {e.lineno}: {e.msg}" if e.lineno else str(e),
            )

        # Create entry node
        self.entry_node = self._create_node("entry", 1, 1, "")
        self.current_successors = [self.entry_node]

        # Visit AST
        self.visit(tree)

        # Connect remaining successors to exit
        self.exit_node = self._create_node("exit", len(source_code.splitlines()), 
                                           len(source_code.splitlines()), "")
        for succ in self.current_successors:
            if succ != self.exit_node:
                self._add_edge(succ, self.exit_node, "flow")

        # Calculate cyclomatic complexity: M = E - N + 2P
        # For single function/module: M = edges - nodes + 2
        # Alternative: M = decision_points + 1
        num_edges = len(self.edges)
        num_nodes = len(self.nodes)
        
        # Use decision point formula (more intuitive)
        cyclomatic = self.decision_count + 1

        return CFGReport(
            source_code=source_code,
            parse_successful=True,
            nodes=self.nodes,
            edges=self.edges,
            entry_node=self.entry_node,
            exit_node=self.exit_node,
            cyclomatic_complexity=cyclomatic,
            num_nodes=num_nodes,
            num_edges=num_edges,
            num_decision_points=self.decision_count,
        )

    def _create_node(
        self, 
        node_type: str, 
        start_line: int, 
        end_line: int, 
        source: str = ""
    ) -> int:
        """Create a new CFG node and return its ID."""
        node_id = self.next_node_id
        self.next_node_id += 1
        
        node = CFGNode(
            node_id=node_id,
            node_type=node_type,
            start_line=start_line,
            end_line=end_line,
            source_code=source.strip()[:100],  # Truncate long lines
        )
        self.nodes[node_id] = node
        return node_id

    def _add_edge(self, source: int, target: int, edge_type: str = "flow"):
        """Add an edge between two nodes."""
        edge = CFGEdge(source=source, target=target, edge_type=edge_type)
        self.edges.append(edge)
        
        # Update node successor/predecessor lists
        if source in self.nodes:
            if target not in self.nodes[source].successors:
                self.nodes[source].successors.append(target)
        if target in self.nodes:
            if source not in self.nodes[target].predecessors:
                self.nodes[target].predecessors.append(source)

    def visit_Module(self, node: ast.Module):
        """Visit module - process all top-level statements."""
        for stmt in node.body:
            self.visit(stmt)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Visit function definition."""
        func_node = self._create_node(
            "statement", 
            node.lineno, 
            node.end_lineno or node.lineno,
            ast.unparse(node)[:50] if hasattr(ast, 'unparse') else f"def {node.name}"
        )
        
        # Connect from current flow
        new_successors = []
        for succ in self.current_successors:
            self._add_edge(succ, func_node, "flow")
            new_successors.append(func_node)
        
        self.current_successors = new_successors
        # Don't traverse into function body for top-level CFG

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        """Visit async function definition."""
        self.visit_FunctionDef(node)

    def visit_ClassDef(self, node: ast.ClassDef):
        """Visit class definition."""
        class_node = self._create_node(
            "statement",
            node.lineno,
            node.end_lineno or node.lineno,
            f"class {node.name}"
        )
        
        new_successors = []
        for succ in self.current_successors:
            self._add_edge(succ, class_node, "flow")
            new_successors.append(class_node)
        
        self.current_successors = new_successors

    def visit_If(self, node: ast.If):
        """Visit if statement - creates branch in CFG."""
        self.decision_count += 1
        
        if_node = self._create_node(
            "if",
            node.lineno,
            node.lineno,
            ast.unparse(node.test) if hasattr(ast, 'unparse') else "if condition"
        )
        
        # Connect to if node
        for succ in self.current_successors:
            self._add_edge(succ, if_node, "flow")
        
        # Process then branch (true path)
        self.current_successors = [if_node]
        then_successors = []
        for stmt in node.body:
            self.visit(stmt)
            then_successors = list(self.current_successors)
        
        # Process else branch (false path)
        false_successors = []
        if node.orelse:
            self.current_successors = [if_node]
            for stmt in node.orelse:
                self.visit(stmt)
            false_successors = list(self.current_successors)
        else:
            # No else - false path continues after if
            false_successors = [if_node]
        
        # Add true/false edges
        self._add_edge(if_node, then_successors[0] if then_successors else if_node, "true")
        if node.orelse:
            self._add_edge(if_node, false_successors[0] if false_successors else if_node, "false")
        
        # Merge successors
        self.current_successors = list(set(then_successors + false_successors))

    def visit_For(self, node: ast.For):
        """Visit for loop."""
        self.decision_count += 1
        
        loop_header = self._create_node(
            "for",
            node.lineno,
            node.lineno,
            ast.unparse(node.iter) if hasattr(ast, 'unparse') else "for loop"
        )
        
        # Connect to loop header
        for succ in self.current_successors:
            self._add_edge(succ, loop_header, "flow")
        
        # Setup break/continue targets
        self.break_targets.append(loop_header)  # break exits loop
        self.continue_targets.append(loop_header)  # continue goes to next iteration
        
        # Process loop body
        self.current_successors = [loop_header]
        for stmt in node.body:
            self.visit(stmt)
        
        # Connect body end back to loop header (continue edge)
        body_end = list(self.current_successors)
        for end in body_end:
            if end != loop_header:
                self._add_edge(end, loop_header, "continue")
        
        # Process else clause
        if node.orelse:
            for stmt in node.orelse:
                self.visit(stmt)
        
        # Break exits to after loop
        loop_exit = self._create_node(
            "statement",
            node.end_lineno or node.lineno,
            node.end_lineno or node.lineno,
            "loop exit"
        )
        
        # Remove continue target
        self.continue_targets.pop()
        break_target = self.break_targets.pop()
        
        # All current successors go to loop exit
        new_successors = []
        for succ in self.current_successors:
            if succ != loop_header:
                self._add_edge(succ, loop_exit, "flow")
                new_successors.append(loop_exit)
        
        if not new_successors:
            new_successors = [loop_exit]
        
        self.current_successors = new_successors

    def visit_While(self, node: ast.While):
        """Visit while loop."""
        self.decision_count += 1
        
        loop_header = self._create_node(
            "while",
            node.lineno,
            node.lineno,
            ast.unparse(node.test) if hasattr(ast, 'unparse') else "while condition"
        )
        
        # Connect to loop header
        for succ in self.current_successors:
            self._add_edge(succ, loop_header, "flow")
        
        # Setup break/continue targets
        self.break_targets.append(loop_header)
        self.continue_targets.append(loop_header)
        
        # Process loop body
        self.current_successors = [loop_header]
        for stmt in node.body:
            self.visit(stmt)
        
        # Connect body end back to loop header
        body_end = list(self.current_successors)
        for end in body_end:
            if end != loop_header:
                self._add_edge(end, loop_header, "continue")
        
        # Process else clause
        if node.orelse:
            for stmt in node.orelse:
                self.visit(stmt)
        
        # Loop exit
        loop_exit = self._create_node(
            "statement",
            node.end_lineno or node.lineno,
            node.end_lineno or node.lineno,
            "loop exit"
        )
        
        self.continue_targets.pop()
        self.break_targets.pop()
        
        new_successors = []
        for succ in self.current_successors:
            if succ != loop_header:
                self._add_edge(succ, loop_exit, "flow")
                new_successors.append(loop_exit)
        
        if not new_successors:
            new_successors = [loop_exit]
        
        self.current_successors = new_successors

    def visit_Return(self, node: ast.Return):
        """Visit return statement."""
        return_node = self._create_node(
            "return",
            node.lineno,
            node.lineno,
            ast.unparse(node) if hasattr(ast, 'unparse') else "return"
        )
        
        for succ in self.current_successors:
            self._add_edge(succ, return_node, "return")
        
        # Return doesn't continue to normal flow
        self.current_successors = [return_node]

    def visit_Break(self, node: ast.Break):
        """Visit break statement."""
        break_node = self._create_node(
            "statement",
            node.lineno,
            node.lineno,
            "break"
        )
        
        for succ in self.current_successors:
            self._add_edge(succ, break_node, "flow")
        
        # Break jumps to loop exit
        if self.break_targets:
            self._add_edge(break_node, self.break_targets[-1], "break")
        
        self.current_successors = [break_node]

    def visit_Continue(self, node: ast.Continue):
        """Visit continue statement."""
        cont_node = self._create_node(
            "statement",
            node.lineno,
            node.lineno,
            "continue"
        )
        
        for succ in self.current_successors:
            self._add_edge(succ, cont_node, "flow")
        
        # Continue jumps to loop header
        if self.continue_targets:
            self._add_edge(cont_node, self.continue_targets[-1], "continue")
        
        self.current_successors = [cont_node]

    def visit_Try(self, node: ast.Try):
        """Visit try statement."""
        self.decision_count += 1
        
        try_node = self._create_node(
            "try",
            node.lineno,
            node.lineno,
            "try"
        )
        
        for succ in self.current_successors:
            self._add_edge(succ, try_node, "flow")
        
        # Process try body
        self.current_successors = [try_node]
        try_successors = []
        for stmt in node.body:
            self.visit(stmt)
        try_successors = list(self.current_successors)
        
        # Process exception handlers
        handler_successors = []
        for handler in node.handlers:
            self.current_successors = [try_node]  # Exception can come from try
            self.visit(handler)
            handler_successors.extend(self.current_successors)
        
        # Process else clause
        if node.orelse:
            self.current_successors = try_successors
            for stmt in node.orelse:
                self.visit(stmt)
            try_successors = list(self.current_successors)
        
        # Process finally clause
        finally_successors = []
        if node.finalbody:
            self.current_successors = try_successors + handler_successors
            for stmt in node.finalbody:
                self.visit(stmt)
            finally_successors = list(self.current_successors)
        
        # Merge all paths
        self.current_successors = list(set(try_successors + handler_successors + finally_successors))

    def visit_ExceptHandler(self, node: ast.ExceptHandler):
        """Visit exception handler."""
        handler_node = self._create_node(
            "except",
            node.lineno,
            node.lineno,
            f"except {ast.unparse(node.type) if hasattr(ast, 'unparse') and node.type else '*'}"
        )
        
        for succ in self.current_successors:
            self._add_edge(succ, handler_node, "exception")
        
        self.current_successors = [handler_node]
        
        for stmt in node.body:
            self.visit(stmt)

    def visit_Raise(self, node: ast.Raise):
        """Visit raise statement."""
        raise_node = self._create_node(
            "statement",
            node.lineno,
            node.lineno,
            "raise"
        )
        
        for succ in self.current_successors:
            self._add_edge(succ, raise_node, "flow")
        
        # If there's an exception handler, connect to it
        if self.exception_handlers:
            for handler_start, _ in self.exception_handlers:
                self._add_edge(raise_node, handler_start, "exception")
        
        self.current_successors = [raise_node]

    def visit_Assign(self, node: ast.Assign):
        """Visit assignment statement."""
        assign_node = self._create_node(
            "statement",
            node.lineno,
            node.lineno,
            ast.unparse(node) if hasattr(ast, 'unparse') else "assignment"
        )
        
        new_successors = []
        for succ in self.current_successors:
            self._add_edge(succ, assign_node, "flow")
            new_successors.append(assign_node)
        
        self.current_successors = new_successors

    def visit_Expr(self, node: ast.Expr):
        """Visit expression statement."""
        expr_node = self._create_node(
            "statement",
            node.lineno,
            node.lineno,
            ast.unparse(node) if hasattr(ast, 'unparse') else "expression"
        )
        
        new_successors = []
        for succ in self.current_successors:
            self._add_edge(succ, expr_node, "flow")
            new_successors.append(expr_node)
        
        self.current_successors = new_successors

    def visit_Pass(self, node: ast.Pass):
        """Visit pass statement."""
        pass_node = self._create_node(
            "statement",
            node.lineno,
            node.lineno,
            "pass"
        )
        
        new_successors = []
        for succ in self.current_successors:
            self._add_edge(succ, pass_node, "flow")
            new_successors.append(pass_node)
        
        self.current_successors = new_successors

    def generic_visit(self, node):
        """Default visitor for unhandled node types."""
        # Create a generic statement node
        if hasattr(node, 'lineno'):
            stmt_node = self._create_node(
                "statement",
                node.lineno,
                getattr(node, 'end_lineno', node.lineno),
                ast.unparse(node) if hasattr(ast, 'unparse') else type(node).__name__
            )
            
            new_successors = []
            for succ in self.current_successors:
                self._add_edge(succ, stmt_node, "flow")
                new_successors.append(stmt_node)
            
            self.current_successors = new_successors
        
        # Continue visiting children
        for child in ast.iter_child_nodes(node):
            self.visit(child)
