"""
Khwarizmi Agentic Loop — Test-Time Compute with Tool Integration.

This module implements the core agentic execution loop:
    1. THINK: TestTimeComputeScaling (AdaptiveComputeBlock) generates reasoning
    2. PICK: CognitiveRouter selects domain and appropriate tool
    3. EXECUTE: ToolRegistry invokes the selected tool
    4. VERIFY: ProcessRewardModel / UnitVerdict / SandboxResult validates output
    5. REFINE or HALT: Iterate if verification fails, halt if successful

Constraints:
    - Offline-first operation (no network calls)
    - <4GB RAM footprint
    - Deterministic execution where possible
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Callable
import time

from .tool_registry import ToolRegistry, ToolResult, get_registry, invoke_tool
from .cognitive_router import CognitiveRouter, DomainResult, get_router, route_task


@dataclass
class AgentStep:
    """Single step in the agentic loop."""
    step_number: int
    action: str  # "think", "pick_tool", "execute", "verify", "refine", "halt"
    input_data: Any = None
    output_data: Any = None
    verification_result: Optional[Dict[str, Any]] = None
    duration_ms: float = 0.0
    error_message: str = ""


@dataclass
class AgenticLoopResult:
    """Final result from the agentic loop."""
    success: bool
    task: str
    domain: str
    steps: List[AgentStep] = field(default_factory=list)
    final_output: Any = None
    total_duration_ms: float = 0.0
    max_iterations_reached: bool = False
    error_message: str = ""
    
    @property
    def is_valid(self) -> bool:
        return self.success and not self.error_message
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "task": self.task,
            "domain": self.domain,
            "step_count": len(self.steps),
            "final_output": self.final_output,
            "total_duration_ms": self.total_duration_ms,
            "max_iterations_reached": self.max_iterations_reached,
            "error_message": self.error_message,
        }


class ProcessRewardModel:
    """
    Lightweight process reward model for verifying tool outputs.
    
    Provides deterministic heuristics for assessing output quality
    without requiring neural inference.
    """
    
    @staticmethod
    def verify_sandbox_result(result: Dict[str, Any]) -> Tuple[bool, str]:
        """Verify ExecutionSandbox output."""
        if not result.get("success"):
            if result.get("timed_out"):
                return False, "Execution timed out"
            return False, "Sandbox execution failed"
        if result.get("timed_out"):
            return False, "Execution timed out"
        if result.get("memory_exceeded"):
            return False, "Memory limit exceeded"
        if result.get("exit_code", 0) != 0:
            return False, f"Non-zero exit code: {result.get('exit_code')}"
        return True, "Sandbox execution successful"
    
    @staticmethod
    def verify_unit_verdict(result: Dict[str, Any]) -> Tuple[bool, str]:
        """Verify UnitConsistencyVerifier output."""
        if not result.get("success"):
            return False, "Unit verification failed"
        lhs = result.get("lhs_dimensions", {})
        rhs = result.get("rhs_dimensions", {})
        if lhs != rhs:
            return False, f"Dimension mismatch: {lhs} vs {rhs}"
        return True, "Units are consistent"
    
    @staticmethod
    def verify_aesthetic_score(result: Dict[str, Any]) -> Tuple[bool, str]:
        """Verify AestheticScorer output."""
        if not result.get("success"):
            return False, "Aesthetic scoring failed"
        overall = result.get("overall_score", 0)
        if not isinstance(overall, (int, float)):
            return False, "Invalid score type"
        if overall < 0 or overall > 100:
            return False, f"Score out of range: {overall}"
        return True, f"Aesthetic score: {overall}"
    
    @staticmethod
    def verify_scamper_candidates(result: Dict[str, Any]) -> Tuple[bool, str]:
        """Verify ScamperEngine output."""
        if not result.get("success"):
            return False, "SCAMPER generation blocked or failed"
        candidates = result.get("candidates", [])
        if not candidates:
            return False, "No candidates generated"
        if len(candidates) < 5:
            return False, f"Too few candidates: {len(candidates)}"
        return True, f"Generated {len(candidates)} SCAMPER candidates"
    
    @staticmethod
    def verify_code_analysis(result: Dict[str, Any], tool_name: str) -> Tuple[bool, str]:
        """Verify code analysis tools (DataFlow, CFG, TypeInference)."""
        if not result.get("success"):
            return False, f"{tool_name} analysis failed"
        
        if tool_name == "DataFlowAnalyzer":
            issues = result.get("total_unused", 0) + result.get("total_undefined", 0) + result.get("total_dead", 0)
            if issues > 10:
                return False, f"Too many data flow issues: {issues}"
        elif tool_name == "ControlFlowGraph":
            complexity = result.get("cyclomatic_complexity", 0)
            if complexity > 50:
                return False, f"Cyclomatic complexity too high: {complexity}"
        elif tool_name == "TypeInference":
            total = result.get("total_inferences", 0)
            if total == 0:
                return False, "No type inferences made"
        
        return True, f"{tool_name} analysis complete"
    
    @classmethod
    def verify(cls, tool_name: str, result: Dict[str, Any]) -> Tuple[bool, str]:
        """Dispatch to appropriate verifier based on tool name."""
        verifiers = {
            "ExecutionSandbox": cls.verify_sandbox_result,
            "UnitConsistencyVerifier": cls.verify_unit_verdict,
            "AestheticScorer": cls.verify_aesthetic_score,
            "ScamperEngine": cls.verify_scamper_candidates,
            "DataFlowAnalyzer": lambda r: cls.verify_code_analysis(r, "DataFlowAnalyzer"),
            "ControlFlowGraph": lambda r: cls.verify_code_analysis(r, "ControlFlowGraph"),
            "TypeInference": lambda r: cls.verify_code_analysis(r, "TypeInference"),
        }
        
        verifier = verifiers.get(tool_name)
        if verifier is None:
            return True, f"No verifier for {tool_name}; accepting by default"
        
        return verifier(result)


class TestTimeComputeScaling:
    """
    Simulates test-time compute scaling for the THINK phase.
    
    In a full implementation, this would use AdaptiveComputeBlock
    to perform recurrent reasoning cycles. For now, provides
    lightweight task decomposition and planning.
    """
    
    def __init__(self, max_cycles: int = 5):
        self.max_cycles = max_cycles
    
    def think(self, task: str, domain: str) -> Dict[str, Any]:
        """
        Perform thinking/planning phase.
        
        Returns structured plan for tool selection and execution.
        """
        # Simple deterministic planning based on domain
        plans = {
            "CODE": {
                "suggested_tools": ["DataFlowAnalyzer", "ControlFlowGraph", "TypeInference", "ExecutionSandbox"],
                "reasoning": "Code tasks benefit from static analysis followed by execution",
                "priority": "analysis_first",
            },
            "SCIENCE": {
                "suggested_tools": ["UnitConsistencyVerifier"],
                "reasoning": "Science/physics tasks require unit consistency verification",
                "priority": "verify_equation",
            },
            "ART": {
                "suggested_tools": ["AestheticScorer"],
                "reasoning": "Art tasks require aesthetic evaluation",
                "priority": "score_composition",
            },
            "CREATIVITY": {
                "suggested_tools": ["ScamperEngine"],
                "reasoning": "Creative tasks benefit from SCAMPER ideation",
                "priority": "generate_alternatives",
            },
            "GENERAL": {
                "suggested_tools": [],
                "reasoning": "General tasks may not require specialized tools",
                "priority": "direct_response",
            },
        }
        
        return plans.get(domain, plans["GENERAL"])


class AgenticLoop:
    """
    Main agentic execution loop integrating all components.
    
    Loop phases:
        1. THINK: Analyze task and generate plan
        2. PICK: Select domain and tool
        3. EXECUTE: Run tool with validated inputs
        4. VERIFY: Check output quality
        5. REFINE/HALT: Iterate or terminate
    
    Usage:
        loop = AgenticLoop()
        result = loop.execute("Write a function to calculate factorial")
    """
    
    def __init__(
        self,
        tool_registry: Optional[ToolRegistry] = None,
        cognitive_router: Optional[CognitiveRouter] = None,
        max_iterations: int = 5,
        max_think_cycles: int = 3,
    ):
        self.tool_registry = tool_registry or get_registry()
        self.cognitive_router = cognitive_router or get_router()
        self.max_iterations = max_iterations
        self.max_think_cycles = max_think_cycles
        self.thinker = TestTimeComputeScaling(max_cycles=max_think_cycles)
        self.reward_model = ProcessRewardModel()
    
    def execute(self, task: str, domain_hint: Optional[str] = None) -> AgenticLoopResult:
        """
        Execute the full agentic loop for a given task.
        
        Args:
            task: The task description or prompt
            domain_hint: Optional hint about the expected domain
            
        Returns:
            AgenticLoopResult with execution details
        """
        start_time = time.time()
        steps: List[AgentStep] = []
        
        # Step 1: Route task to domain
        step_num = 1
        route_start = time.time()
        
        if domain_hint:
            domain_result = DomainResult(
                domain=domain_hint,
                confidence=0.8,
                reasoning=f"Domain hint provided: {domain_hint}",
            )
        else:
            domain_result = self.cognitive_router.route(task)
        
        steps.append(AgentStep(
            step_number=step_num,
            action="route",
            input_data=task,
            output_data=domain_result.to_dict(),
            duration_ms=(time.time() - route_start) * 1000,
        ))
        
        domain = domain_result.domain
        
        # Handle GENERAL domain - may not need tools
        if domain == "GENERAL":
            return AgenticLoopResult(
                success=True,
                task=task,
                domain=domain,
                steps=steps,
                final_output={"message": "General task - no specialized tools required"},
                total_duration_ms=(time.time() - start_time) * 1000,
            )
        
        # Step 2: THINK - Generate plan
        step_num += 1
        think_start = time.time()
        plan = self.thinker.think(task, domain)
        
        steps.append(AgentStep(
            step_number=step_num,
            action="think",
            input_data={"task": task, "domain": domain},
            output_data=plan,
            duration_ms=(time.time() - think_start) * 1000,
        ))
        
        # Step 3+: Execute loop - pick tool, execute, verify
        iteration = 0
        suggested_tools = plan.get("suggested_tools", [])
        
        while iteration < self.max_iterations:
            iteration += 1
            
            # Pick next tool
            step_num += 1
            pick_start = time.time()
            
            if iteration <= len(suggested_tools):
                tool_name = suggested_tools[iteration - 1]
            else:
                # Fall back to first tool if we've exhausted suggestions
                tool_name = suggested_tools[0] if suggested_tools else None
            
            if tool_name is None:
                # No tools available for this domain
                return AgenticLoopResult(
                    success=True,
                    task=task,
                    domain=domain,
                    steps=steps,
                    final_output={"message": f"No tools available for {domain} domain"},
                    total_duration_ms=(time.time() - start_time) * 1000,
                )
            
            steps.append(AgentStep(
                step_number=step_num,
                action="pick_tool",
                input_data={"iteration": iteration, "available_tools": suggested_tools},
                output_data={"selected_tool": tool_name},
                duration_ms=(time.time() - pick_start) * 1000,
            ))
            
            # Execute tool
            step_num += 1
            exec_start = time.time()
            
            # Prepare arguments based on tool type
            tool_args = self._prepare_tool_args(task, tool_name, domain)
            result = self.tool_registry.invoke(tool_name, **tool_args)
            
            steps.append(AgentStep(
                step_number=step_num,
                action="execute",
                input_data={"tool": tool_name, "args": tool_args},
                output_data=result.to_dict(),
                duration_ms=(time.time() - exec_start) * 1000,
                error_message=result.error_message,
            ))
            
            if not result.success:
                # Tool execution failed - try refinement or next tool
                if iteration < self.max_iterations:
                    continue  # Try next iteration/tool
                else:
                    return AgenticLoopResult(
                        success=False,
                        task=task,
                        domain=domain,
                        steps=steps,
                        final_output=None,
                        total_duration_ms=(time.time() - start_time) * 1000,
                        max_iterations_reached=True,
                        error_message=f"All tool attempts failed. Last error: {result.error_message}",
                    )
            
            # Verify output
            step_num += 1
            verify_start = time.time()
            
            verified, verify_msg = self.reward_model.verify(tool_name, result.output or {})
            
            verification_data = {
                "verified": verified,
                "message": verify_msg,
                "tool": tool_name,
            }
            
            steps.append(AgentStep(
                step_number=step_num,
                action="verify",
                input_data={"tool_result": result.to_dict()},
                output_data=verification_data,
                duration_ms=(time.time() - verify_start) * 1000,
                verification_result=verification_data,
            ))
            
            if verified:
                # Success! Return result
                return AgenticLoopResult(
                    success=True,
                    task=task,
                    domain=domain,
                    steps=steps,
                    final_output=result.output,
                    total_duration_ms=(time.time() - start_time) * 1000,
                )
            else:
                # Verification failed - try refinement
                step_num += 1
                refine_start = time.time()
                
                steps.append(AgentStep(
                    step_number=step_num,
                    action="refine",
                    input_data={"verification_failure": verify_msg},
                    output_data={"strategy": "try_next_tool" if iteration < self.max_iterations else "give_up"},
                    duration_ms=(time.time() - refine_start) * 1000,
                ))
        
        # Max iterations reached without success
        return AgenticLoopResult(
            success=False,
            task=task,
            domain=domain,
            steps=steps,
            final_output=None,
            total_duration_ms=(time.time() - start_time) * 1000,
            max_iterations_reached=True,
            error_message="Max iterations reached without verified success",
        )
    
    def _prepare_tool_args(self, task: str, tool_name: str, domain: str) -> Dict[str, Any]:
        """Prepare appropriate arguments for a tool based on task and domain."""
        # Default: pass task as the main argument
        args_map = {
            "ExecutionSandbox": {"code": task},
            "UnitConsistencyVerifier": {"equation": task},
            "AestheticScorer": {"description": self._parse_art_brief(task)},
            "ScamperEngine": {"brief": task},
            "DataFlowAnalyzer": {"source_code": task},
            "ControlFlowGraph": {"source_code": task},
            "TypeInference": {"source_code": task},
        }
        return args_map.get(tool_name, {"task": task})
    
    def _parse_art_brief(self, task: str) -> Dict[str, Any]:
        """Extract art brief parameters from task description."""
        # Simple heuristic extraction - in production would use NLP
        return {
            "focal_point": (0.5, 0.5),
            "symmetry": 0.5,
            "balance": 0.5,
            "negative_space": 0.3,
            "harmony": "analogous",
            "contrast": 0.5,
            "temperature": "balanced",
            "saturation": 0.55,
        }


# Convenience functions
def run_agentic_loop(task: str, domain_hint: Optional[str] = None) -> AgenticLoopResult:
    """Run a single agentic loop execution."""
    loop = AgenticLoop()
    return loop.execute(task, domain_hint)


def create_loop(
    max_iterations: int = 5,
    max_think_cycles: int = 3,
) -> AgenticLoop:
    """Create a configured agentic loop instance."""
    return AgenticLoop(
        max_iterations=max_iterations,
        max_think_cycles=max_think_cycles,
    )
