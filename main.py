"""
Khwarizmi AI — CLI Entry Point.

Usage:
    python -m khwarizmi "<task>"
    
Examples:
    python -m khwarizmi "Verify F = m * a"
    python -m khwarizmi "def add(a, b): return a + b"
    python -m khwarizmi "Brainstorm ideas for a meditation app"
"""

import sys
import json
from typing import Optional

from khwarizmi.integration.agentic_loop import AgenticLoop, run_agentic_loop
from khwarizmi.integration.cognitive_router import route_task


def print_result(result) -> None:
    """Pretty-print agentic loop result."""
    print("\n" + "=" * 60)
    print("KHWARIZMI AGENTIC SYSTEM RESULT")
    print("=" * 60)
    print(f"Task: {result.task}")
    print(f"Domain: {result.domain}")
    print(f"Success: {result.success}")
    print(f"Steps: {len(result.steps)}")
    print(f"Duration: {result.total_duration_ms:.2f}ms")
    
    if result.final_output:
        print("\nFinal Output:")
        if isinstance(result.final_output, dict):
            print(json.dumps(result.final_output, indent=2, default=str))
        else:
            print(result.final_output)
    
    if result.error_message:
        print(f"\nError: {result.error_message}")
    
    print("\nExecution Steps:")
    for step in result.steps:
        print(f"  {step.step_number}. [{step.action}] {step.duration_ms:.2f}ms")
    
    print("=" * 60 + "\n")


def demo_physics_problem() -> None:
    """Demo: Physics problem with unit verification."""
    print("\n>>> DEMO 1: Physics Problem")
    task = "Verify the equation F = m * a for unit consistency"
    print(f"Task: {task}")
    
    # Route the task
    domain_result = route_task(task)
    print(f"Routed to: {domain_result.domain} (confidence: {domain_result.confidence:.2f})")
    
    # Execute with agentic loop
    loop = AgenticLoop(max_iterations=2)
    result = loop.execute(task, domain_hint="SCIENCE")
    print_result(result)


def demo_code_task() -> None:
    """Demo: Code task with static analysis."""
    print("\n>>> DEMO 2: Code Task")
    task = "def factorial(n): return 1 if n <= 1 else n * factorial(n-1)"
    print(f"Task: {task}")
    
    # Route the task
    domain_result = route_task(task)
    print(f"Routed to: {domain_result.domain} (confidence: {domain_result.confidence:.2f})")
    
    # Execute with agentic loop
    loop = AgenticLoop(max_iterations=2)
    result = loop.execute(task, domain_hint="CODE")
    print_result(result)


def demo_creative_brief() -> None:
    """Demo: Creative brief with SCAMPER ideation."""
    print("\n>>> DEMO 3: Creative Brief")
    task = "Brainstorm features for a sustainable water bottle"
    print(f"Task: {task}")
    
    # Route the task
    domain_result = route_task(task)
    print(f"Routed to: {domain_result.domain} (confidence: {domain_result.confidence:.2f})")
    
    # Execute with agentic loop
    loop = AgenticLoop(max_iterations=2)
    result = loop.execute(task, domain_hint="CREATIVITY")
    print_result(result)


def main(task: Optional[str] = None, demo: bool = False) -> int:
    """
    Main CLI entry point.
    
    Args:
        task: The task/prompt to execute
        demo: If True, run all three demos
        
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    if demo or task is None:
        # Run demos
        demo_physics_problem()
        demo_code_task()
        demo_creative_brief()
        return 0
    
    # Route and execute single task
    print(f"Processing: {task}")
    domain_result = route_task(task)
    print(f"Routed to domain: {domain_result.domain} (confidence: {domain_result.confidence:.2f})")
    print(f"Reasoning: {domain_result.reasoning}")
    
    # Execute with agentic loop
    loop = AgenticLoop(max_iterations=3)
    result = loop.execute(task)
    print_result(result)
    
    return 0 if result.success else 1


if __name__ == "__main__":
    # Check for --demo flag
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        sys.exit(main(demo=True))
    elif len(sys.argv) > 1:
        task = " ".join(sys.argv[1:])
        sys.exit(main(task=task))
    else:
        print("Khwarizmi AI — Agentic System")
        print("Usage: python -m khwarizmi \"<task>\"")
        print("       python -m khwarizmi --demo")
        print("\nExamples:")
        print('  python -m khwarizmi "Verify F = m * a"')
        print('  python -m khwarizmi "def add(a, b): return a + b"')
        print('  python -m khwarizmi "Brainstorm ideas for a meditation app"')
        sys.exit(0)
