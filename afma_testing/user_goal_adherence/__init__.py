"""
User Goal Adherence Testing Module

This module provides functionality to test whether users adhere to their stated goals
during conversations with AI agents. It analyzes conversation traces to ensure that:

1. Users do not ask for anything outside their stated goal
2. Users do not accept additional help from agents that goes beyond their goal scope

The module includes:
- UserGoalAdherenceAssessor: Main assessor for evaluating goal adherence
- UserGoalAdherenceTester: Complete testing pipeline
- Configuration and result structures
"""

from .user_goal_adherence_assessor import UserGoalAdherenceAssessor, UserGoalAdherenceResult
from .user_goal_adherence_tester import UserGoalAdherenceTester, UserGoalAdherenceTestResult, UserGoalAdherenceTestSummary

__all__ = [
    "UserGoalAdherenceAssessor",
    "UserGoalAdherenceResult", 
    "UserGoalAdherenceTester",
    "UserGoalAdherenceTestResult",
    "UserGoalAdherenceTestSummary"
] 