# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Cloud FinOps Environment.

The FinOps environment simulates cloud infrastructure cost optimization.
"""

from typing import Dict, Literal, Optional
from pydantic import BaseModel, Field


class FinOpsAction(BaseModel):
    """Action for the Cloud FinOps environment - infrastructure modifications."""

    model_config = {"extra": "forbid"}
    
    reasoning: str = Field(..., description="Chain-of-thought explaining the action")
    target_resource_id: str = Field(..., description="ID of the resource to modify (e.g., i-12345)")
    action_type: Literal["DELETE", "RESIZE"] = Field(..., description="Type of action: DELETE or RESIZE")
    new_instance_type: Optional[str] = Field(None, description="Target tier for RESIZE actions (e.g., t3.micro)")


class FinOpsObservation(BaseModel):
    """Observation from the Cloud FinOps environment - current cloud state and metrics."""

    model_config = {"extra": "forbid"}
    
    task_instruction: str = Field(default="", description="Context for the current step")
    cloud_state: Dict = Field(default_factory=dict, description="Live JSON representation of cloud infrastructure")
    current_savings: float = Field(default=0.0, description="Running total of dollars saved")
    task_score: float = Field(default=0.0, description="Current grade from 0.0 to 1.0")
    done: bool = Field(default=False, description="Whether the episode is complete")
    reward: Optional[float] = Field(default=None, description="Reward for the last action")
    metadata: Dict = Field(default_factory=dict, description="Additional metadata")
