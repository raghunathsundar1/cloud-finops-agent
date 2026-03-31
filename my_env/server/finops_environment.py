# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Cloud FinOps Environment Implementation.

Simulates cloud infrastructure cost optimization challenges.
"""

import json
import copy
from pathlib import Path
from uuid import uuid4
from typing import Dict, List, Optional

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import FinOpsAction, FinOpsObservation
except ImportError:
    from models import FinOpsAction, FinOpsObservation


# Valid instance type tiers for resizing
VALID_INSTANCE_TYPES = ["t3.micro", "t3.small", "t3.medium", "t3.large", "t3.xlarge"]
VALID_VOLUME_TYPES = ["gp3", "io1"]


class FinOpsEnvironment(Environment):
    """
    Cloud FinOps optimization environment.
    
    Challenges an AI agent to optimize cloud infrastructure costs by:
    - Deleting zombie resources (unattached IPs, orphaned volumes)
    - Right-sizing underutilized EC2 instances
    - Optimizing storage tier configurations
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        """Initialize the FinOps environment."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._current_task = 1
        self._cloud_state = {}
        self._initial_cost = 0.0
        self._optimal_cost = 0.0
        self._current_savings = 0.0
        self._load_initial_state()

    def _load_initial_state(self):
        """Load the initial cloud state from JSON file."""
        data_path = Path(__file__).parent / "mock_cloud_state.json"
        optimal_path = Path(__file__).parent / "optimal_cloud_state.json"
        
        if data_path.exists():
            with open(data_path, 'r') as f:
                self._cloud_state = json.load(f)
        else:
            # Fallback to minimal state if file doesn't exist
            self._cloud_state = self._generate_minimal_state()
        
        if optimal_path.exists():
            with open(optimal_path, 'r') as f:
                optimal_state = json.load(f)
                self._optimal_cost = self._calculate_total_cost(optimal_state)
        
        self._initial_cost = self._calculate_total_cost(self._cloud_state)

    def _generate_minimal_state(self) -> Dict:
        """Generate a minimal cloud state for testing."""
        return {
            "resources": [
                {
                    "resource_id": "eip-001",
                    "resource_type": "ElasticIP",
                    "status": "unattached",
                    "monthly_cost": 3.60
                }
            ]
        }

    def _calculate_total_cost(self, state: Dict) -> float:
        """Calculate total monthly cost from cloud state."""
        total = 0.0
        for resource in state.get("resources", []):
            total += resource.get("monthly_cost", 0.0)
        return total

    def _get_task_instruction(self) -> str:
        """Get instruction for current task."""
        instructions = {
            1: "Task 1: Zombie Resource Killer - Delete unattached ElasticIPs and orphaned EBS volumes.",
            2: "Task 2: EC2 Right-Sizing - Downsize underutilized instances (< 5% CPU) to t3.micro. Leave high-usage (> 80% CPU) instances alone.",
            3: "Task 3: Storage Tier Arbitrage - Optimize expensive io1 volumes with low IOPS usage to gp3."
        }
        return instructions.get(self._current_task, "Optimize cloud infrastructure costs.")

    def reset(self) -> FinOpsObservation:
        """Reset the environment to initial state."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._current_task = 1
        self._load_initial_state()
        self._current_savings = 0.0

        return FinOpsObservation(
            task_instruction=self._get_task_instruction(),
            cloud_state=copy.deepcopy(self._cloud_state),
            current_savings=self._current_savings,
            task_score=0.0,
            done=False,
            reward=0.0
        )

    def step(self, action: FinOpsAction) -> FinOpsObservation:  # type: ignore[override]
        """Execute an infrastructure modification action."""
        self._state.step_count += 1
        
        reward = 0.0
        done = False
        
        # Find the target resource
        resource = self._find_resource(action.target_resource_id)
        
        if not resource:
            reward = -1.0  # Penalty for invalid resource ID
        elif action.action_type == "DELETE":
            reward = self._handle_delete(resource)
        elif action.action_type == "RESIZE":
            reward = self._handle_resize(resource, action.new_instance_type)
        
        # Recalculate savings and score
        current_cost = self._calculate_total_cost(self._cloud_state)
        self._current_savings = self._initial_cost - current_cost
        
        max_savings = self._initial_cost - self._optimal_cost
        task_score = max(0.0, min(1.0, self._current_savings / max_savings if max_savings > 0 else 0.0))
        
        # Check if task is complete (score >= 0.95)
        if task_score >= 0.95:
            done = True
        
        return FinOpsObservation(
            task_instruction=self._get_task_instruction(),
            cloud_state=copy.deepcopy(self._cloud_state),
            current_savings=self._current_savings,
            task_score=task_score,
            done=done,
            reward=reward,
            metadata={
                "step": self._state.step_count,
                "action_reasoning": action.reasoning,
                "initial_cost": self._initial_cost,
                "current_cost": current_cost
            }
        )

    def _find_resource(self, resource_id: str) -> Optional[Dict]:
        """Find a resource by ID."""
        for resource in self._cloud_state.get("resources", []):
            if resource.get("resource_id") == resource_id:
                return resource
        return None

    def _handle_delete(self, resource: Dict) -> float:
        """Handle DELETE action."""
        resource_type = resource.get("resource_type")
        status = resource.get("status")
        
        # Check for dangerous deletions
        if resource_type == "RDS_Instance" and status == "active":
            return -5.0  # Heavy penalty for deleting production database
        
        # Valid deletions: unattached IPs, orphaned volumes
        if (resource_type == "ElasticIP" and status == "unattached") or \
           (resource_type == "EBS_Volume" and status == "available"):
            cost_saved = resource.get("monthly_cost", 0.0)
            self._cloud_state["resources"].remove(resource)
            return cost_saved / 10.0  # Positive reward proportional to savings
        
        return -1.0  # Penalty for invalid deletion

    def _handle_resize(self, resource: Dict, new_type: Optional[str]) -> float:
        """Handle RESIZE action."""
        resource_type = resource.get("resource_type")
        
        if resource_type == "EC2_Instance":
            return self._resize_ec2(resource, new_type)
        elif resource_type == "EBS_Volume":
            return self._resize_volume(resource, new_type)
        
        return -1.0

    def _resize_ec2(self, resource: Dict, new_type: Optional[str]) -> float:
        """Resize EC2 instance."""
        if not new_type or new_type not in VALID_INSTANCE_TYPES:
            return -5.0  # Invalid instance type
        
        current_type = resource.get("instance_type")
        if current_type == new_type:
            return -1.0  # No change
        
        cpu_usage = resource.get("avg_cpu_percent", 0)
        
        # Don't downsize high-usage instances
        if cpu_usage > 80 and VALID_INSTANCE_TYPES.index(new_type) < VALID_INSTANCE_TYPES.index(current_type):
            return -5.0
        
        # Calculate cost difference based on instance type pricing
        # t3.micro: $7.59, t3.small: $15.18, t3.medium: $30.44, t3.large: $60.88, t3.xlarge: $121.76
        instance_costs = {
            "t3.micro": 7.59,
            "t3.small": 15.18,
            "t3.medium": 30.44,
            "t3.large": 60.88,
            "t3.xlarge": 121.76
        }
        
        old_cost = instance_costs.get(current_type, resource.get("monthly_cost", 0.0))
        new_cost = instance_costs.get(new_type, 0.0)
        
        resource["instance_type"] = new_type
        resource["monthly_cost"] = new_cost
        
        savings = old_cost - new_cost
        return max(0.0, savings / 10.0)  # Positive reward proportional to savings

    def _resize_volume(self, resource: Dict, new_type: Optional[str]) -> float:
        """Resize EBS volume."""
        if not new_type or new_type not in VALID_VOLUME_TYPES:
            return -5.0
        
        current_type = resource.get("volume_type")
        if current_type == new_type:
            return -1.0  # No change
        
        # io1 to gp3 is valid if IOPS usage is low
        if current_type == "io1" and new_type == "gp3":
            max_iops_used = resource.get("max_iops_used", 0)
            provisioned_iops = resource.get("provisioned_iops", 0)
            
            if max_iops_used < provisioned_iops * 0.3:  # Using < 30% of provisioned
                old_cost = resource.get("monthly_cost", 0.0)
                new_cost = old_cost * 0.4  # gp3 is ~60% cheaper
                
                resource["volume_type"] = new_type
                resource["monthly_cost"] = new_cost
                
                savings = old_cost - new_cost
                return savings / 10.0
        
        return -1.0

    @property
    def state(self) -> State:
        """Get the current environment state."""
        return self._state
