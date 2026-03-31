# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Cloud FinOps Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import FinOpsAction, FinOpsObservation


class FinOpsEnv(
    EnvClient[FinOpsAction, FinOpsObservation, State]
):
    """
    Client for the Cloud FinOps Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> with FinOpsEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.task_instruction)
        ...
        ...     result = client.step(FinOpsAction(
        ...         reasoning="Delete unattached IP",
        ...         target_resource_id="eip-001",
        ...         action_type="DELETE"
        ...     ))
        ...     print(result.observation.current_savings)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = FinOpsEnv.from_docker_image("finops-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(FinOpsAction(
        ...         reasoning="Downsize underutilized instance",
        ...         target_resource_id="i-12345",
        ...         action_type="RESIZE",
        ...         new_instance_type="t3.micro"
        ...     ))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: FinOpsAction) -> Dict:
        """
        Convert FinOpsAction to JSON payload for step message.

        Args:
            action: FinOpsAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        payload = {
            "reasoning": action.reasoning,
            "target_resource_id": action.target_resource_id,
            "action_type": action.action_type,
        }
        if action.new_instance_type:
            payload["new_instance_type"] = action.new_instance_type
        return payload

    def _parse_result(self, payload: Dict) -> StepResult[FinOpsObservation]:
        """
        Parse server response into StepResult[FinOpsObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with FinOpsObservation
        """
        obs_data = payload.get("observation", {})
        observation = FinOpsObservation(
            task_instruction=obs_data.get("task_instruction", ""),
            cloud_state=obs_data.get("cloud_state", {}),
            current_savings=obs_data.get("current_savings", 0.0),
            task_score=obs_data.get("task_score", 0.0),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
