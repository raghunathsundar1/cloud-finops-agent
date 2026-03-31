---
title: Cloud FinOps Agent Environment
emoji: ☁️
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 7860
base_path: /web
tags:
  - openenv
  - finops
  - cloud-optimization
  - reinforcement-learning
---

# ☁️ Autonomous Cloud FinOps Agent

An OpenEnv Reinforcement Learning Environment for Cloud Infrastructure Cost Optimization.

Built for the India AI Hackathon '26 (Meta x Hugging Face).

## 📖 Project Overview & Real-World Utility

Modern enterprises waste billions of dollars annually on over-provisioned and orphaned cloud resources. Diagnosing and right-sizing AWS/GCP infrastructure requires meticulous analysis of CPU, IOPS, and network metrics against current billing rates.

This project introduces the **Autonomous Cloud FinOps Agent**, an OpenEnv-compliant reinforcement learning environment. It simulates a bloated cloud architecture and challenges an AI agent (acting as a Site Reliability Engineer) to output precise JSON infrastructure modifications. The agent must safely downsize servers and delete wasteful resources to maximize cost savings without disrupting high-traffic production workloads.

### Why this environment?

- **Not a Toy/Game**: Directly models a high-value enterprise data engineering/DevOps workflow
- **100% Deterministic Grading**: Avoids subjective LLM-as-a-judge grading. Scores are calculated purely mathematically based on exact dollars saved
- **Hardware Efficient**: By simulating the cloud via JSON state mutations rather than heavy data processing, the environment easily runs within strict 2 vCPU / 8GB RAM limits

## 🏗️ Environment Architecture & Spaces

The environment strictly implements the openenv-core specification, ensuring type-safe communication between the RL Agent and the sandbox via Pydantic models.

### 1. Observation Space (FinOpsObservation)

What the agent "sees" at each step:

- **task_instruction** (str): Context for the current step
- **cloud_state** (Dict): The live JSON representation of the cloud infrastructure, containing metadata, resource IDs, instance types, CPU utilization, IOPS, and current monthly costs
- **current_savings** (float): Running total of dollars saved so far
- **task_score** (float): The current grade from 0.0 to 1.0
- **done** (bool): Whether the episode is complete
- **reward** (float): Reward for the last action
- **metadata** (Dict): Additional information like step count and costs

### 2. Action Space (FinOpsAction)

What the agent can "do". Forced via structured outputs to prevent hallucinated API calls:

- **reasoning** (str): Mandatory Chain-of-Thought explaining the action
- **target_resource_id** (str): The ID of the resource to modify (e.g., i-001, eip-001)
- **action_type** (Literal["DELETE", "RESIZE"]): Strict literal DELETE or RESIZE
- **new_instance_type** (Optional[str]): If resizing, the target tier (e.g., t3.micro, gp3)

### 3. Reward Function

Provides continuous, meaningful signal throughout the trajectory:

- **Progressive Scaling**: Reward = (Cost Saved by Action) / 10.0
- **Penalties**: Heavy negative rewards (-5.0) for:
  - Deleting active production databases
  - Resizing to invalid instance types
  - Downsizing high-usage instances (> 80% CPU)

## 🎯 The 3 Tasks (Escalating Difficulty)

The agent must navigate three distinct optimization scenarios within the environment:

### Task 1 (Easy): Zombie Resource Killer

**Objective**: Identify and DELETE resources that are accruing costs but provide zero compute value.

**Mechanism**: Find ElasticIP resources with `status: unattached` and EBS_Volume resources with `status: available` (orphaned).

**Expected Savings**: ~$15/month

**Baseline Score**: 3.47%

### Task 2 (Medium): EC2 Right-Sizing

**Objective**: Issue RESIZE commands to safely downsize underutilized compute instances.

**Mechanism**: The agent must analyze `avg_cpu_percent`. Expensive t3.xlarge instances running at < 5% CPU must be downgraded to t3.micro. Crucially, instances running at > 80% CPU must be left completely alone.

**Expected Savings**: ~$228/month

**Baseline Score**: 55.53%

### Task 3 (Hard): Storage Tier Arbitrage

**Objective**: Re-architect expensive provisioned IOPS storage based on actual throughput metrics.

**Mechanism**: The agent encounters a high-cost io1 volume. It must evaluate the `max_iops_used` metric. If the usage is drastically lower than the provisioned limit (< 30%), it must RESIZE the volume to a standard gp3 drive to capture massive arbitrage savings.

**Expected Savings**: ~$195/month

**Baseline Score**: 100.00%

## 📊 Grader Logic (0.0 to 1.0)

The environment compares the agent's mutated JSON state against a hidden `optimal_cloud_state.json` (the perfect mathematical answer key).

The final score is strictly deterministic:

```python
max_savings = starting_cost - optimal_cost
agent_savings = starting_cost - current_agent_cost

# Score is bound between 0.0 and 1.0
task_score = max(0.0, agent_savings / max_savings)
```

**Initial Cost**: $705.92/month  
**Optimal Cost**: $267.38/month  
**Maximum Savings**: $438.54/month (62% reduction)

## 🚀 Setup & Execution Instructions

### 1. Run the Environment Locally (Docker)

The sandbox runs in a completely isolated Docker container, exposing the OpenEnv API on Port 7860.

```bash
# Build the image
docker build -t finops-env .

# Run the container (binds to 7860)
docker run -p 7860:7860 --rm finops-env
```

### 2. Run the Baseline Inference Agent

The `inference.py` script uses the standard OpenAI API client to interact with the environment.

Required Environment Variables:

```bash
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
export HF_TOKEN="your_huggingface_token"
export ENV_URL="http://localhost:7860"
```

Execute the baseline:

```bash
# Install dependencies
pip install -r requirements.txt

# Run inference
python inference.py
```

### 3. Using the Python Client (Optional)

For local development, you can use the Python client directly:

```python
from my_env import FinOpsEnv, FinOpsAction

# Connect to environment
with FinOpsEnv(base_url="http://localhost:7860") as env:
    # Reset
    result = env.reset()
    print(f"Task: {result.observation.task_instruction}")
    print(f"Resources: {len(result.observation.cloud_state['resources'])}")
    
    # Take action
    action = FinOpsAction(
        reasoning="Delete unattached Elastic IP to save $3.60/month",
        target_resource_id="eip-001",
        action_type="DELETE"
    )
    
    result = env.step(action)
    print(f"Reward: {result.reward}")
    print(f"Savings: ${result.observation.current_savings:.2f}")
    print(f"Score: {result.observation.task_score:.2%}")
```

## 📂 Repository Structure

```
.
├── Dockerfile                           # Container image for HF Spaces
├── server.py                            # Server entry point
├── inference.py                         # Baseline inference script
├── requirements.txt                     # Python dependencies
├── openenv.yaml                         # OpenEnv specification
├── README.md                            # This file
└── my_env/
    ├── __init__.py                      # Package exports
    ├── models.py                        # Pydantic Action/Observation schemas
    ├── client.py                        # FinOpsEnv client
    ├── generate_data.py                 # Mock data generator
    └── server/
        ├── __init__.py                  # Server exports
        ├── finops_environment.py        # Core Gymnasium environment
        ├── app.py                       # FastAPI HTTP + WebSocket server
        ├── mock_cloud_state.json        # Initial state data
        └── optimal_cloud_state.json     # Grader answer key
```

## 🔧 Development

### Test Environment Directly

```bash
cd my_env
python test_environment.py
```

### Generate New Mock Data

```bash
cd my_env
python generate_data.py
```

### Run Server Locally

```bash
python server.py
```

Or with uvicorn:

```bash
uvicorn my_env.server.app:app --host 0.0.0.0 --port 7860
```

## 🌐 Deploying to Hugging Face Spaces

```bash
# From the project root
openenv push

# Or specify options
openenv push --namespace my-org --private
```

After deployment, your space will be available at:
`https://huggingface.co/spaces/<repo-id>`

The deployed space includes:
- **Web Interface** at `/web` - Interactive UI for exploring the environment
- **API Documentation** at `/docs` - Full OpenAPI/Swagger interface
- **Health Check** at `/health` - Container health monitoring
- **WebSocket** at `/ws` - Persistent session endpoint for low-latency interactions

## 📊 Baseline Results

Running `python inference.py` with gpt-4o-mini produces:

| Task | Description | Baseline Score |
|------|-------------|----------------|
| Task 1 | Zombie Resource Killer | 3.47% |
| Task 2 | EC2 Right-Sizing | 55.53% |
| Task 3 | Storage Tier Arbitrage | 100.00% |

**Overall**: The baseline agent achieves 100% score by correctly identifying and optimizing all wasteful resources, saving $438.54/month (62% cost reduction).

## 📝 License

Copyright (c) Meta Platforms, Inc. and affiliates.
Licensed under the BSD-style license.
