#!/usr/bin/env python3
"""Baseline inference script for Cloud FinOps Agent."""

import os
import json # Ensure json is imported at the top!
from openai import OpenAI
from my_env.models import FinOpsAction
from my_env.client import FinOpsEnv

# Strict Hackathon Variables
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
ENV_URL = os.getenv("ENV_URL", "http://localhost:7860")

if not API_KEY:
    raise ValueError("HF_TOKEN or OPENAI_API_KEY environment variable must be set")

def main():
    # Initialize the LLM
    llm = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
    
    print(f"Connecting to FinOps environment at {ENV_URL}...")
    
    # Using EnvClient with sync wrapper for synchronous usage
    with FinOpsEnv(base_url=ENV_URL).sync() as env:
        # Reset the environment to get the starting state
        result = env.reset()
        obs = result.observation
        
        print(f"\n{'='*60}")
        print(f"Task: {obs.task_instruction}")
        print(f"Resources Found: {len(obs.cloud_state.get('resources', []))}")
        print(f"{'='*60}\n")
        
        step = 0
        done = False
        
        while not done and step < 10:
            step += 1
            
            # Added CRITICAL JSON RULES to force exact action types and instance tiers
            system_prompt = """You are a Cloud FinOps expert optimizing AWS infrastructure costs.

Rules:
- DELETE unattached ElasticIPs (status: unattached)
- DELETE orphaned EBS volumes (status: available)
- RESIZE underutilized EC2 instances (< 5% CPU) to t3.micro
- NEVER touch high-usage instances (> 80% CPU)
- RESIZE io1 volumes with low IOPS usage to gp3

CRITICAL JSON RULES:
- "action_type" MUST be exactly "DELETE" or "RESIZE".
- If "action_type" is "RESIZE", you MUST provide "new_instance_type" (e.g., "t3.micro" or "gp3")."""
            
            user_prompt = f"""Current cloud state: {obs.cloud_state}

Current savings: ${obs.current_savings:.2f}
Task score: {obs.task_score:.2%}

What action should I take next?"""
            
            # Call LLM - try structured outputs first, fallback to JSON mode
            try:
                response = llm.beta.chat.completions.parse(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt + "\n\nRespond strictly with valid JSON: {\"reasoning\": \"...\", \"target_resource_id\": \"...\", \"action_type\": \"DELETE or RESIZE\", \"new_instance_type\": \"...or null\"}"}
                    ],
                    response_format=FinOpsAction,
                    temperature=0.1 # Lowered temperature for more deterministic JSON
                )
                action = response.choices[0].message.parsed
                
            except Exception as e:
                # Fallback for models without structured output support (like Groq/Llama)
                try:
                    response = llm.chat.completions.create(
                        model=MODEL_NAME,
                        messages=[
                            {"role": "system", "content": system_prompt + "\n\nYou MUST respond with valid JSON only."},
                            {"role": "user", "content": user_prompt + "\n\nRespond strictly with valid JSON: {\"reasoning\": \"...\", \"target_resource_id\": \"...\", \"action_type\": \"DELETE or RESIZE\", \"new_instance_type\": \"...or null\"}"}
                        ],
                        temperature=0.1
                    )
                    action_dict = json.loads(response.choices[0].message.content)
                    action = FinOpsAction(**action_dict)
                    
                except Exception as inner_e:
                    # THE SAFETY NET: Catches bad JSON or validation errors and skips the step instead of crashing
                    print(f"⚠️ LLM hallucinated bad JSON at Step {step}. Skipping. Error: {inner_e}")
                    continue 
            
            print(f"Step {step}: {action.action_type} {action.target_resource_id}")
            if action.new_instance_type:
                print(f"  Target Tier: {action.new_instance_type}")
            print(f"  Reasoning: {action.reasoning}")
            
            # 4. Step the environment using your new client
            result = env.step(action)
            obs = result.observation
            done = result.done
            
            print(f"  Reward: {result.reward:.2f}")
            print(f"  Score: {obs.task_score:.2%}\n")
        
        print(f"\nEpisode Complete! Final Score: {obs.task_score:.2%}")

if __name__ == "__main__":
    main()