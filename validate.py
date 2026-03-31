#!/usr/bin/env python3
"""
Pre-submission validation script for Cloud FinOps Agent.

Checks all hackathon requirements before submission.
"""

import os
import sys
import json
from pathlib import Path

def check_file_exists(filepath, description):
    """Check if a required file exists."""
    if Path(filepath).exists():
        print(f"✓ {description}: {filepath}")
        return True
    else:
        print(f"✗ MISSING {description}: {filepath}")
        return False

def check_env_vars():
    """Check required environment variables."""
    print("\n" + "="*60)
    print("Checking Environment Variables")
    print("="*60)
    
    required_vars = ["API_BASE_URL", "MODEL_NAME", "HF_TOKEN"]
    all_present = True
    
    for var in required_vars:
        value = os.getenv(var)
        if value:
            print(f"✓ {var} is set")
        else:
            print(f"✗ {var} is NOT set")
            all_present = False
    
    return all_present

def check_files():
    """Check required files exist."""
    print("\n" + "="*60)
    print("Checking Required Files")
    print("="*60)
    
    required_files = [
        ("inference.py", "Baseline inference script"),
        ("Dockerfile", "Docker configuration"),
        ("requirements.txt", "Python dependencies"),
        ("openenv.yaml", "OpenEnv specification"),
        ("README.md", "Documentation"),
        ("server.py", "Server entry point"),
        ("my_env/models.py", "Pydantic models"),
        ("my_env/client.py", "Environment client"),
        ("my_env/server/finops_environment.py", "Environment implementation"),
        ("my_env/server/app.py", "FastAPI application"),
        ("my_env/server/mock_cloud_state.json", "Initial state data"),
        ("my_env/server/optimal_cloud_state.json", "Grader answer key"),
    ]
    
    all_present = True
    for filepath, description in required_files:
        if not check_file_exists(filepath, description):
            all_present = False
    
    return all_present

def check_openenv_yaml():
    """Validate openenv.yaml structure."""
    print("\n" + "="*60)
    print("Validating openenv.yaml")
    print("="*60)
    
    try:
        import yaml
        with open("openenv.yaml", 'r') as f:
            config = yaml.safe_load(f)
        
        required_keys = ["spec_version", "name", "type", "runtime", "app", "port"]
        all_present = True
        
        for key in required_keys:
            if key in config:
                print(f"✓ {key}: {config[key]}")
            else:
                print(f"✗ Missing key: {key}")
                all_present = False
        
        if config.get("port") == 7860:
            print("✓ Port is 7860 (hackathon requirement)")
        else:
            print(f"✗ Port should be 7860, got {config.get('port')}")
            all_present = False
        
        return all_present
    except Exception as e:
        print(f"✗ Error reading openenv.yaml: {e}")
        return False

def check_models():
    """Check that models are properly defined."""
    print("\n" + "="*60)
    print("Checking Pydantic Models")
    print("="*60)
    
    try:
        from my_env.models import FinOpsAction, FinOpsObservation
        
        # Check FinOpsAction fields
        action_fields = FinOpsAction.model_fields.keys()
        required_action_fields = ["reasoning", "target_resource_id", "action_type", "new_instance_type"]
        
        for field in required_action_fields:
            if field in action_fields:
                print(f"✓ FinOpsAction.{field}")
            else:
                print(f"✗ Missing FinOpsAction.{field}")
                return False
        
        # Check FinOpsObservation fields
        obs_fields = FinOpsObservation.model_fields.keys()
        required_obs_fields = ["task_instruction", "cloud_state", "current_savings", "task_score"]
        
        for field in required_obs_fields:
            if field in obs_fields:
                print(f"✓ FinOpsObservation.{field}")
            else:
                print(f"✗ Missing FinOpsObservation.{field}")
                return False
        
        print("✓ All model fields present")
        return True
    except Exception as e:
        print(f"✗ Error importing models: {e}")
        return False

def check_environment():
    """Test environment can be instantiated."""
    print("\n" + "="*60)
    print("Testing Environment")
    print("="*60)
    
    try:
        from my_env.server.finops_environment import FinOpsEnvironment
        from my_env.models import FinOpsAction
        
        env = FinOpsEnvironment()
        print("✓ Environment instantiated")
        
        obs = env.reset()
        print(f"✓ reset() works - Task: {obs.task_instruction[:50]}...")
        
        action = FinOpsAction(
            reasoning="Test action",
            target_resource_id="eip-001",
            action_type="DELETE"
        )
        obs = env.step(action)
        print(f"✓ step() works - Reward: {obs.reward:.2f}, Score: {obs.task_score:.2%}")
        
        return True
    except Exception as e:
        print(f"✗ Environment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all validation checks."""
    print("="*60)
    print("Cloud FinOps Agent - Pre-Submission Validation")
    print("="*60)
    
    checks = [
        ("Environment Variables", check_env_vars),
        ("Required Files", check_files),
        ("OpenEnv YAML", check_openenv_yaml),
        ("Pydantic Models", check_models),
        ("Environment Logic", check_environment),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ {name} check crashed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(passed for _, passed in results)
    
    if all_passed:
        print("\n🎉 All validation checks passed!")
        print("\nNext steps:")
        print("1. Build Docker image: docker build -t finops-env .")
        print("2. Run container: docker run -p 7860:7860 finops-env")
        print("3. Test inference: python inference.py")
        print("4. Deploy to HF Spaces")
        return 0
    else:
        print("\n⚠️  Some validation checks failed. Please fix before submitting.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
