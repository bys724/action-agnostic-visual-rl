#!/usr/bin/env python3
"""
OpenVLA 통합 구문 검사 - ManiSkill 없이 테스트
"""

import ast
import sys
from pathlib import Path

def check_python_syntax(file_path):
    """Check if Python file has valid syntax"""
    try:
        with open(file_path, 'r') as f:
            ast.parse(f.read())
        return True, None
    except SyntaxError as e:
        return False, str(e)

def check_openvla_integration_in_file(file_path):
    """Check if OpenVLA is properly integrated in the file"""
    with open(file_path, 'r') as f:
        content = f.read()
    
    checks = []
    
    # Check for OpenVLA import/usage
    if 'openvla' in content.lower():
        checks.append(("OpenVLA mentioned", True))
        
        # Check for proper import statement
        if 'from src.policies.openvla import OpenVLAPolicy' in content:
            checks.append(("OpenVLA import statement", True))
        elif 'from src.policies.openvla' in content:
            checks.append(("OpenVLA import statement", True))
        else:
            checks.append(("OpenVLA import statement", False))
            
        # Check for model path handling
        if 'openvla/openvla-7b' in content or '"openvla"' in content:
            checks.append(("OpenVLA model path handling", True))
        else:
            checks.append(("OpenVLA model path handling", False))
    else:
        checks.append(("OpenVLA mentioned", False))
    
    return checks

def main():
    print("="*60)
    print("OpenVLA Integration Syntax Check")
    print("="*60)
    
    # Files to check
    files_to_check = [
        "src/policies/openvla/__init__.py",
        "src/policies/openvla/openvla_model.py",
        "src/eval_simpler.py",
        "src/collect_trajectories.py"
    ]
    
    all_passed = True
    
    for file_path in files_to_check:
        file_path = Path(file_path)
        print(f"\n📄 Checking {file_path}...")
        
        if not file_path.exists():
            print(f"   ✗ File not found!")
            all_passed = False
            continue
        
        # Check syntax
        valid, error = check_python_syntax(file_path)
        if valid:
            print(f"   ✓ Valid Python syntax")
        else:
            print(f"   ✗ Syntax error: {error}")
            all_passed = False
            continue
        
        # Check OpenVLA integration for relevant files
        if "eval_simpler" in str(file_path) or "collect_trajectories" in str(file_path):
            checks = check_openvla_integration_in_file(file_path)
            for check_name, passed in checks:
                if passed:
                    print(f"   ✓ {check_name}")
                else:
                    print(f"   ✗ {check_name}")
                    all_passed = False
    
    # Check if OpenVLA model file has required methods
    print(f"\n📄 Checking OpenVLA model methods...")
    openvla_model = Path("src/policies/openvla/openvla_model.py")
    if openvla_model.exists():
        with open(openvla_model, 'r') as f:
            content = f.read()
        
        required_methods = ["__init__", "reset", "step", "get_action", "_parse_action"]
        for method in required_methods:
            if f"def {method}" in content:
                print(f"   ✓ Method '{method}' found")
            else:
                print(f"   ✗ Method '{method}' not found")
                all_passed = False
        
        # Check for required classes
        required_classes = ["OpenVLAPolicy", "OpenVLAInference"]
        for cls in required_classes:
            if f"class {cls}" in content:
                print(f"   ✓ Class '{cls}' found")
            else:
                print(f"   ✗ Class '{cls}' not found")
                all_passed = False
    
    # Summary
    print("\n" + "="*60)
    if all_passed:
        print("✅ All syntax and integration checks passed!")
        print("\nNext steps:")
        print("1. Build Docker image: docker build -t simpler-env:latest .")
        print("2. Run container: docker compose up -d eval")
        print("3. Test in container: docker exec -it simpler-dev bash")
        print("4. Run: python src/eval_simpler.py --model simple --n-episodes 2")
        return 0
    else:
        print("❌ Some checks failed. Please fix the issues.")
        return 1

if __name__ == "__main__":
    exit(main())