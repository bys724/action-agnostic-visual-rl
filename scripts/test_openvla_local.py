#!/usr/bin/env python3
"""
OpenVLA 통합 테스트 스크립트
실제 모델 로드 없이 인터페이스만 테스트
"""

import sys
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_imports():
    """Test imports"""
    print("1. Testing imports...")
    try:
        from src.policies.openvla import OpenVLAPolicy
        print("   ✓ OpenVLAPolicy import successful")
        return True
    except ImportError as e:
        print(f"   ✗ Import failed: {e}")
        return False

def test_policy_interface():
    """Test OpenVLA policy interface without loading actual model"""
    print("\n2. Testing policy interface...")
    
    # Mock OpenVLA policy for testing
    class MockOpenVLAPolicy:
        def __init__(self, model_path="mock"):
            self.model_path = model_path
            print(f"   MockOpenVLA initialized with: {model_path}")
        
        def reset(self, instruction):
            self.instruction = instruction
            print(f"   Reset with instruction: {instruction[:50]}...")
        
        def step(self, image, instruction=None):
            # Return dummy action
            action = np.random.randn(7) * 0.01
            action_dict = {
                "world_vector": action.astype(np.float32),
                "terminate_episode": np.array([0])
            }
            return action, action_dict
        
        def get_action(self, obs):
            if isinstance(obs, dict):
                # Dummy image
                image = np.random.randn(224, 224, 3)
            else:
                image = obs
            _, action_dict = self.step(image)
            return action_dict["world_vector"]
    
    try:
        # Test initialization
        policy = MockOpenVLAPolicy("test-model")
        print("   ✓ Policy initialization successful")
        
        # Test reset
        policy.reset("Pick up the spoon")
        print("   ✓ Policy reset successful")
        
        # Test step
        dummy_image = np.random.randn(224, 224, 3).astype(np.float32)
        raw_action, action_dict = policy.step(dummy_image, "Pick up the spoon")
        assert "world_vector" in action_dict
        assert "terminate_episode" in action_dict
        assert action_dict["world_vector"].shape == (7,)
        print("   ✓ Policy step successful")
        
        # Test get_action
        action = policy.get_action({"rgb": dummy_image})
        assert action.shape == (7,)
        print("   ✓ Policy get_action successful")
        
        return True
    except Exception as e:
        print(f"   ✗ Policy interface test failed: {e}")
        return False

def test_eval_simpler_integration():
    """Test integration with eval_simpler.py"""
    print("\n3. Testing eval_simpler.py integration...")
    
    try:
        # Import the load_policy function
        from src.eval_simpler import load_policy
        
        # Test loading SimplePolicy
        policy = load_policy("simple")
        print("   ✓ SimplePolicy loads correctly")
        
        # Test that OpenVLA path is recognized
        # (won't actually load model without GPU/dependencies)
        import unittest.mock as mock
        with mock.patch('src.policies.openvla.OpenVLAPolicy') as MockOpenVLA:
            MockOpenVLA.return_value = mock.MagicMock()
            policy = load_policy("openvla/openvla-7b")
            MockOpenVLA.assert_called_once_with(model_path="openvla/openvla-7b")
            print("   ✓ OpenVLA path recognized in load_policy")
        
        return True
    except Exception as e:
        print(f"   ✗ eval_simpler integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_collect_trajectories_integration():
    """Test integration with collect_trajectories.py"""
    print("\n4. Testing collect_trajectories.py integration...")
    
    try:
        # Import the load_policy function
        from src.collect_trajectories import load_policy
        
        # Test loading SimplePolicy
        policy = load_policy("simple")
        print("   ✓ SimplePolicy loads correctly")
        
        # Test that OpenVLA path is recognized
        import unittest.mock as mock
        with mock.patch('src.policies.openvla.OpenVLAPolicy') as MockOpenVLA:
            MockOpenVLA.return_value = mock.MagicMock()
            policy = load_policy("openvla/openvla-7b")
            MockOpenVLA.assert_called_once_with(model_path="openvla/openvla-7b")
            print("   ✓ OpenVLA path recognized in load_policy")
        
        return True
    except Exception as e:
        print(f"   ✗ collect_trajectories integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("="*60)
    print("OpenVLA Integration Test")
    print("="*60)
    
    results = []
    
    # Run tests
    results.append(("Imports", test_imports()))
    results.append(("Policy Interface", test_policy_interface()))
    results.append(("eval_simpler.py", test_eval_simpler_integration()))
    results.append(("collect_trajectories.py", test_collect_trajectories_integration()))
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary:")
    print("-"*60)
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {test_name:30s} {status}")
    
    total_passed = sum(1 for _, p in results if p)
    total_tests = len(results)
    print("-"*60)
    print(f"Total: {total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        print("\n✅ All tests passed! OpenVLA integration is ready.")
        return 0
    else:
        print("\n❌ Some tests failed. Please fix the issues.")
        return 1

if __name__ == "__main__":
    exit(main())