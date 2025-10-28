import torch
import time
from ACNet_pytorch import ACNet, test_gradients, test_gradient_flow_all, benchmark_gpu, test_gpu_memory

def run_all_tests(grid_size=10, a_size=5):

    print("\n=== Starting All Tests ===\n")
    
    print("Creating model...")
    model = ACNet("test", a_size=a_size, grid_size=grid_size)
    
    print("\n=== Basic Model Test ===")
    test_basic_forward(model)
    
    print("\n=== Gradient Flow Test ===")
    gradient_results = test_gradient_flow_all(model)
    
    if torch.cuda.is_available():
        print("\n=== GPU Tests ===")
        
        print("\nRunning GPU benchmarks...")
        benchmark_results = benchmark_gpu(model)
        
        print("\nTesting GPU memory usage...")
        memory_results = test_gpu_memory(model)
        
        results = {
            "gradient_results": gradient_results,
            "gpu_results": {
                "benchmark": benchmark_results,
                "memory": memory_results
            }
        }
    else:
        print("\nGPU not available, skipping GPU tests")
        results = {
            "gradient_results": gradient_results,
            "gpu_results": None
        }
    
    print("\n=== All Tests Completed ===\n")
    return results

def test_basic_forward(model):
    """Базовий тест forward pass"""
    print("Testing forward pass...")
    
    
    batch_size = 32
    grid_size = 10
    
    obs = torch.randn(batch_size, 4, grid_size, grid_size)
    goals = torch.randn(batch_size, 3)
    hidden_state = model.init_hidden(batch_size)
    
    policy, value, new_hidden, blocking, on_goal, valids = model(obs, goals, hidden_state)
    
    assert policy.shape == (batch_size, model.a_size), f"Policy shape wrong: {policy.shape}"
    assert value.shape == (batch_size, 1), f"Value shape wrong: {value.shape}"
    assert blocking.shape == (batch_size, 1), f"Blocking shape wrong: {blocking.shape}"
    assert on_goal.shape == (batch_size, 1), f"On_goal shape wrong: {on_goal.shape}"
    assert valids.shape == (batch_size, model.a_size), f"Valids shape wrong: {valids.shape}"
    
    print("Basic forward pass test passed!")
    
    return {
        "policy_shape": policy.shape,
        "value_shape": value.shape,
        "blocking_shape": blocking.shape,
        "on_goal_shape": on_goal.shape,
        "valids_shape": valids.shape
    }

if __name__ == "__main__":
    print("Starting tests...")
    results = run_all_tests()
    
    print("\nGradient Test Results:")

    if "cpu" in results["gradient_results"]:
        print("\nCPU Gradients:")
        for layer, grad_info in results["gradient_results"]["cpu"]["gradients"].items():
            if grad_info["grad_exists"]:
                grad_std_str = f"{grad_info['grad_std']:.6f}" if grad_info['grad_std'] is not None else "N/A (single value)"
                print(f"{layer}: mean={grad_info['grad_mean']:.6f}, std={grad_std_str}")

    if results["gpu_results"]:
        print("\nGPU Benchmark Results:")
        for batch_size, metrics in results["gpu_results"]["benchmark"].items():
            print(f"Batch size {batch_size}:")
            print(f"  Average time: {metrics['avg_time_ms']:.2f} ms")

            print(f"  Throughput: {metrics['throughput']:.2f} samples/sec")
