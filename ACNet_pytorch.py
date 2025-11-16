import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import psutil

GRAD_CLIP = 1000.0
KEEP_PROB1 = 0.5
KEEP_PROB2 = 0.7
RNN_SIZE = 512
GOAL_REPR_SIZE = 12

def normalized_columns_initializer(std=1.0):
    def initializer(tensor):
        out = torch.randn_like(tensor)
        out *= std / torch.sqrt(torch.sum(out.pow(2), dim=0, keepdim=True) + 1e-8)
        return out
    return initializer

class ACNet(nn.Module):
    def __init__(self, scope, a_size, grid_size, training=True):
        super(ACNet, self).__init__()
        self.scope = scope
        self.a_size = a_size
        self.training = training
        
        self.w_init = torch.nn.init.kaiming_normal_
        
        self.conv1 = nn.Conv2d(4, RNN_SIZE//4, kernel_size=3, stride=1, padding='same')
        self.conv1a = nn.Conv2d(RNN_SIZE//4, RNN_SIZE//4, kernel_size=3, stride=1, padding='same')
        self.conv1b = nn.Conv2d(RNN_SIZE//4, RNN_SIZE//4, kernel_size=3, stride=1, padding='same')
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        self.conv2 = nn.Conv2d(RNN_SIZE//4, RNN_SIZE//2, kernel_size=3, stride=1, padding='same')
        self.conv2a = nn.Conv2d(RNN_SIZE//2, RNN_SIZE//2, kernel_size=3, stride=1, padding='same')
        self.conv2b = nn.Conv2d(RNN_SIZE//2, RNN_SIZE//2, kernel_size=3, stride=1, padding='same')
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        self.conv3 = nn.Conv2d(RNN_SIZE//2, RNN_SIZE-GOAL_REPR_SIZE, kernel_size=2, stride=1)
        
        self.goal_layer = nn.Linear(3, GOAL_REPR_SIZE)
        
        self.fc1 = nn.Linear(RNN_SIZE, RNN_SIZE)
        self.fc2 = nn.Linear(RNN_SIZE, RNN_SIZE)
        
        self.lstm = nn.LSTMCell(RNN_SIZE, RNN_SIZE)
        
        self.policy_layer = nn.Linear(RNN_SIZE, a_size)
        self.value_layer = nn.Linear(RNN_SIZE, 1)
        self.blocking_layer = nn.Linear(RNN_SIZE, 1)
        self.on_goal_layer = nn.Linear(RNN_SIZE, 1)
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in [self.conv1, self.conv1a, self.conv1b, self.conv2, self.conv2a, self.conv2b, self.conv3]:
            self.w_init(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        
        for m in [self.fc1, self.fc2]:
            self.w_init(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
                
        policy_init = normalized_columns_initializer(1.0/float(self.a_size))
        policy_init(self.policy_layer.weight)
        if self.policy_layer.bias is not None:
            nn.init.constant_(self.policy_layer.bias, 0.0)
            
        value_init = normalized_columns_initializer(1.0)
        value_init(self.value_layer.weight)
        if self.value_layer.bias is not None:
            nn.init.constant_(self.value_layer.bias, 0.0)
            
        blocking_init = normalized_columns_initializer(1.0)
        blocking_init(self.blocking_layer.weight)
        if self.blocking_layer.bias is not None:
            nn.init.constant_(self.blocking_layer.bias, 0.0)
            
        goal_init = normalized_columns_initializer(1.0)
        goal_init(self.on_goal_layer.weight)
        if self.on_goal_layer.bias is not None:
            nn.init.constant_(self.on_goal_layer.bias, 0.0)
            
        self.w_init(self.goal_layer.weight)
        if self.goal_layer.bias is not None:
            nn.init.constant_(self.goal_layer.bias, 0.0)

    def init_hidden(self, batch_size=1):
        return (torch.zeros(batch_size, RNN_SIZE),
                torch.zeros(batch_size, RNN_SIZE))

    def forward(self, inputs, goal_pos, hidden_state):
        x = inputs
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv1a(x))
        x = F.relu(self.conv1b(x))
        x = self.pool1(x)
        
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv2a(x))
        x = F.relu(self.conv2b(x))
        x = self.pool2(x)
        
        x = self.conv3(x)
        
        flat = F.relu(torch.flatten(x, start_dim=1))
        goal_repr = self.goal_layer(goal_pos)
        
        hidden_input = torch.cat([flat, goal_repr], dim=1)
        h1 = F.relu(self.fc1(hidden_input))
        d1 = F.dropout(h1, p=1-KEEP_PROB1, training=self.training)
        h2 = self.fc2(d1)
        d2 = F.dropout(h2, p=1-KEEP_PROB2, training=self.training)
        h3 = F.relu(d2 + hidden_input)
        
        hx, cx = self.lstm(h3, hidden_state)
        
        policy_logits = self.policy_layer(hx)
        policy = F.softmax(policy_logits, dim=-1)
        policy_sig = torch.sigmoid(policy_logits)
        value = self.value_layer(hx)
        blocking = torch.sigmoid(self.blocking_layer(hx))
        on_goal = torch.sigmoid(self.on_goal_layer(hx))
        
        return policy, value, (hx, cx), blocking, on_goal, policy_sig

    def save(self, path):
        torch.save({
            'model_state_dict': self.state_dict(),
        }, path)
    
    def load(self, path):
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])
    
    def to_device(self, device):
        self.to(device)
        return self

    def train_step_gpu(self, batch, optimizer, device):
        inputs = batch['obs'].to(device)
        goals = batch['goals'].to(device)
        hidden_state = (batch['hx'].to(device), batch['cx'].to(device))
        
        policy, value, new_hidden, blocking, on_goal, valids = self(inputs, goals, hidden_state)
        
        loss = self.calculate_losses(policy, value, batch)
        
        optimizer.zero_grad()
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=GRAD_CLIP)
        
        optimizer.step()
        
        return {
            'loss': loss.item(),
            'policy_grad_norm': self.policy_layer.weight.grad.norm().item(),
            'value_grad_norm': self.value_layer.weight.grad.norm().item(),
            'lstm_grad_norm': self.lstm.weight_ih.grad.norm().item()
        }

    @torch.no_grad()
    def inference_gpu(self, inputs, goals, hidden_state, device):
        inputs = inputs.to(device)
        goals = goals.to(device)
        hidden_state = (hidden_state[0].to(device), hidden_state[1].to(device))
        
        policy, value, new_hidden, blocking, on_goal, valids = self(inputs, goals, hidden_state)
        
        return policy, value, new_hidden, blocking, on_goal, valids


def benchmark_gpu(model, batch_sizes=[1, 32, 64, 128, 256], num_iterations=100):

    print("\nRunning GPU benchmark...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    results = {}
    for batch_size in batch_sizes:
        dummy_input = torch.randn(batch_size, 4, 10, 10).to(device)
        dummy_goals = torch.randn(batch_size, 3).to(device)
        hidden_state = model.init_hidden(batch_size)
        hidden_state = (hidden_state[0].to(device), hidden_state[1].to(device))
        
        for _ in range(10):
            _ = model(dummy_input, dummy_goals, hidden_state)
        
        torch.cuda.synchronize()
        start_time = time.time()
        for _ in range(num_iterations):
            _ = model(dummy_input, dummy_goals, hidden_state)
            torch.cuda.synchronize()
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_iterations * 1000  
        throughput = batch_size / (avg_time / 1000)  
        
        results[batch_size] = {
            'avg_time_ms': avg_time,
            'throughput': throughput
        }
        
        print(f"\nBatch size: {batch_size}")
        print(f"Average inference time: {avg_time:.2f} ms")
        print(f"Throughput: {throughput:.2f} samples/sec")
    
    return results


def test_gpu_memory(model, max_batch_size=512, step=32):

    print("\nTesting GPU memory usage...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    batch_size = step
    memory_usage = []
    
    try:
        while batch_size <= max_batch_size:
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated()
            
            dummy_input = torch.randn(batch_size, 4, 10, 10).to(device)
            dummy_goals = torch.randn(batch_size, 3).to(device)
            hidden_state = model.init_hidden(batch_size)
            hidden_state = (hidden_state[0].to(device), hidden_state[1].to(device))
            
            _ = model(dummy_input, dummy_goals, hidden_state)
            
            current_memory = torch.cuda.memory_allocated()
            memory_used = (current_memory - initial_memory) / 1024 / 1024  
            
            memory_usage.append((batch_size, memory_used))
            print(f"Batch size: {batch_size}, Memory used: {memory_used:.2f} MB")
            
            batch_size += step
            
    except RuntimeError as e:
        print(f"\nOut of memory at batch size {batch_size}")
        print("Maximum safe batch size is", batch_size - step)
    
    return memory_usage


def test_gradients(model, batch_size=32, device="cpu"):

    print("\nTesting gradient flow...")
    model = model.to(device)
    model.train()
    
    torch.manual_seed(42)
    dummy_input = torch.randn(batch_size, 4, 10, 10, requires_grad=True).to(device)
    dummy_goals = torch.randn(batch_size, 3, requires_grad=True).to(device)
    hidden_state = model.init_hidden(batch_size)
    hidden_state = (hidden_state[0].to(device), hidden_state[1].to(device))
    
    policy, value, new_hidden, blocking, on_goal, valids = model(dummy_input, dummy_goals, hidden_state)
    
    value_target = torch.randn(batch_size, 1).to(device)
    action_target = torch.randint(0, 5, (batch_size,)).to(device)
    blocking_target = torch.randint(0, 2, (batch_size, 1)).float().to(device)
    on_goal_target = torch.randint(0, 2, (batch_size, 1)).float().to(device)
    
    policy_loss = -torch.sum(torch.log(policy) * F.one_hot(action_target, 5))

    value_loss = F.mse_loss(value, value_target)

    blocking_loss = F.binary_cross_entropy(blocking, blocking_target)

    on_goal_loss = F.binary_cross_entropy(on_goal, on_goal_target)
    
    loss = policy_loss + 0.5 * value_loss + 0.5 * blocking_loss + 0.5 * on_goal_loss
    
    model.zero_grad()
    loss.backward()
    
    grad_dict = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            # для параметрів розміру [1] (bias) не рахуємо std
            if param.grad.numel() > 1:
                grad_std = param.grad.std().item()
            else:
                grad_std = None
                
            grad_dict[name] = {
                "grad_exists": True,
                "grad_mean": param.grad.abs().mean().item(),
                "grad_std": grad_std,
                "grad_shape": param.grad.shape,
                "grad_norm": param.grad.norm().item()
            }
            
            grad_dict[name] = {
                "grad_exists": True,
                "grad_mean": param.grad.abs().mean().item(),
                "grad_std": grad_std,
                "grad_shape": param.grad.shape,
                "grad_norm": param.grad.norm().item()
            }
        else:
            grad_dict[name] = {
                "grad_exists": False,
                "grad_mean": None,
                "grad_std": None,
                "grad_shape": None,
                "grad_norm": None
            }

    results_str = []
    for name, info in grad_dict.items():
        layer_info = [f"\nLayer: {name}"]
        if info["grad_exists"]:
            layer_info.extend([
                "Gradient exists: True",
                f"Gradient mean: {info['grad_mean']:.6f}",
                "Gradient std: {:.6f}".format(info['grad_std']) if info['grad_std'] is not None else "Gradient std: N/A (single value)",
                f"Gradient shape: {info['grad_shape']}",
                f"Gradient norm: {info['grad_norm']:.6f}"
            ])
        else:
            layer_info.append("No gradients")
        results_str.extend(layer_info)
    
    print("\n".join(results_str))
            
    test_results = {
        "policy": {"shape": policy.shape, "grad_enabled": policy.requires_grad},
        "value": {"shape": value.shape, "grad_enabled": value.requires_grad},
        "blocking": {"shape": blocking.shape, "grad_enabled": blocking.requires_grad},
        "on_goal": {"shape": on_goal.shape, "grad_enabled": on_goal.requires_grad},
        "valids": {"shape": valids.shape, "grad_enabled": valids.requires_grad}
    }        
    print(f"\nLayer: {name}")
    print(f"Gradient exists: {grad_dict[name]['grad_exists']}")
    if grad_dict[name]['grad_exists']:
        print(f"Gradient mean: {grad_dict[name]['grad_mean']:.6f}")
        if grad_dict[name]['grad_std'] is not None:
            print(f"Gradient std: {grad_dict[name]['grad_std']:.6f}")
        else:
            print("Gradient std: N/A (single value)") 
        print(f"Gradient shape: {grad_dict[name]['grad_shape']}")
    
    input_grad_dict = {
        "input_grad": {
            "exists": dummy_input.grad is not None,
            "mean": dummy_input.grad.abs().mean().item() if dummy_input.grad is not None else None,
            "std": dummy_input.grad.std().item() if dummy_input.grad is not None else None
        },
        "goal_grad": {
            "exists": dummy_goals.grad is not None,
            "mean": dummy_goals.grad.abs().mean().item() if dummy_goals.grad is not None else None,
            "std": dummy_goals.grad.std().item() if dummy_goals.grad is not None else None
        }
    }
    
    print("\nInput gradients:")
    print(f"Input gradient exists: {input_grad_dict['input_grad']['exists']}")
    print(f"Goal gradient exists: {input_grad_dict['goal_grad']['exists']}")
    
    return {
        "test_results": test_results,
        "gradients": grad_dict,
        "input_gradients": input_grad_dict
    }


def test_gradient_flow_all(model):

    print("Testing gradient flow on CPU...")
    cpu_results = test_gradients(model, device="cpu")
    
    if torch.cuda.is_available():
        print("\nTesting gradient flow on GPU...")
        gpu_results = test_gradients(model, device="cuda")
        return {"cpu": cpu_results, "gpu": gpu_results}
    
    return {"cpu": cpu_results}


def test_model(use_gpu=False):
    print("Testing ACNet PyTorch implementation...")
    device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
    
    batch_size = 32
    grid_size = 10
    a_size = 5
    
    model = ACNet("test", a_size, grid_size, training=True)
    
    obs = torch.randn(batch_size, 4, grid_size, grid_size)
    goals = torch.randn(batch_size, 3)
    hidden_state = model.init_hidden(batch_size)
    
    policy, value, new_hidden, blocking, on_goal, valids = model(obs, goals, hidden_state)
    
    assert policy.shape == (batch_size, a_size), f"Policy shape wrong: {policy.shape}"
    assert value.shape == (batch_size, 1), f"Value shape wrong: {value.shape}"
    assert blocking.shape == (batch_size, 1), f"Blocking shape wrong: {blocking.shape}"
    assert on_goal.shape == (batch_size, 1), f"On_goal shape wrong: {on_goal.shape}"
    assert valids.shape == (batch_size, a_size), f"Valids shape wrong: {valids.shape}"
    
    print("All tests passed!")

if __name__ == "__main__":
    test_model()