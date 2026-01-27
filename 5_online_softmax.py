"""
In 4_softmax.py we were able to write a kernel that processed
every row of an MxN matrix (basically compute softmax over every N elements,
where each thread processed one of the M rows)

Although performant there are a few issues with this when N is (very) large

    1. Each thread is processing a lot of data, so its expensive on GPU memory and Cache
    2. For stable softmax we first find the max over the row and then process the row with exp and sum (two passes)

Online Softmax allows us to do this in one pass without storing all the logits! We can do this element by element or 
block of elements by block of elements. 

### Lets Recap ###

NAIVE SOFTMAX: softmax(xᵢ) = exp(xᵢ) / Σⱼ exp(xⱼ)
STABLE SOFTMAX: softmax(xᵢ) = exp(xᵢ - m) / Σⱼ exp(xⱼ - m), where m = max(x)

Normally this is a two pass algorithm:

    First Pass: Find maximum m = max(x)
    Second Pass: Compute exp(xᵢ - m) and sum, then divide

### How is Online Softmax Different? ###

If we cannot look at the entire vector that we are computing softmax over then we have 
some issues. For example, we need the entire max over the vector, but we cant look at the
whole vector, so what do we do? We have to keep track of some running statistics!!

At every step i we have to:
    
    1. mᵢ = max(x_1,...xᵢ) -> the max upto this specific step
    2. dᵢ = sum{j=(1->i)} exp(x_j-mᵢ) -> running sum of exponentials

But as our estimate for the max mᵢ continues to change (as we may later see a larger value) we need
to update our running sum of exponentials! Lets see how this all comes together


### Derivation 

When we see a new element xᵢ₊₁, we compare xᵢ₊₁ to xᵢ by doing mᵢ₊₁ = max(mᵢ, xᵢ₊₁)
to see if we have a new max value along this vector

Remember we also have dᵢ = sum{j=(1->i)} exp(x_j-mᵢ) upto step i. What we want is for 
step i+1:

dᵢ₊₁ = sum{j=(1->i+1)} exp(x_j-mᵢ₊₁)

Thus we have two cases here:

Case 1:

    The new xᵢ₊₁ > mᵢ, which means our running stat for the max(x) has changed. Unfortunately this means
    we have been using the incorrect max(x) for all the steps before this, and need to update and rescale
    all our previous terms!

    To be specific, the max we were using uptil now was mᵢ, but the new max we should have been using is mᵢ₊₁

    So the sum we had been doing is:

        dᵢ = sum{j=(1->i)} exp(x_j-mᵢ)

    But what we really want now is:
        
        dᵢ = sum{j=(1->i)} exp(x_j-mᵢ₊₁) using the new max mᵢ₊₁

    But for our new step i+1 we must express everything relative to the new max mᵢ₊₁

    To rescale the sum all we need to do this write each term like this:

        exp(x_j - mᵢ) = exp(x_j - mᵢ₊₁ + mᵢ₊₁ - mᵢ) = exp(x_j - mᵢ₊₁) * exp(mᵢ₊₁ - mᵢ)

        We now see the exp(x_j - mᵢ₊₁) that we wanted, and the scaling factor exp(mᵢ₊₁ - mᵢ), so lets
        rearrange this

        exp(x_j - mᵢ) = exp(x_j - mᵢ₊₁) * exp(mᵢ₊₁ - mᵢ)
        exp(x_j - mᵢ₊₁) = exp(x_j - mᵢ) / exp(mᵢ₊₁ - mᵢ) = exp(x_j - mᵢ) * exp(mᵢ - mᵢ₊₁)

        Now remember from earlier we have the sum still, so what we want is 

        sum{j=(1->i)} exp(x_j-mᵢ₊₁)

        What we have expressed is only

        exp(x_j - mᵢ₊₁)

        So lets introduce that sum back

        sum{j=(1->i)} exp(x_j-mᵢ₊₁) = sum{j=(1->i)} exp(x_j - mᵢ) * exp(mᵢ - mᵢ₊₁)

        But exp(mᵢ - mᵢ₊₁) has no dependecy on j so we can pull it out and get:

        sum{j=(1->i)} exp(x_j-mᵢ₊₁) = exp(mᵢ - mᵢ₊₁) * sum{j=(1->i)} exp(x_j - mᵢ)

        But wait! Wasnt sum{j=(1->i)} exp(x_j - mᵢ) just our dᵢ from earlier?? Yes!
        So we can finally write our adjustment term as 

        sum{j=(1->i)} exp(x_j-mᵢ₊₁) = dᵢ * exp(mᵢ - mᵢ₊₁)

        But this only adjusts our sum UPTO step i, we are currently on i+1. But that is 
        easy because if our max is mᵢ₊₁ = xᵢ₊₁, then exp(xᵢ₊₁ - mᵢ₊₁) = exp(0) = 1

        Therefore, our running sum:

        dᵢ₊₁ = sum{j=(1->i+1)} exp(x_j-mᵢ₊₁) = dᵢ * exp(mᵢ - mᵢ₊₁) + 1, where:

            dᵢ * exp(mᵢ - mᵢ₊₁) is the adjustment to our previous sum
            1 is the new term we are adding on

Case 2:

    The new xᵢ₊₁ <= mᵢ, and then mᵢ₊₁ = mᵢ this means our the running max we have until now is still valid! There
    is nothing to adjust then and we just add onto our sum like normal

        dᵢ₊₁ = dᵢ + exp(xᵢ₊₁ - mᵢ₊₁)

We can then combine both cases as follows:

    mᵢ₊₁ = max(mᵢ, xᵢ₊₁)
    dᵢ₊₁ = dᵢ * exp(mᵢ - mᵢ₊₁) + exp(xᵢ₊₁ - mᵢ₊₁)

    This works because:

        if mᵢ₊₁ = mᵢ (case 2) then exp(mᵢ - mᵢ₊₁) = 1 and dᵢ₊₁ = dᵢ * 1 + exp(xᵢ₊₁ - mᵢ₊₁)
        if mᵢ₊₁ = xᵢ₊₁ (case 1) then exp(xᵢ₊₁ - mᵢ₊₁) = exp(0) = 1, and then dᵢ₊₁ = dᵢ * exp(mᵢ - mᵢ₊₁) + 1

### Computing Softmax From This:

Once we have looped through all the elements in the vector x we have the following for the entire vector:

    - The true max m = max(x)
    - the true exp-sum d = sum{j=(1->N)} exp(x_j-m)

Remember that softmax(xᵢ) = exp(xᵢ - m) / Σⱼ exp(xⱼ - m), where m = max(x)

So we can compute all of this from what we have!

softmax(xᵢ) = exp(xᵢ - m) / d

"""

import pytest
from pathlib import Path
import torch
import math
import triton
import triton.language as tl

def naive_softmax(x):

    """
    Implementation of Naive Stable Softmax
    """
    # Subtract max for numerical stability
    row_max = torch.max(x, axis=1, keepdims=True).values # shape (n_rows, 1)
    x_stable = x - row_max

    # Exponentiate
    x_exp = torch.exp(x_stable)

    # Sum along rows
    row_sum = torch.sum(x_exp, axis=1, keepdims=True)

    # Divide
    out = x_exp / row_sum

    return out

def online_softmax(x):

    """
    Implementation of Online Softmax
    """

    ### Get shape of X ###
    M, N = x.shape

    ### Create Empty Tensor to Store Final Softmax Output ###
    output = torch.empty_like(x)

    ### We will parallelize over the row dimension ###
    for i in range(M):

        ### Index the Row ###
        xi = x[i]

        ### Initialize for Online Softmax ###
        m = float("-inf")
        d = 0

        ### Loop over elements of data ###
        for j in range(N):
            
            ### Get element ###
            xj = xi[j].item()

            ### Update max Statistic ###
            m_prev = m
            m = max(m, xj)

            ### Update Sum ###
            d = d * math.exp(m_prev - m) + math.exp(xj - m)

        ### Compute Softmax ###
        softmaxed_x = torch.exp(xi - m) / d

        ### Store in output tensor ###
        output[i] = softmaxed_x

    return output

def online_blocked_softmax(x, block_size=64):

    """
    Implementation of Online Softmax with Block processing

    So instead of looping through element by element, we will
    instead loop through a block of elements at a time!
    """

    ### Get shape of X ###
    M, N = x.shape

    ### Create Empty Tensor to Store Final Softmax Output ###
    output = torch.empty_like(x)

    ### We will parallelize over the row dimension ###
    for i in range(M):

        ### Index the Row ###
        xi = x[i]

        ### Initialize for Online Softmax ###
        m = float("-inf")
        d = 0

        ### Loop over blocks of elements of data ###
        num_blocks = math.ceil(N/block_size)
        for j in range(num_blocks):

            ### Get start and End of Block ###
            block_start = j * block_size
            block_end = block_start + block_size
            block_idx = torch.arange(block_start, block_end)

            ### Compute Mask So we dont index invalid positions ###
            mask = (block_idx < N)
            block_idx = block_idx[mask]

            ### Load Block ###
            block = xi[block_idx]

            ### Compute block max ###
            block_max = block.max().item()

            ### Update running max ###
            m_prev = m
            m = max(m, block_max)

            ### Update running denominator ###
            d = d * math.exp(m_prev - m) + torch.exp(block - m).sum().item()

        ### Compute Softmax ###
        softmaxed_x = torch.exp(xi - m) / d

        ### Store in output tensor ###
        output[i] = softmaxed_x

    return output

@triton.jit
def online_blocked_softmax_kernel(
    output_ptr, 
    input_ptr, 
    output_row_stride, 
    input_row_stride,
    n_cols: tl.constexpr, 
    BLOCK_SIZE: tl.constexpr
):
    
    ### Which Row Are We Processing? ###
    pid = tl.program_id(0)

    ### Advance pointer to start of row ###
    row_start = pid * input_row_stride
    
    ### Start running statistics ###
    m = float("-inf")
    d = 0.0

    ### First Pass to compute m, d ###
    for offs in range(0, tl.cdiv(n_cols, BLOCK_SIZE)):

        ### Indexes for block ###
        ### offs * BLOCK_SIZE advances to the correct start of the block
        ### + tl.arange(0, BLOCK_SIZE)  gives all the indexes for that block
        col_offsets = offs * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE) 
        
        ### Compute mask for invalid positions ###
        mask = col_offsets < n_cols

        ### Load Data ###
        ### input_ptr + row_start -> advance to the correct row of data
        ### + col_offsets will advance to the indexes in this block of the row we are processing
        ### We set masked positions as -inf so when we compute max, it doesnt effect anything, and when
        ### we exponentiate for the accumulated sum it will be exp(-inf) = 0 so it doesnt effect that either
        x = tl.load(input_ptr + row_start + col_offsets, mask=mask, other=-float("inf"))

        ### Get the max in the block ###
        block_max = tl.max(x, axis=0)

        ### Update our max estimate ###
        m_prev = m
        m = tl.maximum(m, block_max)

        ### Update our accumulated sum estimate ###
        d = d * tl.exp(m_prev - m) + tl.sum(tl.exp(x-m), axis=0)

    ### Compute Softmax ###
    for offs in range(0, tl.cdiv(n_cols, BLOCK_SIZE)):

        ### Advance to the block again ###
        col_offsets = offs * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

        ### Compute mask for invalid positions ###
        mask = col_offsets < n_cols

        ### Load Block of Data ###
        x = tl.load(input_ptr + row_start + col_offsets, mask=mask, other=-float("inf"))

        ### Compute softmax for the output ###
        y = tl.exp(x - m) / d

        ### Store the Output ###
        tl.store(output_ptr + pid * output_row_stride + col_offsets, y, mask=mask)

def fused_online_blocked_softmax(x, block_size=256):
    """
    x: (M, N) torch.Tensor (CUDA)
    """
    assert x.is_cuda
    M, N = x.shape
    y = torch.empty_like(x)

    grid = (M,)

    online_blocked_softmax_kernel[grid](
        y,
        x,
        y.stride(0),
        x.stride(0),
        N,
        BLOCK_SIZE=block_size,
    )

    return y

### TESTS ###
@pytest.mark.parametrize("M, N", [
    (64, 64),
    (128, 128),
    (256, 256),
    (512, 512),
    (1024, 1024),
    (2048, 2048),
    (256, 512),
    (512, 256),
    (1024, 512),
    (64, 1024),
    (1024, 64),
    (128, 100),
    (256, 300),
    (512, 777),
])
def test_softmax(M, N):
    """
    Test fused softmax against PyTorch's naive softmax implementation.
    M = number of rows
    N = number of columns
    """
    # Create random input
    x = torch.randn(size=(M, N), dtype=torch.float32, device="cuda")
    
    # Reference implementation (naive_softmax from your code)
    out_ref = naive_softmax(x)
    
    # Your fused kernel
    out = fused_online_blocked_softmax(x)

    # Validate results
    assert torch.allclose(out, out_ref, atol=1e-5, rtol=1e-5), \
        f"Softmax failed for M={M}, N={N}"
    
    # Log useful info
    max_diff = float(torch.max(torch.abs(out - out_ref)))
    mean_diff = float(torch.mean(torch.abs(out - out_ref)))
    print(f"Softmax Success: M={M}, N={N} | Max Diff={max_diff:.6f} | Mean Diff={mean_diff:.6f}")

# Benchmark configuration
configs = [
    triton.testing.Benchmark(
        x_names=["N"],  # Number of columns
        x_vals=[256 * i for i in range(1, 384, 16)],  # 128 to 4096
        line_arg="provider",
        line_vals=["torch", "naive", "fused"],
        line_names=["PyTorch F.softmax", "Naive Softmax", "Fused Softmax"],
        styles=[("blue", "-"), ("orange", "-"), ("green", "-")],
        ylabel="GB/s",
        plot_name="softmax-performance",
        args={"M": 4096},  # Fixed number of rows
    )
]

@triton.testing.perf_report(configs)
def benchmark_softmax(M, N, provider):
    """
    Benchmark softmax implementations.
    Reports throughput in GB/s (gigabytes per second).
    """
    x = torch.randn((M, N), device="cuda", dtype=torch.float32)
    
    if provider == 'torch':
        ms = triton.testing.do_bench(lambda: torch.softmax(x, dim=1))
    elif provider == 'naive':
        ms = triton.testing.do_bench(lambda: naive_softmax(x))
    elif provider == 'fused':
        ms = triton.testing.do_bench(lambda: fused_online_blocked_softmax(x))

    gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms)

if __name__ == "__main__":
    
    ### Check our Pseudocode ###
    M, N = 25, 500
    rand = torch.randn(M, N, device="cuda")
    naive_out = naive_softmax(rand)
    online_out = online_softmax(rand)
    blocked_online_out = online_blocked_softmax(rand)
    assert torch.allclose(naive_out, online_out)
    assert torch.allclose(naive_out, blocked_online_out)

    print("Running Test...")
    test_path = Path(__file__).resolve()
    pytest.main([f"{test_path}::test_softmax", "-v", "-s"])

    # Run benchmarks
    print("Running column size benchmark...")
    benchmark_softmax.run(show_plots=True, print_data=True)

# if __name__ == "__main__":

    # ### Check our Pseudocode ###
    # M, N = 25, 500
    # rand = torch.randn(M, N, device="cuda")
    # naive_out = naive_softmax(rand)
    # online_out = online_softmax(rand)
    # blocked_online_out = online_blocked_softmax(rand)
    # assert torch.allclose(naive_out, online_out)
    # assert torch.allclose(naive_out, blocked_online_out)

#     ### Check our Fused Online Softmax ###
#     fused_output = triton_online_blocked_softmax(rand)
#     assert torch.allclose(naive_out, fused_output)

    





    
