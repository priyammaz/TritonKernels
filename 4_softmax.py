"""
Fused Softmax Kernel (Forward)

This is an implementation of Stable Softmax for numerical
stability!

What is a "Fused" kernel? Every time you do an operation on a
GPU, this triggers a GPU operation. This involved taking data
from the main GPU memory, copying to cache (SRAM) and then copying
back to main memory. 

Unfortunately, each of these memory transfers are the slow part (as most 
GPU ops are memory bound and not compute bound). So what if instead, we did
one copy to SRAM, did all our operations, and then copied back?

That is exactly what Kernel Fusion is! We cannot do this in Pytorch be 
default, each call in pytorch like .sum(), .max(), etc..., each are a
separate kernel call. But in Triton (or CUDA) we can organize this better
and group all the operations together into a single call!
"""
import pytest
from pathlib import Path
import torch
import triton
import triton.language as tl

def naive_softmax(x):

    """
    Implementation of Stable Softmax
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

def calc_num_warps(block_size):
        num_warps = 4
        if block_size >= 2048:
            num_warps = 8
        if block_size >= 4096:
            num_warps = 16
        return num_warps

@triton.heuristics({"num_warps": lambda args: calc_num_warps(args["BLOCK_SIZE"])})
@triton.jit
def softmax_kernel_forward(
    output_ptr, 
    input_ptr, 
    input_row_stride, 
    output_row_stride, 
    n_cols: tl.constexpr,  
    BLOCK_SIZE: tl.constexpr):

    """

    ### NAIVE SOFTMAX ###
    In normal softmax, we do the following ops:

    1) max on every row
    2) subtract every row by max
    3) Exponentiate every value
    4) Sum together everything
    5) divide every value by this sum

    That is 5 kernel launches on our GPUs to do this one Op!!!

    ### FUSED SOFTMAX ###
    We will fuse these ops into a single Kernel Launch using Triton!! 

    This is dumb softmax implementation. The idea is that we have the following matrix (N x M), 
    and we want to compute softmax for each N across the M. 

    row 0: [x00 x01 x02 ... x0M]
    row 1: [x10 x11 x12 ... x1M]
    row 2: [x20 x21 x22 ... x2M]
    ...
    row N: [xR0 xR1 xR2 ... xRM]

    ### GPU Programming Basics ###
    Softmax is a great place to start as its pretty easy to setup. Some terminology:
    
    GPUs are massively parallel processes that run thousands of threads at the same time (each thread doing a tiny bit of work)
    This means we need some structure on these threads so we can map different threads to our operation. 

    # TERMINOLOGY
    Thread -> Smallest unit of execution. They perform a sequence of instructions to the data you point to

    Thread Blocks -> A group of threads that work together on a chunk of data. Thread blocks can also 
                        share memory so one thread can refer to data on another thread as long as they are in 
                        that same block. These can have upto 3 dimensions (x,y,z) for complex mappings


    Grid -> The grid is the collection of thread blocks that cover your full tensor. 

    Kernel Launch -> A kernel is a function that runs on GPU. It operates in parallel with many threads executing
                     the same instruction on different chunks of the data. 

    Warp -> Threads are grouped into groups of 32 (one warp) and exectute the instruction in lock-step. 
            Thus a single blocks threads are grouped into warps. 32 is an upper limit though. The larger
            your block the more warps you need. Warps are also executed concurrently, but again has hardwarde
            limitations. Most modern gpus can launch 64 warps (with 32 threads per warp) in parallel

    # TUNING

    Things lie your warp size, block size, etc.. need to be tuned! It is hardware and operation specific. Triton supports
    autotuning, but for now we will just use a dumb heuristic, mainly because our block size is fixed so all we can really
    mess with is the warp

    ```
    def calc_num_warps(block_size):
        num_warps = 4
        if block_size >= 2048:
            num_warps = 8
        if block_size >= 4096:
            num_warps = 16
        return num_warps
    ```

    As you can see as our block size increases we want more warps to better parallize the ops. 
   
    # Example w/ Softmax
    There will be two assumptions we are making here that makes our softmax naive. 
    
    1) Block_Size = Num Columns. I will set the block size to be at least as large as the number of columns I want to process in a row. 
                    This ensures that the threads in the block can collectively cover the entire row.

        LIMITATION: On modern GPUs, the maximum number of threads per block is 1024, if the number of columns exceed that
                    we cannot assign one thread per column directly. In this case, each thread has to process multiple
                    columns in sequence! Thus we lose some of our parallelism. Luckily, this looping is handled for us 
                    by triton automatically! 

                    The actual thread count per block in the underlying CUDA kernel generated by triton is 
                    controlled by num_warps parameter multiplied by the warp size (default of 32). This means if you
                    have 8 warps, each warp with 32 threads, you will have a total of 256 threads being used. You can 
                    go upto 32 warps for an effective 32 * 32 = 1024 which is the hardware max. 

                    When block size exceeds this thread counts (i.e. BLOCK_SIZE=2048) and we only have 256 threads, Triton
                    will automagically generate code where each thread processes multiple elements sequentially within the block. 


    2) Each block will process one row

        LIMITATION: The expensive part of GPU compute is the CPU/GPU communication overhead. And every kernel launch is expensive. 
                    In our case, we will launch a bunch of tiny kernels, each processing a row. In the example provided in the 
                    Triton tutorials (https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html#sphx-glr-getting-started-tutorials-02-fused-softmax-py)
                    you will see that they actually loop over the rows. This means each kernel launch is processing a chunk of rows
                    at a time rather than one at a time. Basically we want every kernel launch to do more work!


    # Structure of Kernels

    Lets say we have the following input matrix. This has 4 rows and 8 columns

    Input matrix:
    row 0: [x00 x01 x02 x03 x04 x05 x06 x07]
    row 1: [x10 x11 x12 x13 x14 x15 x16 x17]
    row 2: [x20 x21 x22 x23 x24 x25 x26 x27]
    row 3: [x30 x31 x32 x33 x34 x35 x36 x37]

    This means we will define a block size of 8 and a grid size of 4. This is because we have 8 things we want 
    to compute over in every block, and we have 4 rows. We will then get something like this

    Thread Block 0 -> processes row 0
    Thread Block 1 -> processes row 1
    Thread Block 2 -> processes row 2
    Thread Block 3 -> processes row 3

    Now inside a single block, we have our 8 threads doing something

    Block 0 (row 0):
    Thread 0 -> x00
    Thread 1 -> x01
    Thread 2 -> x02
    Thread 3 -> x03
    Thread 4 -> x04
    Thread 5 -> x05
    Thread 6 -> x06
    Thread 7 -> x07

    ### Strides Tell Us the Layout of the Matrix ###

    Now we see matricies as a 2d array in this case. But internally on the GPU, its organized as a single long vector. The stride
    tells us the relation of our matrix this vector. For example we see our matrix as:

    (4 x 8) Input matrix:
    [x00 x01 x02 x03 x04 x05 x06 x07]
    [x10 x11 x12 x13 x14 x15 x16 x17]
    [x20 x21 x22 x23 x24 x25 x26 x27]
    [x30 x31 x32 x33 x34 x35 x36 x37]

    But internally its stored as:

    [x00 x01 x02 x03 x04 x05 x06 x07 x10 x11 x12 x13 x14 x15 x16 x17 x20 x21 x22 x23 x24 x25 x26 x27 x30 x31 x32 x33 x34 x35 x36 x37]

    The stride tells me the number of elements I need to jump in memory (this vector form) to move from one row (or column) to the next. 
    In this case, to go from one column to the next, I only need to move over one element! But to move from one row to the next I need
    to move over 8 elements:

    Thus:

        row_stride = 8
        col_stride = 1

    This can be more complicated for higher dimensional arrays, but the principle is the same. For every dimension in your array, how many 
    elements do you need to move to get to the next index of that dimension. 

    ### Pointers ###
    Triton (like CUDA) uses pointers to do their operations. A pointer is just a fancy way of saying, "address to specific location in memory". 
    In most cases, the pointer will point to the STARTING value of your matrix (in our case x00). We then need to use offsets to get to the 
    actual location we want. For example:

    If we input_ptr points to x00, and we want the start of every row:

        x10 = input_ptr + 8 = input_ptr + row_stride * 1
        x20 = input_ptr + 16 = input_ptr + row_stride * 2
        x30 = input_ptr + 24 = input_ptr + row_stride * 3

    ######## SUPER CAVEAT!!! ############
    We will be using CuPY as triton works with anything that has a GPU pointer. torch.stides() returns the number of elements you need to 
    move over to go to the next row. Cupy.stride returns the number of bytes you need to move to go to the next row! This means when working 
    in cupy, we have to convert. For example. 

    ```
    rand_torch = torch.randn((32,32), dtype=torch.float32)
    print(rand_torch.stride()) -> (32,1) 

    rand = cp.random.normal(size=(32,32), dtype=cp.float32)
    print(rand.strides) -> (128,4)
    
    ```

    As we can see, cupy gives (128,4,) because it says you have to move over 128 bytes to get to the next row and 4 bytes to get to the next 
    column. Well in float32 precision, each element has 4 bytes. So if you divide, its (128/4=32, 4/4=1) = (32,1) which is the same as torch!

    In cupy we can get this easily with rand.itemsize

    ```
    rand = cp.random.normal(size=(32,32), dtype=cp.float32)
    print(rand.strides) -> (128,4)
    print(rand.strides[0]/rand.itemsize, rand.strides[1]/rand.itemsize) -> (32,1)
    ```

    ### Saving Results ###
    Once we have done our computation, we need a place to save it. This is why triton kernels take both an input pointer (the data you
    want to perform the operation on) and the output pointer (the matrix you want to store these results in). Softmax has the same input
    and output shape so its easy, but other ops could require a bit more effort in indexing/


    ### TLDR ###
    GPU Programming is all about 2 things:

        1) Mapping indexes of your threads/threadblocks to a grid that covers your whole tensor efficiently. This requires some
           pointer arithmetic to index correctly. 
        2) Tuning block sizes and warps to optimize parallelism and memory pressure 

    """
    
    ### Each thread block will process one row. I provide ahead of time the ###
    ### number of thread blocks (its our grid) which in thise case is just number of rows ###
    ### Thus this thread block index is just our row index ###
    row_idx = tl.program_id(0)

    ### We have the row index in our threadblock, but we actually need to grab that row of data ###
    ### from memory. So lets index it! Our input_ptr points to the starting point of our data ###
    ### so we just need to index to the row_idxth row of our matrix. 
    row_start_ptr = input_ptr + row_idx * input_row_stride

    ### Now that we have the starting point of a row, we need indexes for the full column ###
    ### for example if we are at index 8 and there are 8 elements we need to do ###
    ### 8 + [0,1,2,3,4,5,6,7] -> [8,9,10,11,12,13,14,15] ###
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets

    ### Mask invalid positions. Now here is a consideration for Triton. Each block must have ###
    ### a power of two number of elements! This is fine if we were computing softmax over a  ###
    ### power of two number of elements (like 8). But what if we had 7 elements? ###
    ### Lets say for the first row, we have 7 elements, so the indexes would be ###
    ### [0,1,2,3,4,5,6] ###
    ### But our block size has to be 8 as that is the closest power of 2 larger than 7. Then our ###
    ### block size will try to index [0,1,2,3,4,5,6,7]! But this it not correct! If each row had only ###
    ### 7 elements, the block will index the entire row as well as the first element of the next row. ###
    ### thus we have a mask, that tells triton which indexes we have are valid and which are invalid. ###
    ### if our col_offsets [0,1,2,3,4,5,6,7] is less than 7 we are good, otherwise its masked out! ###
    mask = col_offsets < n_cols

    ### Now that we have indexes its time to load our data ###
    ### But another consideration. Invalid values that are indexed, although are masked out and are not read ###
    ### in, triton will still load something, itll just be some undefined value. So we have to provide some ###
    ### fill value that wont affect our operations. This depends on what we are doing. In softmax we have to ###
    ### take a max and exponeniate + sum. So if we fill in -inf, the max stays the same as before, and after ###
    ### exponentiating (whcih makes it 0) it has to effect on the sum ###
    row = tl.load(input_ptrs, mask=mask, other=-float("inf"))

    ### Now that I have my row lets compute softmax like normal! ###
    row_minus_max = row - tl.max(row, axis=0)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = (numerator / denominator).to(output_ptr.dtype.element_ty)

    ### Store output in our output matrix. Again we compute the pointer to the output (which in this case is the) ###
    ### same logic as the input so pretty easy !! ###
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=mask)

def fused_softmax_forward(x):

    n_rows, n_cols = x.shape

    ### Each block needs to process the ENTIRE row (or whatever is the next power of 2 masked)
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    y = torch.empty_like(x)
    row_stride = x.stride(0)

    grid = (n_rows,)
    softmax_kernel_forward[grid](
        y,       # output_ptr
        x,       # input_ptr
        row_stride,       # input_row_stride
        row_stride,       # output_row_stride
        n_cols,           # n_cols (constexpr)
        BLOCK_SIZE        # BLOCK_SIZE (constexpr)
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
    out = fused_softmax_forward(x)

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
        x_vals=[128 * i for i in range(1, 33)],  # 128 to 4096
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
        ms = triton.testing.do_bench(lambda: fused_softmax_forward(x))

    gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms)

if __name__ == "__main__":
    
    print("Running Test...")
    test_path = Path(__file__).resolve()
    pytest.main([f"{test_path}::test_softmax", "-v", "-s"])

    # Run benchmarks
    print("Running column size benchmark...")
    benchmark_softmax.run(show_plots=True, print_data=True)