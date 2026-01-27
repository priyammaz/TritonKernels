"""
Sum of Two Vectors a + b w/ Blocking

In our previous example in 1_vector_add.py each PID processed a single value. The problem with this
typically latency especially for very large tensor processing. There are hardware limits of how many 
things we can do at once and how they are organized to the SM (streaming multiprocessors) on your GPU. 

The job of CUDA is to take the operation I provide with some thread structure and to schedule it onto the GPU. 
But if we have a vector sum with millions of elements, that would then be millions of things that need to be 
scheduled, leading to a scheduling overhead. 

It would be much smarter to have each scheduled operation process a chunk of our vector rather than a single 
element! The chunk size is indicated typically as BLOCK_SIZE and is mostly a hyperparameter you can tune 
for optimal performance, but in Triton BLOCK_SIZE MUST BE A POWER OF 2. This is totally fine, but means
we have to keep an eye on some stuff for operations that have non-power of of 2 elements through masking!
""" 

import time
import math
import torch 
import triton
import triton.language as tl

def blocked_vector_add_pseudocode(a, # first vectors
                                  b, # second vector
                                  output, # output vector to store results
                                  grid, # Index grid that we will use to process our vectors (this time one idx per block of operation)
                                  n_elements, # Number of total elements in this vector sum
                                  BLOCK_SIZE): # number of elements in the vector processed by each PID
    """
    Simple vector sum where every PID will process a BLOCK_SIZE number of elements
    """

    ### Define our Indexes For our Grid. In Triton this is already made for us ###
    PIDS = torch.arange(grid[0])

    ### Fake Pointers. In Triton we pass pointers in directly and we offset them by the PID ###
    ### that is being processed. In this case we will just start our pointers as "0" and offset it ###
    ### NOTE: In this psuedocode we are offsetting the index of the data, in Triton we will be offsetting ###
    ### the memory locations of the data which is an important distinction for later kernels!
    a_ptr = 0
    b_ptr = 0
    output_ptr = 0 
    
    ### This loop is parallelized in triton ###
    for pid in PIDS:
        
        ### Advance to the block of values we are processing ###
        block_start = pid * BLOCK_SIZE

        ### Compute Block Indexes ###
        block_indexes = block_start + torch.arange(BLOCK_SIZE)

        ### Create a Mask so we dont index values that arent there, incase our number of elements ###
        ### is not divisibly by the BLOCK_SIZE ###
        mask = (block_indexes < n_elements)

        ### Only keep the valid indexes in block_indexes ###
        block_indexes = block_indexes[mask]

        ### Advance our "pointers" ###
        a_ptr_offset = a_ptr + block_indexes
        b_ptr_offset = b_ptr + block_indexes
        output_ptr_offset = output_ptr + block_indexes

        ### Perform the Op ###
        output[output_ptr_offset] = a[a_ptr_offset] + b[b_ptr_offset]

def blocked_vector_add_pseudocode_wrapper(a, b):
    """
    Our grid now will be a single PID per block processed rather than 
    per item processed! So we need to pick some reasonable block size
    and just go with it
    """

    ### Let each block be 1024 elements ###
    BLOCK_SIZE = 1024

    ### Simple ceiling division to see how many blocks we need to cover our vector ###
    n_elements = len(a)
    grid = (math.ceil(n_elements / BLOCK_SIZE), )

    output = torch.empty_like(a)

    blocked_vector_add_pseudocode(a, b, output, grid, n_elements, BLOCK_SIZE)

    return output


@triton.jit
def blocked_vector_add(a_ptr, 
                       b_ptr, 
                       output_ptr,
                       n_elements, 
                       BLOCK_SIZE: tl.constexpr): # block size is a constant so we indicate this here as a type hint
            
    """
    Now we repeat our block vector add in Triton
    """

    ### Which block are we processing? ###
    pid = tl.program_id(axis=0)

    ### Get the starting point ###
    block_start = pid * BLOCK_SIZE

    ### Compute Offsets (creates a vector of indexes) ###
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    ### Create Mask ###
    mask = (offsets < n_elements)

    ### Load all the valid (unmasked) data ###
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)

    ### Perform Operation ###
    output = a + b

    ### Write output back (with the mask so we dont save invalid positions) ###
    tl.store(output_ptr + offsets, output, mask=mask)

def blocked_vector_add_wrapper(a, b):

    """
    We now actually perform the vector sum here! This is how we typically call
    and use Triton Kernels
    """

    ### Create Storage for our output ###
    output = torch.empty_like(a)

    ### Create the Grid (we use a lambda function so triton can read the variable "BLOCK_SIZE" from the 
    ### function arguments. We dont have to do this but there are some benefits such as easy autotuning which 
    ### we will see later! ###
    n_elements = len(a)
    grid = lambda args: (triton.cdiv(n_elements, args['BLOCK_SIZE']), )

    ### Run Kernel ###
    ### We can directly pass in a, b, output as PyTorch Tensors and Triton will automatically ###
    ### extract the GPU pointers for these. We could also just pass pointers ###
    ### either will work! ###

    blocked_vector_add[grid](a, b, output, n_elements, BLOCK_SIZE=1024)
    
    return output


def benchmark_torch(a, b, iters=100):

    # Warmup
    for _ in range(10):
        _ = a + b
    torch.cuda.synchronize()

    start = time.time()
    for _ in range(iters):
        out = a + b
    torch.cuda.synchronize()
    end = time.time()

    return (end - start) / iters


def benchmark_triton(a, b, iters=100):

    # Warmup
    for _ in range(10):
        _ = blocked_vector_add_wrapper(a, b)
    torch.cuda.synchronize()

    start = time.time()
    for _ in range(iters):
        out = blocked_vector_add_wrapper(a, b)
    torch.cuda.synchronize()
    end = time.time()

    return (end - start) / iters


if __name__ == "__main__":

    ### Correctness Check ###
    a = torch.randn(1000).to("cuda")
    b = torch.randn(1000).to("cuda")
    
    psuedo_output = blocked_vector_add_pseudocode_wrapper(a,b)
    triton_output = blocked_vector_add_wrapper(a,b)
    torch_output = a + b

    assert torch.allclose(psuedo_output, triton_output)
    assert torch.allclose(triton_output, torch_output)

    print("Triton Matches Torch!!")

    ### Speed Check ###
    N = 50_000_000
    a = torch.randn(N, device="cuda")
    b = torch.randn(N, device="cuda")

    # Benchmark
    torch_time = benchmark_torch(a, b)
    triton_time = benchmark_triton(a, b)

    print(f"Torch time per iter  : {torch_time * 1e3:.3f} ms")
    print(f"Triton time per iter : {triton_time * 1e3:.3f} ms")
