"""
Sum of Two Vectors a + b

"""
import time
import torch 
import triton
import triton.language as tl

def vector_add_pseudocode(a, # first vectors
                          b, # second vector
                          output, # output vector to store results
                          grid): # Index grid that we will use to process our vectors
    """
    Simple vector sum where every PID will process a single sum
    In triton this will all happen in parallel, but we do it
    with a for loop!
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

        ### Advance our "pointers" ###
        a_ptr_offset = a_ptr + pid
        b_ptr_offset = b_ptr + pid
        output_ptr_offset = output_ptr + pid

        ### Perform the Op ###
        output[output_ptr_offset] = a[a_ptr_offset] + b[b_ptr_offset]

def vector_add_pseudocode_wrapper(a, b):
    """
    Triton works on something known as the GRID. The Grid is just indexes that
    identify how we are organizing our computation. For a vector sum, we only need
    a 1D grid, where each value along that 1D grid is the index along the vector. 

    Later we will need more fancy higher dim grids, but for now lets keep it simple!
    """

    n_elements = len(a)
    grid = (n_elements, )

    output = torch.empty_like(a)

    vector_add_pseudocode(a, b, output, grid)

    return output


@triton.jit
def vector_add(a_ptr, 
               b_ptr, 
               output_ptr):
    
    """
    This is the general structure of a Triton Kernel. The function arguments take in pointers
    to data we want to process and the pointer to the output in which we want to store the result

    There are other things we pass in here as well later on but for now we keep it simple!
    """
    
    ### PID is Provided by Triton. Because our sum is along a single dimension vector ###
    ### we only need a single dimension of indexes for our PID. Later for higher dim tensors, ###
    ### we will be introducing more of these! For now though its simple, just an index for every ###
    ### value in our vector ###

    ### How many PIDs we have is determined by the Grid Size that we define later! But in this simple
    ### case we should have one per element in our vector ###
    pid = tl.program_id(axis=0)

    ### a_ptr, b_ptr, output_ptr indicates the starting address of wherever the data for a, b, and output are stored ###
    ### But we are currently processing the pid'th element in this vector. So we need to simply offset our pointers ###
    ### by our pid. This will happen in parallel for all PIDs that we process ###
    a_ptr_offset = a_ptr + pid
    b_ptr_offset = b_ptr + pid
    output_ptr_offset = output_ptr + pid

    ### Load our a and b (directly loads values into SRAM) ###
    a = tl.load(a_ptr_offset)
    b = tl.load(b_ptr_offset)

    ### Perform Operation ###
    output = a + b

    ### Write output back ###
    tl.store(output_ptr_offset, output)

def vector_add_wrapper(a, b):

    """
    We now actually perform the vector sum here! This is how we typically call
    and use Triton Kernels
    """

    ### Create Storage for our output ###
    output = torch.empty_like(a)

    ### Create the Grid ###
    n_elements = len(a)
    grid = (n_elements, )

    ### Run Kernel ###
    ### We can directly pass in a, b, output as PyTorch Tensors and Triton will automatically ###
    ### extract the GPU pointers for these. We could also just pass pointers ###
    ### either will work! ###

    vector_add[grid](a, b, output)
    
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
        _ = vector_add_wrapper(a, b)
    torch.cuda.synchronize()

    start = time.time()
    for _ in range(iters):
        out = vector_add_wrapper(a, b)
    torch.cuda.synchronize()
    end = time.time()

    return (end - start) / iters


if __name__ == "__main__":

    ### Correctness Check ###
    a = torch.randn(1000).to("cuda")
    b = torch.randn(1000).to("cuda")
    
    psuedo_output = vector_add_pseudocode_wrapper(a,b)
    triton_output = vector_add_wrapper(a,b)
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
