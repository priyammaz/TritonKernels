"""
FlashAttention2 Kernel (Online-Softmax Applied to Attention)

We will be implementing here the Forward pass of the Flash Attention kernel. 
Once you understand this, you can go study the backward pass here!
https://github.com/priyammaz/MyTorch/blob/main/mytorch/nn/functional/fused_ops/flash_attention.py

Problem With (Multihead) Attention:

softmax(Q @ K.T / d) @ V

Q is a [B x H x N x E] matrix
K is a [B x H x N x E] matrix and K.T will then be [B x H x E x N]

This means the final matmul of our Q @ K.T is 

[B x H x N x E] @ [B x H x E x N] wich gives a final [B x H x N x N] tensor!

If N is large (lets say N is 1000) then we have a 1 Million elements 
per sample in the batch!!!

This is a huge memory cost as Attention is an O(N^2) operation. Flash 
attention is a rewrite of this exact same attention mechanism. 

### WHAT WE WANT: The final output of attention!
### WHAT WE DONT CARE ABOUT: The intermediate (massive) tensors in attention!

So this is the perks of kernel fusion. We can rewrite the forward pass
so our GPU never explicitly stores the massive temporary output, and we 
only work towards computing the final result!

### PREREQ!
You should feel comfortable with online softmax, which we saw in 5_online_softmax.py
as we will use the same principles here!

You should also understand attention already! 
"""

import torch
import math
import triton
import triton.language as tl

def naive_attention(Q, K, V, is_causal=False):
    """
    Naive scaled dot-product attention with heads pre-split

    Args:
        Q, K, V: (B, H, T, Dh)
        is_causal: if True, apply causal masking (each position can only attend to earlier positions)

    Returns:
        output: (B, H, T, Dh)
        attn_weights: (B, H, T, T)
    """
    B, H, T, Dh = Q.shape

    # (B, H, T, T)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Dh)

    if is_causal:
        # Create causal mask: upper triangular matrix of True values
        # Position i can only attend to positions <= i
        # 
        # For T=4, causal_mask looks like:
        # [[False,  True,  True,  True],
        #  [False, False,  True,  True],
        #  [False, False, False,  True],
        #  [False, False, False, False]]
        #
        # We want to mask out (set to -inf) the True positions
        causal_mask = torch.triu(torch.ones(T, T, device=Q.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal_mask, float("-inf"))

    attn_weights = torch.softmax(scores, dim=-1)

    # (B, H, T, Dh)
    output = torch.matmul(attn_weights, V)

    return output 

def flash_attention_forward(
    Q,  # (batch, num_heads, seq_len, head_dim)
    K,  # (batch, num_kv_heads, seq_len, head_dim)
    V,  # (batch, num_kv_heads, seq_len, head_dim)
    softmax_scale=None,
    is_causal=False,
    BLOCK_SIZE_Q=64,
    BLOCK_SIZE_KV=64
):
    
    ### Get the shape of the data we are working with
    batch_size, num_heads, seq_len, head_dim = Q.shape
    
    ### use default softmax scaling if not provided ###
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)

    ### In the original softmax implementation we use e^(x), but for mixed-precision stability ###
    ### and efficiency we will instead use 2^(x). But this will obviously give a different value ###
    ### So we need to preadjust our values before we compute our 2^(x). ###
    ### recall your log rules! ###
    ### a^(log_a(b)) = b, so taking a=2 and b=e we get:
    ### 2^(log_2(e)) = e
    ### now log_2(e) = log_e(e) / log_e(2) = ln(e) / ln(2) = 1 / ln(2)
    ### Thus we can write that e = 2 ^ (1 / ln(2))
    ### and raising both sides by an exponential:
    ### e^x = 2^(x/ln(2))
    ### Thus, if we predivide all our values by ln(2), then when we do 2^(x) thats the same as doing e^(x)!
    INV_LN2 = 1.442695040888963

    ### Just adjust our softmax scale with this as we will be using it in a bit! 
    softmax_scale *= INV_LN2

    ### Prescale our Queries ([Q @ K.T] * softmax_scale is the same as [(Q*softmax_scale) @ K.T])###
    Q = Q * softmax_scale

    # ### Now remember, the softmax we are doing is applied to our attention  matrix ###
    # ### which will be in the shape (batch, num_heads, seq_len, seq_len) and we do this ###
    # ### softmax along the last dimension (so for every seq_len x seq_len matrix in this tensor ### 
    # ### we do the softmax along the rows of this matrix) ###
    # ### In our online softmax, we wanted to compute this as we go rather than all at once, and this was ##
    # ### done by tracking some running statistics: 
    # ###     M: the running max
    # ###     L: the running exp sum
    # ### We need to track this for every sample in the batch, for every head of attention, and for every row of the 
    # ### final attention matrix! So lets initialize them  here!  
    # ### As we compute them in the inner loop which you see later, we will store that information here
    # M = torch.full((batch_size, num_heads, seq_len), float('-inf'), 
    #                device=Q.device, dtype=torch.float32)  # running max
    # L = torch.zeros((batch_size, num_heads, seq_len), 
    #                 device=Q.device, dtype=torch.float32)  # running sum
    
    ### Also our output of attention has the same shape tensor as the input. If the output of our softmax
    ### is (batch, num_heads, seq_len, seq_len) and we matmul this with our values tensor which has the 
    ### shape (batch, num_heads, seq_len, embed_dim), we will have a final tensor of
    ### (batch, num_heads, seq_len, embed_dim), which is the same shape as our original inputs!
    ### Lets create an empty tensor and we will populate it as we go!
    O = torch.zeros_like(Q)

    ### Our Triton kernel will work as follows:
    ###     Parallelism over the Batch and Head dimensions (each will be processed by a different PID)
    ###     Loop over the (seq_len, seq_len) dimension in chunks of BLOCK_SIZE_Q and
    ###     BLOCK_SIZE_KV. This allows us to avoid
    ###     loading a massive N^2 tensor into our GPU memory, and we will compute as we go

    ### If each attention tensor looked like:

    ### ->->->->->->->->-> # direction we are processing in our loops
    ### A00 A01 A02 A03
    ### A10 A11 A12 A13 
    ### A20 A21 A22 A23
    ### A30 A31 A32 A33 
    
    ### And our BLOCK_SIZE_Q was 2 and BLOCK_SIZE_KV was 2
    
    ### The first block of Queries would be the first 2 rows of our attention matrix:
    ### A00 A01 A02 A03
    ### A10 A11 A12 A13 

    ### And then we process the KV also in blocks of 2, so
    ### The loop would first process:

    ### A00 A01
    ### A10 A11 

    ### And then process

    ### A02 A03
    ### A12 A13 

    ### Another block of Q would process the rows
    ### A20 A21 A22 A23
    ### A30 A31 A32 A33 

    ### And we would similarly loop through those again in chunks of 2 in the KV dimensions!

    ### and as we process each chunk of the attention matrix we store the running statistics!

    ### Loops over Batch and Head 
    for b in range(batch_size):
        for h in range(num_heads):
            
            ### Process Queries in Blocks ###
            for q_start in range(0, seq_len, BLOCK_SIZE_Q):

                ### make sure we only grab upto the valid Q, handled by masking in triton 
                ### but we make it simple with something called a block pointer later!
                q_end = min(q_start + BLOCK_SIZE_Q, seq_len)

                ### Store indexes of Q being processed for later
                q_indices = torch.arange(q_start, q_end, device=Q.device)

                ### Load block of queries ###
                Q_block = Q[b, h, q_start:q_end, :] # [block_q, head_dim]

                ### Initialize the running statistics for this block ###
                ### Running max initialized with -inf ###
                m_i = torch.full((q_end - q_start,), float('-inf'), device=Q.device, dtype=torch.float32)

                ### Running sum for our denominator (sum e^x) ###
                ### We initialize with 1, exponentiate in a bit and so e^1 is 0, the start of our sum ###
                l_i = torch.ones((q_end - q_start,), device=Q.device, dtype=torch.float32)
                
                ### Initialize the output for this block ###
                O_block = torch.zeros((q_end - q_start, head_dim), device=Q.device, dtype=torch.float32)

                ###############################################################
                ### Now for the next complexity! Causality vs Non Causality ###
                ###############################################################
                ### If we are doing causal attention, the blocks on the diagonal contain values that contain a transition. ###
                ### for example lets say we look at the top left block, and lets say each block is 4x4 ###

                ### SINGLE BLOCK BEING PROCESSED
                ### [qk_00, qk_01, qk_02, qk_03]
                ### [qk_10, qk_11, qk_12, qk_13]
                ### [qk_20, qk_21, qk_22, qk_23]
                ### [qk_30, qk_31, qk_32, qk_33]

                ### The issue is we are processing entire blocks at a time, but the elements of this block are not all valid. ###
                ### If we are causal, qk_01 for example, means query at time 0 is attending to a key in a future time 1. ###
                ### This breaks causality. So in the previous part, we already computed all the blocks upto the diagonal (if causal) ###

                ### Lets look at it at the block level, remember each block has 4x4 elements inside them:

                ### ALL THE BLOCKS THAT COVER THE ENTIRE ATTENTION MATRIX
                ### [B_00, B_01, B_02, B_03]
                ### [B_10, B_11, B_12, B_13]
                ### [B_20, B_21, B_22, B_23]
                ### [B_30, B_31, B_32, B_33]

                ### An in this case our B_00 block is the top left block from above.

                ### In the first stage (if causal) we can directly compute everything in 
                ### B_10, B_20, B_21, B_30, B_31, B_32

                ### Because its guaranteed that every query in that block is attending to keys that are before it
                ### But along our diagonal B_00, B_11, B_22, B_33, we have a transition inside the block
                
                ### Again for B_00 we had 

                ### [qk_00, qk_01, qk_02, qk_03]
                ### [qk_10, qk_11, qk_12, qk_13]
                ### [qk_20, qk_21, qk_22, qk_23]
                ### [qk_30, qk_31, qk_32, qk_33]

                ### We need to compute the diagonal and this is easiest if we just do it separately and then 
                ### make sure to mask out the top triangle portion so we get 

                ### [qk_00, -inf , -inf , -inf ]
                ### [qk_10, qk_11, -inf , -inf ]
                ### [qk_20, qk_21, qk_22, -inf ]
                ### [qk_30, qk_31, qk_32, qk_33]

                ### The reason we are taking care here is for efficiency. We know that blocks like B_01, B_02, B_03
                ### B_12, B_13, B_23 are blocks that ONLY CONTAIN NON-CAUSAL POSITIONS!. So if we are processing causal
                ### attention, then it would be a waste to actually do anything in these blocks, and so we ignore them!

                if is_causal:

                    ### Process prediagonal blocks ###
                    kv_start_prediag = 0
                    kv_end_prediag = q_start

                    ### Process diagonal blocks separately (as they need to be tril masked) ###
                    kv_start_diag = q_start
                    kv_end_diag = q_end

                    ### Store the kv ranges we process ###
                    kv_ranges = [
                        (kv_start_prediag, kv_end_prediag, "prediagonal"),
                        (kv_start_diag, kv_end_diag, "diag")
                    ]

                else:

                    ### If we are not doing causal attention, then we just process the full sequence length and ###
                    ### dont care about masking. 
                    kv_ranges = [(0, seq_len, 'full')]

                ### Now lets process our KV ranges in blocks of BLOCK_SIZE_KV (looping along the row dimension in chunks)

                for kv_start_range, kv_end_range, pass_type in kv_ranges:

                    ### process KB in blocks with this range ###
                    for kv_start in range(kv_start_range, kv_end_range, BLOCK_SIZE_KV):

                        ### Make sure to only index upto the KV that we care about!
                        kv_end = min(kv_start + BLOCK_SIZE_KV, kv_end_range)

                        ### Store the KV indices we are processing right now
                        kv_indices = torch.arange(kv_start, kv_end, device=Q.device)
                        
                        ### Load K and V blocks
                        K_block = K[b, h, kv_start:kv_end, :]  # [block_kv, head_dim]
                        V_block = V[b, h, kv_start:kv_end, :]  # [block_kv, head_dim]

                        ### Compute attention scores: Q @ K^T
                        ### This gives us a small chunk of the final attention matrix now!
                        QK_block = Q_block @ K_block.T  # [block_q, block_kv]

                        ### Apply causal masking on diagonal blocks (if that is the pass type we are using)
                        if pass_type == "diag":
                            causal_mask = q_indices[:, None] >= kv_indices[None, :]
                            QK_block = QK_block.masked_fill(~causal_mask, float('-inf'))

                        ### Apply padding mask (if needed) so we dont access token indexes longer than our sequence 
                        ### This is technically already handled by our kv_end and min from earlier, but I am trying
                        ### to keep it close to the triton code coming up later!
                        kv_padding_mask = kv_indices < seq_len
                        QK_block = QK_block.masked_fill(
                            ~kv_padding_mask[None, :], float('-inf')
                        )


                        ######################
                        ### ONLINE SOFTMAX ###
                        ######################

                        ### Now that Magic! At this point we have processed a single chunk along the KV dimension 
                        ### (along the vector we want to do softmax over). We need to do our online softmax algorithm
                        ### to do our softmax while saving memory!

                        ### Update our running maximum (per row)
                        m_ij = torch.maximum(m_i, QK_block.max(dim=1).values)

                        ### Subtract the max for numerical stability ###
                        QK_block = QK_block - m_ij[:, None]

                        ### Compute exp2 (instead of exp for stability) as we already adjusted for this ealier
                        P_block = torch.exp2(QK_block).to(Q_block.dtype)
                         
                        ### Compute the sum of the rows for this block ###
                        l_ij = P_block.sum(dim=1)

                        ### Correction factor from the previous block so we can do an online softmax ###
                        alpha = torch.exp2(m_i - m_ij)

                        ### Apply the correction factor ###
                        ### Remember that the correction factor formula from online softmax is
                        ### dᵢ₊₁ = dᵢ * exp(mᵢ - mᵢ₊₁) + exp(xᵢ₊₁ - mᵢ₊₁)
                        ### but in our case the extra exp(xᵢ₊₁ - mᵢ₊₁) term is assuming we are processing
                        ### one element at a time, but we are doing a block at a time, so we just 
                        ### add on all the elements of the block at once (hense the line l_ij = P_block.sum(dim=1))
                        ### Take a look at online_blocked_softmax_kernel in 5_online_softmax.py for more details
                        l_i = l_i * alpha + l_ij


                        ### Now for the new step. Our output is P @ V where P is the final softmax output and V is the 
                        ### value vectors that correspond to those query positions. But as of now we dont have the full 
                        ### softmax output. So as we compute our output, the softmax values will update, so we need to 
                        ### Similarly correct our output values as we loop through the keys/values

                        ### WHAT IS BEING ACCUMULATED ###
                        ###         (4 x 5)                   (5 x 4)                  (4 x 4)                                     (4 x 4)               (4 x 5)                     (4 x 5)
                        ### [Q00 Q01 Q02 Q03 Q04]   @   [K00 K10 K20 K30]   =   [A00 A01 A02 A03]                            [P00 P01 P02 P03]  @  [V00 V01 V02 V03 V04]       [O00 O01 O02 O03 O04]
                        ### [Q10 Q11 Q12 Q13 Q14]       [K01 K11 K21 K31]       [A10 A11 A12 A13]   ==> SOFTMAX(dim=1) ==>   [P10 P11 P12 P13]     [V10 V11 V12 V13 V14] =     [O10 O11 O12 O13 O14]
                        ### [Q20 Q21 Q22 Q23 Q24]       [K02 K12 K22 K32]       [A20 A21 A22 A23]                            [P20 P21 P22 P23]     [V20 V21 V22 V23 V24]       [O20 O21 O22 O23 O24]
                        ### [Q30 Q31 Q32 Q33 Q34]       [K03 K13 K23 K33]       [A30 A31 A32 A33]                            [P30 P31 P32 P33]     [V30 V31 V32 V33 V34]       [O30 O31 O32 O33 O34]
                        ###                             [K04 K14 K24 K34]

                        ### But we are doing this in chunks, so if we are doing BLOCK_SIZE_Q = 2 and BLOCK_SIZE_KV = 2, then we are only
                        ### first looking at the first two queries and first two key/value indices, which look like:

                        ###         (2 x 5)              (5 x 2)         (2 x 2)                                     (2 x 2)               (2 x 5)                   (2 x 5)
                        ### [Q00 Q01 Q02 Q03 Q04]   @   [K00 K10]   =   [A00 A01]                                   [P00 P01]  @  [V00 V01 V02 V03 V04]       [O00 O01 O02 O03 O04]
                        ### [Q10 Q11 Q12 Q13 Q14]       [K01 K11]       [A10 A11]   ==> ONLINE_SOFTMAX(dim=1) ==>   [P10 P11]     [V10 V11 V12 V13 V14] =     [O10 O11 O12 O13 O14]
                        ###                             [K02 K12]                                                                               
                        ###                             [K03 K13]                                                                         
                        ###                             [K04 K14]

                        ### But notice a few things:
                        ### 1. We only have the probabilties of the first two positions for K/V, so our final output is only a partial accumulation of the first two Value vectors, we need the other ones
                        ### 2. We also need to do online softmax to update the probabilties as we go

                        ### SO we keep going, lets get the second loop of KV for this block of queries 

                        ###         (2 x 5)              (5 x 2)         (2 x 2)                                     (4 x 4)               (4 x 5)                     (4 x 5)
                        ### [Q00 Q01 Q02 Q03 Q04]   @   [K20 K30]   =   [A02 A03]                                   [P02 P03]  @  [V20 V21 V22 V23 V24]       [O00 O01 O02 O03 O04]
                        ### [Q10 Q11 Q12 Q13 Q14]       [K21 K31]       [A12 A13]   ==> ONLINE_SOFTMAX(dim=1) ==>   [P12 P13]     [V30 V31 V32 V33 V34] =     [O10 O11 O12 O13 O14]
                        ###                             [K22 K32]                                                                                
                        ###                             [K23 K33]                                                                              
                        ###                             [K24 K34]
                                
                        ### SO now you see we have two outputs. The first output from the first step of KV and the second output from the second step of KV. There are a few problems:
                        ### Technically, we want the sum of these two outputs for the final output, as its simple matmul accumulate. But also the first output could be wrong, it was computed
                        ### based on probabilities from a softmax that may be using the wrong max (the running stats may have updated in this second loop for the P00, P01, P10, P11 could be wrong)
                        ### So lets go over the derivation of how we handle this!

                        ### ONE TRICK. SOFTMAX requires us to divide by the reduction sum as follows: Σⱼ exp(xⱼ - m). This is just a constant (per row of softmax), so we leave it for the end 
                        ### for simplicity. So we will only do the numerator of softmax and then do the denominator at the every end

                        ### SIMPLE DERIVATION OF WHAT WE HAVE DONE SO FAR (SHOULD FEEL FAMILIAR AS ITS JUST ONLINE SOFTMAX) ###

                            # First block of keys/values (indices 0 to k1)
                            # S1 = Q @ K1^T  # attention scores for first block
                            # m1 = max(S1)   # max of scores so far
                            # P1 = exp(S1 - m1)  # shifted exp
                            # l1 = sum(P1)   # sum of exps
                            # O1 = P1 @ V1   # output contribution
                            
                            # Second block of keys/values (indices k1 to k2)
                            # S2 = Q @ K2^T
                            # ```

                            # Now we need to combine S1 and S2 to get the correct softmax over **both** blocks.

                            # ### The Challenge

                            # The softmax denominator should be:
                            # ```
                            # sum(exp(S1 - m_global)) + sum(exp(S2 - m_global))

                            # where m_global = max(S1, S2) is the max over all scores seen so far.

                            # But we already computed the following with the old max value m1
                            
                            # P1 = exp(S1 - m1)
                            # l1 = sum(P1)

                            # Like we did in online softmax we need to adjust these

                            # Let our new max m2 = max(m1, max(S2))  

                            # The corrected sum from block 1 should be:

                            # sum(exp(S1 - m2)) 
                            #     = sum(exp(S1 - m1 - (m2 - m1)))
                            #     = sum(exp(S1 - m1) * exp(-(m2 - m1)))
                            #     = exp(m1 - m2) * sum(exp(S1 - m1))
                            #     = exp(m1 - m2) * l1

                            # The above should feel familiary we did this in the online_softmax derivation

                            # And so we call our correction factor alpha = exp(m1 - m2)
                            
                            # and the corrected denominator is l2 = alpha * l1 + sum(exp(S2 - m2))

                        ### ADJUSTING OUR OUTPUTS IN THE SAME WAY ###
                        
                            # Similarly the output from block 1 would have been

                            # O1 = P1 @ V1 = exp(S1 - m1) @ V1

                            # But once we move onto the second block, it should have used the global max m2 instead
                            # so we apply correction

                            # O1_corrected = exp(S1 - m2) @ V1
                            #     = exp(S1 - m1 - (m2 - m1)) @ V1
                            #     = exp(m1 - m2) * exp(S1 - m1) @ V1
                            #     = alpha * O1

                            # And the new contribution from block 2 is 
                            # O2_contrib = exp(S2 - m2) @ V2

                            # So the total output (not normalized yet as we will divide by the constant sum in softmax later) is 
                            # O_new = O1_corrected + O2_contrib
                            #     = alpha * O1 + exp(S2 - m2) @ V2
                            #     = alpha * O_old + P2 @ V2
                        
                        ### Correction upto the current accumulation (O_old)
                        O_block = O_block * alpha[:, None]

                        ### Add the new accumulation ###
                        O_block = O_block + P_block @ V_block

                        ### Update running max for the next iteration 
                        m_i = m_ij

                ### After all Key/Values are processed lets go ahead and normalize by the final sum ###
                ### l_i is our per row accumulated sum that we have been 
                O_block = O_block / (l_i[:, None] + 1e-6)

                ### Store the results ###
                O[b, h, q_start:q_end, :] = O_block
        
    return O

@triton.jit
def _attn_fwd_inner(
    O_block,
    l_i,
    m_i,
    Q_block,
    K_block_ptr,
    V_block_ptr,
    block_index_q,
    BLOCK_SIZE_Q,
    BLOCK_SIZE_KV,
    PASS_TYPE: tl.constexpr, # 0: pre_diag, 1: diag, 2: full
    offs_q: tl.constexpr,
    offs_kv: tl.constexpr,
    SEQ_LEN,
    DTYPE_FLAG: tl.constexpr, # 0 for float32, 1 for float16
):      
    """
    The inner loop of the forward flash attention method grabs a chunk of queries
    and loops through all the Keys/Values also in chunks, using online-softmax as we go
    """

    if PASS_TYPE == 0:
        ### When in Causal Mode we need to first compute the prediagonal:
        ### I want indexes (for my K,V) that are upto the ###
        ### index of my queries. This way my queries only attend to ###
        ### Keys and values that are before it. 

        ### This applies to all K,V before the diagonal. These are all blocks 
        ### of queries, as long as we are before the diagonal i know for sure 
        ### that every KV must be less that my query. Lets say we have the 
        ### following output from our blockes QKT

        ### [qk00 qk01 qk02 qk03]
        ### [qk10 qk11 qk12 qk13]
        ### [qk20 qk21 qk22 qk23]
        ### [qk30 qk31 qk32 qk33]

        ### And each qk00 is a block of values (lets say 3 x 3). I know for sure
        ### that every value in qk10, qk20, qk21, qk30, qk31, qk32 dont break any 
        ### causality. every value in those specific blocks that queries are ###
        ### looking at k/v that are <= in index 

        lo, hi = 0, block_index_q * BLOCK_SIZE_Q
    elif PASS_TYPE == 1:

        ### When in causal we need to also process the diagonal but they have a transition
        ### Lets say we grab the top left corner (qk00) and each block is processing 
        ### 3 queries and 3 keys. In our output:

        ### [x00 x01 x02]
        ### [x10 x11 x12]
        ### [x20 x21 x22]

        ### x01 x02 x12 are invalid positions as that would mean the query vector
        ### is attending to a vector after it So we just need to remove these extra 
        ### ones. This is just more post processing!

        ### The block we want is the one at the end of our completely valid Q blocks and the 
        ### next one after. So we essentialyl have a low/high containing onyl the single block 
        ### where Q ends! Its just in that diagonal we have to mask out half that block. 

        lo, hi = block_index_q * BLOCK_SIZE_Q, (block_index_q + 1) * BLOCK_SIZE_Q
        lo = tl.multiple_of(lo, BLOCK_SIZE_Q)
    else:

        ### When not causal we just want to process the entire sequence as we attend to everything. 
        lo, hi = 0, SEQ_LEN

    ### KV pointers are currently pointing at the very start of the Key/Value for this ###
    ### Specific batch and head. In the case of STAGE=1 or ELSE, we just start at 0. We will ###
    ### piece by piece load BLOCK_SIZE_KV sizes of our Keys nad Values and do our ops there ###
    ### but in STAGE=2, we only want to do the ops on the diagonal values, so we need to advance ###
    ### our index to there ###

    K_block_ptr = tl.advance(K_block_ptr, (0, lo)) # Keys are transposed so SEQ dim is second
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))

    ### Loop over our Ks and Vs ###
    for start_kv in range(lo, hi, BLOCK_SIZE_KV):

        ### Let the compiler know that start_n is a multiple of BLOCK_N ###
        start_kv = tl.multiple_of(start_kv, BLOCK_SIZE_KV)

        ### Compute a mask so we dont access token indexes longer than our sequence 
        kv_indices = start_kv + offs_kv
        kv_padding_mask = kv_indices < SEQ_LEN

        ### Load our K and V Blocks ###
        K_block = tl.load(K_block_ptr, boundary_check=(1,), padding_option="zero")
        V_block = tl.load(V_block_ptr, boundary_check=(0,), padding_option="zero")

        ### We have a (Q_BLOCK_SIZE x E) and (E x KV_BLOCK_SIZE) matricies ###
        ### we can just use dot to do our dot product computation ### 
        QK_block = tl.dot(Q_block, K_block)

        if PASS_TYPE == 1:

            # Post process the diagonal 
            # off_q is the indexes of the queries we are processing
            # offs_kv is the indexes of the keys/values we are processing
            # we can offset our offs_kv for this specific iteration in the loop
            # and do a broadcast check to see for what spots every q is greater than our 
            # k positions.
            # 
            # [0]   [0 1 2 3] -> [True False False False]
            # [1]                [True True  False False]
            # [2]                [True True  True  False]
            # [3]                [True True  True  True]
            # and then we can just fill the False with a large negative number!

            causal_mask = offs_q[:, None] >= kv_indices[None, :]
            mask = causal_mask & kv_padding_mask[None, :]
            QK_block += tl.where(mask, 0, float("-inf"))

        else:

            ### If we are not pass_type==1, then we are either processing pre-diagonal blocks ###
            ### or we are just processing all blocks. In either case, we want to make sure that ###
            ### we mask our any invalid positions in our QK Block and dont have to worry about inside block transitions ###
            QK_block += tl.where(kv_padding_mask[None, :], 0, float("-inf"))

        ### Update our current estimate for the maximum
        m_ij = tl.maximum(m_i, tl.max(QK_block, 1))
        QK_block -= m_ij[:, None]

        ### We subtracted the max (and masked if needed) now we exponentiate ###
        ### But remember we will use exp2 instead of exp for float16 stability ###
        ### and we already adjusted for this earlier! ###
        P_block = tl.math.exp2(QK_block)

        ### Compute the sum of the rows for this block ###
        l_ij = tl.sum(P_block, 1)

        ### Correction factor from the previous block so we can do an online softmax ###
        alpha = tl.math.exp2(m_i - m_ij)

        ### Apply the correction factor ###
        l_i = l_i * alpha + l_ij

        ### Make sure our Dtype matches our flag By default it will be float32 ###
        ### but we need to fast to fp16 incase thats the precision type we have ###
        P_block = P_block.to(tl.float32 if DTYPE_FLAG==0 else tl.float16)

        ### Use our formuala to iteratively update our outputs O_new = PV + O_old * alpha ###
        O_block = O_block * alpha[:, None]
        O_block = tl.dot(P_block, V_block, acc=O_block)
        
        ### Update Estimate for Next Iter ###
        m_i = m_ij

        ### Advance to the next block ###
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_SIZE_KV, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_SIZE_KV))
        
    return O_block, l_i, m_i


@triton.autotune(
    configs=[
            triton.Config(
                {"BLOCK_SIZE_Q": BLOCK_SIZE_Q, "BLOCK_SIZE_KV": BLOCK_SIZE_KV},
                num_stages=num_stages,
                num_warps=num_warps,
            )
            for BLOCK_SIZE_Q in [16, 32, 64, 128]
            for BLOCK_SIZE_KV in [16, 32, 64]
            for num_stages in [2, 3, 4]
            for num_warps in [4, 8, 16]
            if BLOCK_SIZE_KV < BLOCK_SIZE_Q
        ],
    key=["HEAD_DIM"],
)
@triton.jit
def _attn_fwd(
    Q,  
    K, 
    V,
    softmax_scale: tl.constexpr,
    M,  
    O, 
    stride_Q_batch,
    stride_Q_head,
    stride_Q_seq,
    stride_Q_dim,
    stride_K_batch, 
    stride_K_head, 
    stride_K_seq,
    stride_K_dim,
    stride_V_batch, 
    stride_V_head,
    stride_V_seq,
    stride_V_dim,
    stride_O_seq,
    stride_O_dim,
    NUM_HEADS: tl.constexpr,
    SEQ_LEN,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    ATTN_MODE: tl.constexpr, # 0 for non_causal, 1 for causal
    DTYPE_FLAG: tl.constexpr, # 0 for float32, 1 for float16,
):  
    """
    Main forward method for Flash Attention, where for a block of queries
    we iteratively compute attention by looping over blocks of Keys/Values
    """

    ### When we do Q @ K, we use the tl.dot method to do this ###
    ### So the inner product loads a row/column of K and Q into ###
    ### registers for the actual computation where each row has HEAD_DIM elements ###
    ### So although we are chunking our sequence into BLOCK_SIZE_KV, we still need ###
    ### to load the entire embeddings. We want to make sure this isnt too large for ###
    ### efficiency. So we place a restriction here that our BLOCK_SIZE cannot be any ###
    ### larger than our HEAD_DIM. Id rather have more blocks scheduled to do less work ###
    ### than have fewer blocks each processing massive matricies for better GPU utilization ###
    tl.static_assert(BLOCK_SIZE_KV <= HEAD_DIM)

    ### In the original softmax implementation we use e^(x), but for mixed-precision stability ###
    ### and efficiency we will instead use 2^(x). But this will obviously give a different value ###
    ### So we need to preadjust our values before we compute our 2^(x). ###
    ### recall your log rules! ###
    ### a^(log_a(b)) = b, so taking a=2 and b=e we get:
    ### 2^(log_2(e)) = e
    ### now log_2(e) = log_e(e) / log_e(2) = ln(e) / ln(2) = 1 / ln(2)
    ### Thus we can write that e = 2 ^ (1 / ln(2))
    ### and raising both sides by an exponential:
    ### e^x = 2^(x/ln(2))
    ### Thus, if we predivide all our values by ln(2), then when we do 2^(x) thats the same as doing e^(x)!
    INV_LN2: tl.constexpr = 1.442695040888 # approx 1 / ln(2)

    ### We have to multiply by our 1/sqrt(head_dim) too, so just add the additional multipler to here for later! 
    softmax_scale *= INV_LN2

    #### This is the block index of Q that we will process ###
    block_index_q = tl.program_id(0)

    ### Intermediate buffer M is always a float32 for maintaining precision in the backward pass ###
    M = tl.cast(M, tl.pointer_type(tl.float32))
        
    ### our index batch head is just a flattened vector of our batch_size * number of heads ###
    ### this means if we want what batch we are on, we can divide by num heads ###
    ### if we want which head we are on we can use modulo ###
    index_batch_head = tl.program_id(1) 
    index_batch = index_batch_head // NUM_HEADS
    index_head = index_batch_head % NUM_HEADS

    ### Compute our offset of where a particular batch and head starts ###  
    q_offset = (
        index_batch.to(tl.int64) * stride_Q_batch + index_head.to(tl.int64) * stride_Q_head
    )
    kv_offset = (
        index_batch.to(tl.int64) * stride_K_batch + index_head.to(tl.int64) * stride_K_head
    )

    ### Who likes pointer arithmetic? Remember, my Q data is:
    ### Q.shape = (BATCH x HEADS x SEQ_LEN x EMBED_DIM)
    ### Each thread will process a specific BATCH and HEAD as well as a BLOCK of our SEQ_LEN
    ### So I need ot basically do a Q[batch_idx, head_idx, start_q_idx:end_q_idx, :]
    ### To do this with pointer arithmetic it would kind of look like:

    # row_offset = block_index_q * BLOCK_SIZE_Q                ### starting query vector idx in block
    # col_offset = 0                                           ### no column offset as we want the entire embedding vector

    # for i in range(BLOCK_SIZE_Q):                            ### for every query index I want
    #     for j in range(HEAD_DIM):                            ### for the head I am in
    #         ptr = (Q + qkv_offset                            ### We want to start art the right batch/head starting point and then move over by the row/col offset
    #                 + (row_offset + i) * stride_Q_seq
    #                 + (col_offset + j) * stride_Q_dim)
    #         val = tl.load(ptr)

    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset,                      ### Offset to the batch/head we are processing
        shape=(SEQ_LEN, HEAD_DIM),                ### Because we already indexed batch/head the shape that is left is just (SEQ_LEN, HEAD_DIM)
        strides=(stride_Q_seq, stride_Q_dim),     ### What are the strides of the remaining dimensions
        offsets=(block_index_q * BLOCK_SIZE_Q, 0),### Indexes of the Block of queries we are processing
        block_shape=(BLOCK_SIZE_Q, HEAD_DIM),     ### What is the shape our our block of queries?
        order=(1,0)                               ### Memory coalescing. We make our HEAD DIM in contiguous memory addresses for fast access over the embeddings
    )

    V_block_ptr = tl.make_block_ptr(
        base=V + kv_offset,                      
        shape=(SEQ_LEN, HEAD_DIM),                
        strides=(stride_V_seq, stride_V_dim),     
        offsets=(0,0),                            ### When loading values we dont skip anything, as we will for loop over this in a bit
        block_shape=(BLOCK_SIZE_KV, HEAD_DIM),              
        order=(1,0)                               
    )

    ### Switching our strides transposes our matrix. Take for example
    ### [A B C]
    ### [D E F]
    
    ### This has a strides[0] of 3 and strides[1] of 1. Now in memory, its actually
    ### stored as [A B C D E F]
    
    ### So what if we make our stride[0] = 1 and stride[1] = 3?

    ### Starting at A, to get to the next column, we have to move over 3. So A next 
    ### to A you have D. To get to the next row you move over 1, so from A next next
    ### row would be B. And that is exactly the transpose if you keep going!
    K_block_ptr = tl.make_block_ptr(
        base=K + kv_offset,                      
        shape=(HEAD_DIM, SEQ_LEN),                ### Set shape to transpose dimension
        strides=(stride_K_dim, stride_K_seq),     ### invert the stride     
        offsets=(0,0),                           
        block_shape=(HEAD_DIM, BLOCK_SIZE_KV),              
        order=(0,1)                               ### We want contiguous memory along the HEAD_DIM first
    )
    
    O_block_ptr = tl.make_block_ptr(
        base=O + q_offset,                      
        shape=(SEQ_LEN, HEAD_DIM),               
        strides=(stride_O_seq, stride_O_dim),     
        offsets=(block_index_q * BLOCK_SIZE_Q, 0),
        block_shape=(BLOCK_SIZE_Q, HEAD_DIM),     
        order=(1,0)                               
    )

    ### Lets grab offsets to tell us which indexes of Queries we are processing ###
    offs_q = block_index_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)

    ### We also need our offsets for the kv, of how many kv vectors are we processing with 
    ### Every one of our query blocks? 
    offs_kv = tl.arange(0, BLOCK_SIZE_KV)

    ### Intermediate data we store will be in a higher precision for efficiency ###
    ### Running max initialized with -inf ###
    m_i = tl.full(shape=[BLOCK_SIZE_Q], value=float("-inf"), dtype=tl.float32)

    ### Running sum for our denominator (sum e^x) ###
    ### We initialize with 1, exponentiate in a bit and so e^1 is 0, the start of our sum ###
    l_i = tl.full(shape=[BLOCK_SIZE_Q], value=1.0, dtype=tl.float32) 

    ### Accumulation of our final qk^T v for our specific block of queries/keys/values ###
    O_block = tl.zeros([BLOCK_SIZE_Q, HEAD_DIM], dtype=tl.float32)
    
    ### Load our Query Block ###
    ### Now a super cool ability for block pointers. It can automagically check ###
    ### for invalid indexes (like if our Query we are indexing is greater than SEQ_LEN) ###
    ### And it will fill it with the padding option we give it! ###
    Q_block = tl.load(Q_block_ptr, boundary_check=(0,), padding_option="zero")
    
    ### Prescale our Queries here so we dont need to do it in our inner loop later ###
    Q_block *= softmax_scale
    Q_block = Q_block.to(tl.float32 if DTYPE_FLAG == 0 else tl.float16)

    ### If we are causal (ATTN_MODE==1) then we only process the pre-diagonal stuff (pass_type==0) ###
    ### otherwise we process everything (full attention) ###
    pass_type = 0 if ATTN_MODE == 1 else 2
    O_block, l_i, m_i = _attn_fwd_inner(
        O_block, 
        l_i, 
        m_i, 
        Q_block, 
        K_block_ptr, 
        V_block_ptr, 
        block_index_q, 
        BLOCK_SIZE_Q, 
        BLOCK_SIZE_KV, 
        pass_type,
        offs_q, 
        offs_kv, 
        SEQ_LEN,
        DTYPE_FLAG
    )

    ### IF we are causal, we need to separately handle the diagonal ###
    ### so that is taken care of here with pass_type==1 ###
    if ATTN_MODE == 1:

        ### If we are doing causal attention, the blocks on the diagonal contain values that contain a transition. ###
        ### for example lets say we look at the top left block, and lets say each block is 4x4 ###

        ### [qk_00, qk_01, qk_02, qk_03]
        ### [qk_10, qk_11, qk_12, qk_13]
        ### [qk_20, qk_21, qk_22, qk_23]
        ### [qk_30, qk_31, qk_32, qk_33]

        ### The issue is we are processing entire blocks at a time, but the elements of this block are not all valid. ###
        ### If we are causal, qk_01 for example, means query at time 0 is attending to a key in a future time 1. ###
        ### This breaks causality. So in the previous part, we already computed all the blocks upto the diagonal (if causal) ###

        ### Lets look at it at the block level, remember each block has 4x4 elements inside them:

        ### [B_00, B_01, B_02, B_03]
        ### [B_10, B_11, B_12, B_13]
        ### [B_20, B_21, B_22, B_23]
        ### [B_30, B_31, B_32, B_33]

        ### An in this case our B_00 block is the top left block from above.

        ### In the first stage (if causal) we can directly compute everything in 
        ### B_10, B_20, B_21, B_30, B_31, B_32

        ### Because its guaranteed that every query in that block is attending to keys that are before it
        ### But along our diagonal B_00, B_11, B_22, B_33, we have a transition inside the block
        
        ### Again for B_00 we had 

        ### [qk_00, qk_01, qk_02, qk_03]
        ### [qk_10, qk_11, qk_12, qk_13]
        ### [qk_20, qk_21, qk_22, qk_23]
        ### [qk_30, qk_31, qk_32, qk_33]

        ### We need to compute the diagonal and this is easiest if we just do it separately and then 
        ### make sure to mask out the top triangle portion so we get 

        ### [qk_00, -inf , -inf , -inf ]
        ### [qk_10, qk_11, -inf , -inf ]
        ### [qk_20, qk_21, qk_22, -inf ]
        ### [qk_30, qk_31, qk_32, qk_33]

        O_block, l_i, m_i = _attn_fwd_inner(
            O_block, 
            l_i, 
            m_i, 
            Q_block, 
            K_block_ptr, 
            V_block_ptr, 
            block_index_q, 
            BLOCK_SIZE_Q, 
            BLOCK_SIZE_KV, 
            1,
            offs_q, 
            offs_kv, 
            SEQ_LEN,
            DTYPE_FLAG
        )

    ### Store this as we need it for logsumexp in the backward pass ###
    ### this is the main trick we use so in our backward pass we can just ###
    ### use this to quickly recompute our softmax values, without storing ###
    ### a giant N^2 Softmax matrix in memory! ###
    m_i += tl.math.log2(l_i)

    ### We also now have our true sum along each row of attention, we can divide ###
    ### by them to get our actual normalized outputs ###
    O_block = O_block / (l_i[:, None] + 1e-6)
    
    ### Store M w/ a boundary check###
    m_ptrs = M + index_batch_head * SEQ_LEN + offs_q
    q_padding_mask = offs_q < SEQ_LEN
    tl.store(m_ptrs, m_i, mask=q_padding_mask)

    ### Store Q (again with a boundary check) ###
    tl.store(O_block_ptr, O_block.to(O.type.element_ty), boundary_check=(0,))

def fused_sdpa_forward(Q, K, V, 
                       causal=False, 
                       softmax_scale=None):
    
    BATCH_SIZE, NUM_HEADS, SEQ_LEN_Q, HEAD_DIM_Q = Q.shape

    if softmax_scale is None:
        softmax_scale = 1 / HEAD_DIM_Q**0.5

    ### Make sure there is contiguous memory layout ####
    if not Q.is_contiguous():
        Q = Q.contiguous()
    if not K.is_contiguous():
        K = K.contiguous()
    if not V.is_contiguous():
        V = V.contiguous()

    # Create output tensors
    O = torch.empty_like(Q)
    M = torch.empty(BATCH_SIZE, NUM_HEADS, SEQ_LEN_Q, dtype=torch.float32, device=Q.device)
    grid = lambda args: (triton.cdiv(SEQ_LEN_Q, args["BLOCK_SIZE_Q"]), BATCH_SIZE * NUM_HEADS, 1)

    _attn_fwd[grid](
        Q=Q,
        K=K,
        V=V,
        softmax_scale=softmax_scale,
        M=M,
        O=O,
        stride_Q_batch=Q.stride(0),
        stride_Q_head=Q.stride(1),
        stride_Q_seq=Q.stride(2),
        stride_Q_dim=Q.stride(3),
        stride_K_batch=K.stride(0),
        stride_K_head=K.stride(1),
        stride_K_seq=K.stride(2),
        stride_K_dim=K.stride(3),
        stride_V_batch=V.stride(0),
        stride_V_head=V.stride(1),
        stride_V_seq=V.stride(2),
        stride_V_dim=V.stride(3),
        stride_O_seq=O.stride(2),
        stride_O_dim=O.stride(3),
        NUM_HEADS=NUM_HEADS,
        SEQ_LEN=SEQ_LEN_Q,
        HEAD_DIM=HEAD_DIM_Q,
        ATTN_MODE=1 if causal else 0,
        DTYPE_FLAG=0 if Q.dtype == torch.float32 else 1,
    )

    return O

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["SEQ_LEN"],
        x_vals=[256 * i for i in range(1, 17)],
        line_arg="provider",
        line_vals=["torch", "triton", "naive"],
        line_names=["PyTorch SDPA", "Triton", "Naive"],
        styles=[("green","-"),("red","--"), ("blue","--")],
        ylabel="TFLOPS",
        plot_name="flash_attn_fwd",
        args={"mode": "fwd", "causal": True, "dtype": "float16"},
    )
)
def bench_sdpa_forward(SEQ_LEN, mode, provider, causal, dtype, device="cuda"):
    
    BATCH, N_HEADS, HEAD_DIM = 4, 32, 64
    sm_scale = 1.0 / HEAD_DIM**0.5
    is_fp16 = dtype=="float16"
    
    q = torch.randn(BATCH, N_HEADS, SEQ_LEN, HEAD_DIM, 
                        dtype=torch.float16 if is_fp16 else torch.float32,
                        device=device, requires_grad=(mode=="bwd"))
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    
    if provider=="torch":
        fn = lambda: torch.nn.functional.scaled_dot_product_attention(q,k,v,is_causal=causal)
    elif provider == "triton":
        fn = lambda: fused_sdpa_forward(q, k, v, causal, softmax_scale=None)
    elif provider=="naive":
        fn = lambda: naive_attention(q, k, v, causal)
    
    ms = triton.testing.do_bench(fn, warmup=25, rep=100)
    
    flops_per_matmul = 2.0 * BATCH * N_HEADS * SEQ_LEN * SEQ_LEN * HEAD_DIM
    total_flops = 2 * flops_per_matmul
    tflops = total_flops / (ms * 1e-3) / 1e12
    return tflops

if __name__ == "__main__":

    ### Check the Psuedocode (non-causal) ### 
    Q = torch.randn(2,4,32,64, device="cuda", dtype=torch.float16)
    K = torch.randn(2,4,32,64, device="cuda", dtype=torch.float16)
    V = torch.randn(2,4,32,64, device="cuda", dtype=torch.float16)

    output_naive = naive_attention(Q, K, V)
    output_flash = flash_attention_forward(Q, K, V)
    assert torch.allclose(output_naive, output_flash, rtol=1e-2, atol=1e-2)

    ### Check the Psuedocode (causal) ### 
    output_naive = naive_attention(Q, K, V, is_causal=True)
    output_flash = flash_attention_forward(Q, K, V, is_causal=True)
    assert torch.allclose(output_naive, output_flash, rtol=1e-2, atol=1e-2)
    
    ### Check Flash Attention (non-causal) ###
    output_naive = naive_attention(Q, K, V)
    output_triton = fused_sdpa_forward(Q, K, V)
    assert torch.allclose(output_naive, output_triton, rtol=1e-2, atol=1e-2)

    ### Check Flash Attention (causal) ###
    output_naive = naive_attention(Q, K, V, is_causal=True)
    output_triton = fused_sdpa_forward(Q, K, V, causal=True)
    assert torch.allclose(output_naive, output_triton, rtol=1e-2, atol=1e-2)

    bench_sdpa_forward.run(show_plots=True)

    


                
 



    




    


    








