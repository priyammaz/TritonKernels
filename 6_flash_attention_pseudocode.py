import torch
import math 

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

    ### Now remember, the softmax we are doing is applied to our attention  matrix ###
    ### which will be in the shape (batch, num_heads, seq_len, seq_len) and we do this ###
    ### softmax along the last dimension (so for every seq_len x seq_len matrix in this tensor ### 
    ### we do the softmax along the rows of this matrix) ###
    ### In our online softmax, we wanted to compute this as we go rather than all at once, and this was ##
    ### done by tracking some running statistics: 
    ###     M: the running max
    ###     L: the running exp sum
    ### We need to track this for every sample in the batch, for every head of attention, and for every row of the 
    ### final attention matrix! So lets initialize them  here!  
    ### As we compute them in the inner loop which you see later, we will store that information here
    M = torch.full((batch_size, num_heads, seq_len), float('-inf'), 
                   device=Q.device, dtype=torch.float32)  # running max
    L = torch.zeros((batch_size, num_heads, seq_len), 
                    device=Q.device, dtype=torch.float32)  # running sum
    
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

                ### After all Key/Values are processed, we have:
                ### - m_i: the final max value for each query row
                ### - l_i: the final sum of exp values for each query row
                ### 
                ### We need to convert this to the logsumexp format for storage
                ### Logsumexp = m + log2(l)
                ### 
                ### This allows us to recover the exact softmax in the backward pass!
                ### we will see the derivation for this later when we actually go to use it
                ### But the made idea is we need our softmax output for the backward pass
                ### but the softmax output is an N^2 tensor that we spent so much effort 
                ### here to not compute. But the logsumexp trick gives us a quick way to 
                ### recompute it by just storing this single value per row of softmax later!
                m_i_logsumexp = m_i + torch.log2(l_i)

                ### Store the output in M ###
                M[b, h, q_start:q_end] = m_i_logsumexp

                ### After all Key/Values are processed lets go ahead and normalize by the final sum ###
                ### l_i is our per row accumulated sum that we have been 
                O_block = O_block / (l_i[:, None] + 1e-6)

                ### Store the results ###
                O[b, h, q_start:q_end, :] = O_block
        
    return O, M

def flash_attention_backward(
    Q,  # (batch, num_heads, seq_len, head_dim)
    K,  # (batch, num_kv_heads, seq_len, head_dim)
    V,  # (batch, num_kv_heads, seq_len, head_dim)
    O,  # (batch, num_heads, seq_len, head_dim) - output from forward
    dO, # (batch, num_heads, seq_len, head_dim) - gradient from upstream
    M,  # (batch, num_heads, seq_len) - logsumexp from forward pass
    softmax_scale=None,
    is_causal=False,
    BLOCK_SIZE_Q=64,
    BLOCK_SIZE_KV=64
):
    """
    Backward pass for Flash Attention
    
    Returns:
        dQ: gradient w.r.t Q
        dK: gradient w.r.t K  
        dV: gradient w.r.t V
    """

    batch_size, num_heads, seq_len, head_dim = Q.shape

    ### use default softmax scaling if not provided ###
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)

    ### Same adjustments as forward pass for exp2 vs exp ###
    INV_LN2 = 1.442695040888963
    LN2 = 0.693147182464 # 1 / INV_LN2 so we can unscale later

    ### Adjust our softmax scale with this as we will be using it in a bit! 
    softmax_scale *= INV_LN2

    ### Prescale our Queries ###
    Q = Q * softmax_scale

    ### Initialize gradients ###
    dQ = torch.zeros_like(Q)
    dK = torch.zeros_like(K)
    dV = torch.zeros_like(V)

    ########################################
    ### STEP 1: PREPROCESS - Compute D ###
    ########################################

    ########################################
    ### WHY D = sum(dO * O) APPEARS IN BACKWARD
    ########################################

    ### Forward (per row i):

    ###     S_i = Q_i K^T / sqrt(d)
    ###     P_i = softmax(S_i)
    ###     O_i = P_i V = sum_j P_ij V_j

    ### In the backward pass, we are given dO_i = ∂L / ∂O_i and want ∂L / ∂S_i.

    ### From O_i = sum_j P_ij V_j,
    ###     dP_ij = dO_i · V_j

    ### For softmax, the gradient w.r.t. logits S is:
    ###     dS_ij = P_ij ( dP_ij - sum_k P_ik dP_ik )

    ### The second term couples all elements in the row and must be computed
    ### for each row i.

    ### Consider the row-wise sum:

    ### sum_k P_ik dP_ik
    ###     = sum_k P_ik (dO_i · V_k)
    ###     = dO_i · (sum_k P_ik V_k)
    ###     = dO_i · O_i

    # This shows that the softmax normalization term depends only on
    # the dot product between the output O_i and its gradient dO_i.

    # D removes the component of the gradient that would uniformly shift
    # all logits in a row. This enforces the softmax constraint and avoids
    # explicitly forming the softmax Jacobian or storing dP.
    D = torch.sum(dO * O, dim=-1) # (batch, num_heads, seq_len)

    #########################################
    ### STEP 2: COMPUTE dK and dV ###
    #########################################
    ### For dK and dV, we process blocks of K/V and loop through blocks of Q ###
    ### This is because: ###
    ###   dV = P^T @ dO  (where P is softmax output)
    ###   dK = dS^T @ Q  (where dS is gradient of pre-softmax scores)
    
    ### Loop over Batch and Head 
    for b in range(batch_size):
        for h in range(num_heads):

            ### Process K/V in Blocks (these are the "columns" we hold constant) ###
            for kv_start in range(0, seq_len, BLOCK_SIZE_KV):

                ### Make sure we only grab upto the valid K/V ###
                kv_end = min(kv_start + BLOCK_SIZE_KV, seq_len)

                ### Store indexes of K/V being processed
                kv_indices = torch.arange(kv_start, kv_end, device=Q.device)
                
                ### Load block of keys and values ###
                K_block = K[b, h, kv_start:kv_end, :]  # [block_kv, head_dim]
                V_block = V[b, h, kv_start:kv_end, :]  # [block_kv, head_dim]

                ### Initialize Accumulator for this Block ###
                dK_block = torch.zeros((kv_end - kv_start, head_dim), device=Q.device, dtype=torch.float32)
                dV_block = torch.zeros((kv_end - kv_start, head_dim), device=Q.device, dtype=torch.float32)

                ### Determine which queries to process based on causality ###

                ### [A11 A12 A13 A14 A15 A16]
                ### [A21 A22 A23 A24 A25 A26]
                ### [A31 A32 A33 A34 A35 A36]
                ### [A41 A42 A43 A44 A45 A46]
                ### [A51 A52 A53 A54 A55 A56]
                ### [A61 A62 A63 A64 A65 A66]

                ### For example If we are processing KV indexes 3,4 Then we have two choices
                ### If causal, we are processing Q indexes 3,4,5,6 So we would be processing (+++)
                ### [A11 A12 A13 A14 A15 A16]
                ### [A21 A22 A23 A24 A25 A26]
                ### [A31 A32 +++ +++ A35 A36]
                ### [A41 A42 +++ +++ A45 A46]
                ### [A51 A52 +++ +++ A55 A56]
                ### [A61 A62 +++ +++ A65 A66]

                ### Remember this is the full attention mask we want (without explicitly storing the whole thing)
                ### [A11 xxx xxx xxx xxx xxx]
                ### [A21 A22 xxx xxx xxx xxx]
                ### [A31 A32 A33 xxx xxx xxx]
                ### [A41 A42 A43 A44 xxx xxx]
                ### [A51 A52 A53 A54 A55 xxx]
                ### [A61 A62 A63 A64 A65 A66]

                ### But in the small chunk of:
                ### [A33 A34]
                ### [A43 A44]

                ### A34 is not a valid position (not causal), so this block
                ### would have to be processed with a mask (diagonal block)

                ### On the other hand the next chunk
                ### [A53 A54]
                ### [A63 A64]

                ### All these positions are valid (postdiagonal)

                if is_causal:
                    ### For causal attention:
                    ### 1. Process the diagonal block (where queries == keys in block range)
                    ### 2. Process all blocks AFTER the diagonal (queries > keys)
                    ###    because remember that causal means wherever our queries only
                    ###    attend to keys that are at the same timestep or before
                    
                    ### Diagonal block: queries from kv_start to kv_end
                    q_ranges_diagonal = [(kv_start, kv_end, "diagonal")]
                    
                    ### Post-diagonal blocks: queries from kv_end to seq_len
                    q_ranges_post_diag = [(kv_end, seq_len, "postdiagonal")]
                    
                    q_ranges = q_ranges_diagonal + q_ranges_post_diag

                ### IF we arent causal then just process everything (all queries!)
                else:
                    q_ranges = [(0, seq_len, "full")]
                
                ### Loop through the query ranges
                for q_start_range, q_end_range, pass_type in q_ranges:

                    ### Process queries in blocks within this range
                    for q_start in range(q_start_range, q_end_range, BLOCK_SIZE_Q):

                        ###  Make sure to only index upto the Q that we care about! 
                        q_end = min(q_start + BLOCK_SIZE_Q, q_end_range)

                        ### Store upto the Q indices we are processing right now 
                        q_indices = torch.arange(q_start, q_end, device=Q.device)

                        ### Load the Q Block and cooresponding gradients and stats
                        Q_block = Q[b, h, q_start:q_end, :]      # [block_q, head_dim]
                        dO_block = dO[b, h, q_start:q_end, :]    # [block_q, head_dim]
                        M_block = M[b, h, q_start:q_end]         # [block_q]
                        D_block = D[b, h, q_start:q_end]         # [block_q]

                        ### Compute attention scores: Q @ K^T ###
                        ### But we want the transpose for dK/dV computation ###
                        ### So we compute K @ Q^T to get a transposed version ###
                        S_T_block = K_block @ Q_block.T

                        ### Recover softmax from logsumexp trick ###
                        ### In forward pass we stored: M = max + log2(sum_exp) ###
                        ### So: P = exp2(S - M) gives us softmax output ###
                        ### But we have the transpose, so: ###
                        P_T_block = torch.exp2(S_T_block - M_block[None, :])  # [block_kv, block_q]

                        ### Apply causal masking on diagonal blocks ###
                        if pass_type == "diagonal":
                            ### For the diagonal block, we need to mask the upper triangle ###
                            ### But since we have the transpose, we mask where kv > q ###
                            ### Just so we have the indexes right! We could have just transposed after ###
                            ### too this just saves a transpose op! ###
                            causal_mask = kv_indices[:, None] <= q_indices[None, :]  # [block_kv, block_q]
                            
                            ### Mask fill the block ###
                            P_T_block = P_T_block.masked_fill(~causal_mask, 0.0)

                        ### Compute gradient for V ###
                        ### dV = P^T @ dO ###
                        ### We already have P^T, so: ###
                        dV_block += P_T_block.to(Q_block.dtype) @ dO_block

                        ### Compute gradient for K ###
                        ### First compute dP (gradient w.r.t. softmax output) ###
                        ### dP = dO @ V^T ###
                        ### But we want dP^T, so: dP^T = V @ dO^T ###
                        dP_T_block = V_block @ dO_block.T  # [block_kv, block_q]

                        ### Compute dS (gradient w.r.t. pre-softmax scores) ###
                        ### Formula: dS = P * (dP - D) ###
                        ### In transpose: dS^T = P^T * (dP^T - D) ###
                        ### Note: D is broadcast across the kv dimension ###
                        dS_T_block = P_T_block * (dP_T_block - D_block[None, :])

                        ### Account for the LN2 factor from exp2 vs exp ###
                        ### Remember we want exp() but used exp2(), so to just
                        ### adjust for the scale factor all our values are multiplied 
                        ### by we want to undo it 
                        dS_T_block = dS_T_block * LN2
                        
                        ### Now compute dK ###
                        ### dK = dS^T @ Q ###
                        ### We have dS^T already, so: ###
                        dK_block += dS_T_block.to(Q_block.dtype) @ Q_block
                
                ### Store the gradients ###
                dK[b, h, kv_start:kv_end, :] = dK_block
                dV[b, h, kv_start:kv_end, :] = dV_block
            
    #########################################
    ### STEP 3: COMPUTE dQ ###
    #########################################
    ### For dQ, we process blocks of Q and loop through blocks of K/V ###
    ### This is because: ###
    ###   dQ = dS @ K  (where dS is gradient of pre-softmax scores)
    ### This is basically the same thing, just going the other direction!
    
    ### Loop over Batch and Head 
    for b in range(batch_size):
        for h in range(num_heads):
            
            ### Process Q in Blocks ###
            for q_start in range(0, seq_len, BLOCK_SIZE_Q):
                
                ### Make sure we only grab up to the valid Q
                q_end = min(q_start + BLOCK_SIZE_Q, seq_len)
                
                ### Store indexes of Q being processed
                q_indices = torch.arange(q_start, q_end, device=Q.device)
                
                ### Load block of queries and corresponding data ###
                Q_block = Q[b, h, q_start:q_end, :]      # [block_q, head_dim]
                dO_block = dO[b, h, q_start:q_end, :]    # [block_q, head_dim]
                M_block = M[b, h, q_start:q_end]         # [block_q]
                D_block = D[b, h, q_start:q_end]         # [block_q]
                
                ### Initialize accumulator for this block ###
                dQ_block = torch.zeros((q_end - q_start, head_dim), device=Q.device, dtype=torch.float32)
                
                ### Determine which keys/values to process based on causality ###
                if is_causal:
                    ### For causal attention:
                    ### 1. Process all blocks BEFORE the diagonal (keys < queries)
                    ### 2. Process the diagonal block (where queries == keys in block range)
                    
                    ### Pre-diagonal blocks: keys from 0 to q_start
                    kv_ranges_pre_diag = [(0, q_start, "pre_diagonal")]
                    
                    ### Diagonal block: keys from q_start to q_end
                    kv_ranges_diagonal = [(q_start, q_end, "diagonal")]
                    
                    kv_ranges = kv_ranges_pre_diag + kv_ranges_diagonal
                else:
                    ### For non-causal: process all keys/values
                    kv_ranges = [(0, seq_len, "full")]
                
                ### Loop through key/value ranges ###
                for kv_start_range, kv_end_range, pass_type in kv_ranges:
                    
                    ### Process keys/values in blocks within this range ###
                    for kv_start in range(kv_start_range, kv_end_range, BLOCK_SIZE_KV):
                        
                        ### Make sure to only index up to the K/V that we care about!
                        kv_end = min(kv_start + BLOCK_SIZE_KV, kv_end_range)
                        
                        ### Store the K/V indices we are processing right now
                        kv_indices = torch.arange(kv_start, kv_end, device=Q.device)
                        
                        ### Load K and V blocks ###
                        K_block = K[b, h, kv_start:kv_end, :]  # [block_kv, head_dim]
                        V_block = V[b, h, kv_start:kv_end, :]  # [block_kv, head_dim]
                        
                        ### Compute attention scores: Q @ K^T ###
                        S_block = Q_block @ K_block.T  # [block_q, block_kv]
                        
                        ### Recover softmax from logsumexp trick ###
                        ### P = exp2(S - M) ###
                        P_block = torch.exp2(S_block - M_block[:, None])  # [block_q, block_kv]
                        
                        ### Apply causal masking on diagonal blocks ###
                        if pass_type == "diagonal":
                            ### For the diagonal block, mask upper triangle ###
                            ### (queries should not attend to future keys) ###
                            causal_mask = q_indices[:, None] >= kv_indices[None, :]
                            P_block = P_block.masked_fill(~causal_mask, 0.0)
                        
                        ### Compute dP (gradient w.r.t. softmax output) ###
                        ### dP = dO @ V^T ###
                        dP_block = dO_block @ V_block.T  # [block_q, block_kv]
                        
                        ### Compute dS (gradient w.r.t. pre-softmax scores) ###
                        ### Formula: dS = P * (dP - D) ###
                        dS_block = P_block * (dP_block - D_block[:, None])
                        
                        ### Account for the LN2 factor from exp2 vs exp ###
                        dS_block = dS_block * LN2
                        
                        ### Compute dQ ###
                        ### dQ = dS @ K ###
                        dQ_block += dS_block.to(Q_block.dtype) @ K_block
                
                ### Apply the softmax scale to dQ (same scaling as forward) ###
                dQ_block = dQ_block * softmax_scale
                
                ### Store the gradients ###
                dQ[b, h, q_start:q_end, :] = dQ_block
    
    return dQ, dK, dV


if __name__ == "__main__":

    q = torch.randn(2,4,32,64, device="cuda", requires_grad=True)
    k = torch.randn(2,4,32,64, device="cuda", requires_grad=True)
    v = torch.randn(2,4,32,64, device="cuda", requires_grad=True)
    dO = torch.randn(2,4,32,64, device="cuda", requires_grad=True)
    
    ### Pseudo Forward Pass ###
    O, M = flash_attention_forward(q,k,v, is_causal=True)
    
    ### Pseudo Backward Pass ###
    dQ, dK, dV = flash_attention_backward(q, k, v, O, dO, M, is_causal=True)
    
    ### Torch SDPA Forward ###
    output = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, is_causal=True
    )

    print("Max Forward Pass Output Diff:", torch.max(torch.abs(output - O)).detach().item())    

    output.backward(dO)
    print("Max Backward Pass dQ Diff:", torch.max(torch.abs(q.grad - dQ)).detach().item())    
    print("Max Backward Pass dK Diff:", torch.max(torch.abs(k.grad - dK)).detach().item())    
    print("Max Backward Pass dV Diff:", torch.max(torch.abs(v.grad - dV)).detach().item())    