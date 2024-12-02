from .constants import POSEIDON_C, POSEIDON_M, FIELD_MODULUS

def ark(state, c, it, t):
    
    for i in range(t):
        state[i] = (state[i] + c[it + i]) % FIELD_MODULUS
    return state

def sbox(state, f, p, r, t):
    
    state[0] = pow(state[0], 5, FIELD_MODULUS)

    # Determine whether to apply the S-Box to the rest
    apply_full = (r < f // 2) or (r >= f // 2 + p)
    if apply_full:
        for i in range(1, t):
            state[i] = pow(state[i], 5, FIELD_MODULUS)
    return state

def mix(state, m, t):
    
    out = [0] * t
    for i in range(t):
        acc = 0
        for j in range(t):
            acc += state[j] * m[i][j]
        out[i] = acc % FIELD_MODULUS
    return out


def poseidon_hash(inputs: list[int]):
    """
    Poseidon Hash Function.
    """
    N = len(inputs)
    assert 0 < N <= 6  

    t = N + 1
    rounds_p = [56, 57, 56, 60, 60, 63, 64, 63]  # Round constants
    f = 8  # Full S-Box rounds
    p = rounds_p[t - 2]  # Partial S-Box rounds

    # Constants
    c = POSEIDON_C[t-2]
    m = POSEIDON_M[t-2]

    # Initialize state
    state = [0] * t
    for i in range(1, t):
        state[i] = inputs[i - 1]

    # Apply rounds
    for r in range(f + p):
        state = ark(state, c, r * t, t)
        state = sbox(state, f, p, r, t)
        state = mix(state, m, t)

    return state[0]  


# Usage
def main():
    #inputs = [1, 2, 90]  
    inputs = [
        12271534372237423045703860527745063377868729813954318194982046664680176441910,
        18420990134585988265114260052767351209553100278555759449994308396142725656951
    ]
    hash_result = poseidon_hash(inputs)
    print("poseidon", hash_result)

if __name__ == "__main__":
    main()