import math
import random
import time

# ==========================================
# 1. Matematikk og Primtallsgenerering
# ==========================================

SMALL_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71]


def extended_gcd(a, b):
    if b == 0:
        return a, 1, 0
    g, x1, y1 = extended_gcd(b, a % b)
    return g, y1, x1 - (a // b) * y1


def mod_inv(e, m):
    g, x, _ = extended_gcd(e, m)
    if g != 1:
        raise ValueError("Modulær invers eksisterer ikke")
    return x % m


def is_prime(n, k=25):
    """Miller-Rabin primalitetstest."""
    if n < 2:
        return False
    if n in (2, 3):
        return True
    if n % 2 == 0:
        return False

    r, s = 0, n - 1
    while s % 2 == 0:
        r += 1
        s //= 2

    for _ in range(k):
        a = random.randrange(2, n - 1)
        x = pow(a, s, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True


def get_prime(bits):
    """Genererer et garantert tilfeldig primtall med angitt bitlengde."""
    while True:
        candidate = random.getrandbits(bits) | (1 << (bits - 1)) | 1
        if any(candidate % sp == 0 for sp in SMALL_PRIMES if sp < candidate):
            continue
        if is_prime(candidate):
            return candidate


def next_prime(n):
    """Finner det neste oddetallsprimtallet over n."""
    if n % 2 == 0:
        n += 1
    else:
        n += 2
    while not is_prime(n):
        n += 2
    return n


def rsa_encrypt(message_int, e, n):
    return pow(message_int, e, n)


def rsa_decrypt(cipher_int, d, n):
    return pow(cipher_int, d, n)


# ==========================================
# 2. Faktoriseringsmetoder
# ==========================================

def fermat_factor(n, max_steps=10_000):
    """Sårbarhet: p ≈ q (primtallene er for nær hverandre)."""
    a = math.isqrt(n)
    if a * a < n:
        a += 1

    for _ in range(max_steps):
        b2 = a * a - n
        b = math.isqrt(b2)
        if b * b == b2:
            return a - b, a + b
        a += 1
    return None


def pollards_rho_brent(n, max_steps=50_000):
    """
    Sårbarhet: min(p, q) << sqrt(n).
    Har en øvre steg-grense slik at den gir opp og lar Wiener overta
    dersom begge faktorene er store (balanserte).
    """
    if n % 2 == 0:
        return 2, n // 2
    if n % 3 == 0:
        return 3, n // 3

    y = random.randint(2, n - 1)
    c = random.randint(1, n - 1)
    m = 128
    g = 1
    r = 1
    q = 1
    steps = 0

    while g == 1 and steps < max_steps:
        x = y
        for _ in range(r):
            y = (pow(y, 2, n) + c) % n

        k = 0
        while k < r and g == 1 and steps < max_steps:
            ys = y
            limit = min(m, r - k)
            for _ in range(limit):
                y = (pow(y, 2, n) + c) % n
                q = (q * abs(x - y)) % n
            g = math.gcd(q, n)
            k += limit
            steps += limit

        r *= 2

    if g == n:
        while True:
            ys = (pow(ys, 2, n) + c) % n
            g = math.gcd(abs(x - ys), n)
            if g > 1:
                break

    if 1 < g < n:
        return g, n // g

    return None


def continued_fractions(numerator, denominator):
    coeffs = []
    while denominator != 0:
        q = numerator // denominator
        coeffs.append(q)
        numerator, denominator = denominator, numerator - q * denominator
    return coeffs


def convergents(coeffs):
    n_prev, n_curr = 0, 1
    d_prev, d_curr = 1, 0
    for c in coeffs:
        n_prev, n_curr = n_curr, c * n_curr + n_prev
        d_prev, d_curr = d_curr, c * d_curr + d_prev
        yield n_curr, d_curr


def wieners_attack(e, n):
    """Sårbarhet: d < (1/3) * n^(1/4)."""
    coeffs = continued_fractions(e, n)
    for k, d in convergents(coeffs):
        if k == 0:
            continue
        if (e * d - 1) % k != 0:
            continue

        phi = (e * d - 1) // k
        s = n - phi + 1
        discriminant = s * s - 4 * n

        if discriminant >= 0:
            sqrt_disc = math.isqrt(discriminant)
            if sqrt_disc * sqrt_disc == discriminant:
                p = (s + sqrt_disc) // 2
                q = (s - sqrt_disc) // 2
                if p * q == n and p > 1 and q > 1:
                    return p, q
    return None


# ==========================================
# 3. Felles analysemotor
# ==========================================

def crack_rsa(n, e, ciphertext, original_d=None):
    print(f"\n[+] Angriper Modulus ({n.bit_length()} bits): {n}")

    method = None
    factors = None

    # 1. Fermat
    t0 = time.perf_counter()
    factors = fermat_factor(n, max_steps=10_000)
    t1 = time.perf_counter()
    if factors:
        method = "Fermat (p ≈ q)"
        elapsed = t1 - t0

    # 2. Pollard's Rho
    if not factors:
        t0 = time.perf_counter()
        factors = pollards_rho_brent(n, max_steps=50_000)
        t1 = time.perf_counter()
        if factors:
            method = "Pollard's Rho Brent (p << q)"
            elapsed = t1 - t0

    # 3. Wiener
    if not factors:
        t0 = time.perf_counter()
        factors = wieners_attack(e, n)
        t1 = time.perf_counter()
        if factors:
            method = "Wiener's Continued Fractions (liten d)"
            elapsed = t1 - t0

    if not factors:
        print("[-] Ingen av metodene fant faktorene innenfor ressursgrensene.")
        return

    p, q = factors
    print(f"[!] Løst via {method} på {elapsed:.6f}s")
    print(f"    Funnet p: {p}")
    print(f"    Funnet q: {q}")

    # Beregn Carmichaels lambda
    lambda_n = math.lcm(p - 1, q - 1)
    recovered_d = mod_inv(e, lambda_n)

    if original_d is not None:
        print(f"    Opprinnelig   d : {original_d}")
        print(f"    Gjenopprettet d : {recovered_d}")

        # Sammenlign modulo lambda(n)
        match = (original_d % lambda_n) == (recovered_d % lambda_n)
        print(f"    Match mod λ(n)  : {match}")

    # Dekryptering
    plaintext_int = rsa_decrypt(ciphertext, recovered_d, n)
    byte_len = (plaintext_int.bit_length() + 7) // 8
    plaintext_bytes = plaintext_int.to_bytes(byte_len, byteorder="big")
    print(f"    Dekryptert tekst: '{plaintext_bytes.decode(errors='replace')}'")


# ==========================================
# 4. Verifiserte Testtilfeller
# ==========================================

if __name__ == "__main__":
    secret_message = b"Vulnerable RSA Key"
    msg_int = int.from_bytes(secret_message, byteorder="big")
    e_std = 65537

    # Scenario A: Primtall for like hverandre (Fermat)
    p_close = get_prime(64)
    q_close = next_prime(p_close)  # q ligger rett ved siden av p
    n_close = p_close * q_close
    lambda_close = math.lcm(p_close - 1, q_close - 1)
    d_close = mod_inv(e_std, lambda_close)
    c_close = rsa_encrypt(msg_int, e_std, n_close)

    # Scenario B: For stor forskjell i bitlengde (Pollard's Rho)
    p_far = get_prime(20)  # Liten 20-bit faktor (~1 000 000)
    q_far = get_prime(256)  # Stor 256-bit faktor
    n_far = p_far * q_far
    lambda_far = math.lcm(p_far - 1, q_far - 1)
    d_far = mod_inv(e_std, lambda_far)
    c_far = rsa_encrypt(msg_int, e_std, n_far)

    # Scenario C: Liten hemmelig eksponent d (Wiener)
    # Balanserte 256-bit primtall (n er 512 bit)
    p_wiener = get_prime(256)
    q_wiener = get_prime(256)
    while p_wiener == q_wiener:
        q_wiener = get_prime(256)
    n_wiener = p_wiener * q_wiener
    lambda_wiener = math.lcm(p_wiener - 1, q_wiener - 1)

    # Wiener-grensen krever d < (1/3) * n^(1/4) ≈ (1/3) * 2^128
    # Vi velger en tilfeldig 35-bit d:
    d_wiener = random.getrandbits(35) | 1
    while math.gcd(d_wiener, lambda_wiener) != 1:
        d_wiener += 2

    e_wiener = mod_inv(d_wiener, lambda_wiener)
    c_wiener = rsa_encrypt(msg_int, e_wiener, n_wiener)

    # Kjør angrepene
    crack_rsa(n_close, e_std, c_close, original_d=d_close)
    crack_rsa(n_far, e_std, c_far, original_d=d_far)
    crack_rsa(n_wiener, e_wiener, c_wiener, original_d=d_wiener)
