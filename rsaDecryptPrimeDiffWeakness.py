import json
import math
import multiprocessing as mp
import os
import secrets
import sys
import threading
import time

FILE_PLAIN = "file.txt"
FILE_ENCRYPTED = "encrypted.txt"
FILE_DECRYPTED = "decrypted.txt"
FILE_KEYS = "keys.json"

DEFAULT_E = 65537
CHUNK_SIZE = 120  # 120 bytes = 960 bits, trygt under 1024-bit modulus N

MASK_64 = 0x202021202030213
MASK_63 = 0x402483012450293
MASK_65 = 0x1218A019866014613
MASK_11 = 0x23B


# --- Primtallsgenerering (512 bits) ---

def is_prime(n: int, k: int = 40) -> bool:
    """Miller-Rabin primtallstest."""
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
        a = secrets.randbelow(n - 4) + 2
        x = pow(a, s, n)
        if x in (1, n - 1):
            continue
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True


def is_mersenne(p: int) -> bool:
    return (p + 1 & p) == 0


def generate_random_prime(bits: int = 512) -> int:
    while True:
        cand = secrets.randbits(bits) | 1 | (1 << (bits - 1))
        if is_mersenne(cand):
            continue
        if is_prime(cand, k=40):
            return cand


def generate_keys_for_duration(target_seconds: int = 65, steps_per_second: int = 10_000_000):
    """
    Beregner nøyaktig |p - q| slik at Fermat må iterere gjennom
    ca. target_seconds * steps_per_second steg.
    """
    target_steps = target_seconds * steps_per_second
    p = generate_random_prime(bits=512)

    # Matematisk nødvendig avstand: Delta = sqrt(8 * Steps * p)
    delta = math.isqrt(8 * target_steps * p)

    # Legg til en liten tilfeldig støy så det ikke alltid treffer nøyaktig samme mønster
    delta += secrets.randbelow(10_000_000)

    sign = 1 if secrets.randbits(1) == 1 else -1
    q_cand = p + sign * delta
    if q_cand % 2 == 0:
        q_cand += 1

    print(f"\nSøker etter primtall q med differanse ~2^{delta.bit_length()} bits...")
    while not (is_prime(q_cand, k=40) and not is_mersenne(q_cand)):
        q_cand += 2

    q = q_cand
    n = p * q
    phi = (p - 1) * (q - 1)
    d = pow(DEFAULT_E, -1, phi)

    keys = {
        "p": p,
        "q": q,
        "n": n,
        "e": DEFAULT_E,
        "d": d,
        "diff": abs(p - q),
    }

    with open(FILE_KEYS, "w", encoding="utf-8") as f:
        json.dump(keys, f, indent=4)

    return keys


def load_or_create_keys():
    if os.path.exists(FILE_KEYS):
        with open(FILE_KEYS, "r", encoding="utf-8") as f:
            return json.load(f)
    return generate_keys_for_duration(target_seconds=65)


# --- Parallell Fermat-arbeider ---

def _fermat_worker(
    worker_id: int,
    num_workers: int,
    n: int,
    a_base: int,
    max_steps: int,
    stop_event,
    result_dict,
    global_counter,
    counter_lock,
):
    stride = num_workers
    local_steps = 0
    BATCH = 65536

    m64 = MASK_64
    m63 = MASK_63
    m65 = MASK_65
    m11 = MASK_11

    for step in range(worker_id, max_steps, stride):
        local_steps += 1
        if (local_steps & (BATCH - 1)) == 0:
            with counter_lock:
                global_counter.value += BATCH
            if stop_event.is_set():
                return

        a = a_base + step
        b2 = a * a - n

        if not ((m64 >> (b2 & 63)) & 1):
            continue
        if not ((m63 >> (b2 % 63)) & 1):
            continue
        if not ((m65 >> (b2 % 65)) & 1):
            continue
        if not ((m11 >> (b2 % 11)) & 1):
            continue

        b = math.isqrt(b2)
        if b * b == b2:
            if not stop_event.is_set():
                stop_event.set()
                result_dict["a"] = str(a)
            with counter_lock:
                global_counter.value += (local_steps % BATCH)
            return

    with counter_lock:
        global_counter.value += (local_steps % BATCH)


def _progress_printer(global_counter, est_steps, stop_event, t0):
    bar_length = 30
    while not stop_event.is_set():
        time.sleep(0.3)
        current = global_counter.value
        elapsed = time.perf_counter() - t0

        if elapsed <= 0:
            continue

        speed = current / elapsed
        pct = min(100.0, (current / est_steps) * 100.0) if est_steps > 0 else 0.0

        filled = int(bar_length * (pct / 100.0))
        bar = "=" * filled + (">" if filled < bar_length else "") + "." * max(0, bar_length - filled - 1)
        if filled >= bar_length:
            bar = "=" * bar_length

        rem_sec = (est_steps - current) / speed if (speed > 0 and current < est_steps) else 0

        sys.stdout.write(
            f"\rProgress: [{bar}] {pct:5.1f}% | {current/1e6:7.2f}M/{est_steps/1e6:7.2f}M | "
            f"{speed/1e6:5.2f} Mstep/s | Tid: {int(elapsed)}s | ETA: {int(rem_sec)}s   "
        )
        sys.stdout.flush()


def parallel_fermat_factorization(n: int, max_steps: int, est_steps: int, num_workers: int = None):
    if n % 2 == 0:
        return 2, n // 2

    if num_workers is None:
        num_workers = mp.cpu_count()

    a_base = math.isqrt(n)
    if a_base * a_base < n:
        a_base += 1

    manager = mp.Manager()
    result_dict = manager.dict()
    stop_event = mp.Event()
    global_counter = mp.RawValue("q", 0)
    counter_lock = mp.Lock()

    t0 = time.perf_counter()
    progress_thread = threading.Thread(
        target=_progress_printer,
        args=(global_counter, est_steps, stop_event, t0),
        daemon=True,
    )
    progress_thread.start()

    processes = []
    for wid in range(num_workers):
        p = mp.Process(
            target=_fermat_worker,
            args=(
                wid,
                num_workers,
                n,
                a_base,
                max_steps,
                stop_event,
                result_dict,
                global_counter,
                counter_lock,
            ),
        )
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    stop_event.set()
    progress_thread.join(timeout=0.5)
    sys.stdout.write("\n")
    sys.stdout.flush()

    if "a" in result_dict:
        a = int(result_dict["a"])
        b = math.isqrt(a * a - n)
        return a - b, a + b

    return None


# --- Filkryptering og Dekryptering ---

def ensure_input_file(filepath: str):
    if not os.path.exists(filepath):
        sample = (
            "Dette er en testfil kryptert med et 1024-bits RSA-modulus N.\n"
            "Avstanden mellom p og q er dimensjonert slik at Fermat-angrepet\n"
            "krever over 500 millioner iterasjoner og tar mer enn 1 minutt."
        )
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(sample)
        print(f"Opprettet '{filepath}'.")


def encrypt_file(keys: dict):
    if not os.path.exists(FILE_PLAIN):
        print(f"Finner ikke '{FILE_PLAIN}'.")
        return False

    with open(FILE_PLAIN, "r", encoding="utf-8") as f:
        text = f.read()

    data = text.encode("utf-8")
    cipher_blocks = [len(data)]

    t0 = time.perf_counter()
    for i in range(0, len(data), CHUNK_SIZE):
        chunk = data[i : i + CHUNK_SIZE]
        m = int.from_bytes(chunk, byteorder="big")
        cipher_blocks.append(pow(m, keys["e"], keys["n"]))
    t1 = time.perf_counter()

    with open(FILE_ENCRYPTED, "w", encoding="utf-8") as f:
        for block in cipher_blocks:
            f.write(f"{block}\n")

    print(f"\nKryptering fullført på {(t1 - t0) * 1000:.2f} ms ({len(cipher_blocks) - 1} blokker).")
    print(f"Lagret til: '{FILE_ENCRYPTED}'")
    return True


def decrypt_file(keys: dict, custom_d: int = None):
    if not os.path.exists(FILE_ENCRYPTED):
        print(f"Finner ikke '{FILE_ENCRYPTED}'. Kjør kryptering først.")
        return False

    with open(FILE_ENCRYPTED, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines:
        print(f"'{FILE_ENCRYPTED}' er tom.")
        return False

    total_length = int(lines[0])
    blocks = [int(x) for x in lines[1:]]
    active_d = custom_d if custom_d is not None else keys["d"]

    t0 = time.perf_counter()
    decrypted_bytes = bytearray()
    try:
        for c in blocks:
            m = pow(c, active_d, keys["n"])
            remaining = total_length - len(decrypted_bytes)
            cur_bytes = min(CHUNK_SIZE, remaining)
            decrypted_bytes.extend(m.to_bytes(cur_bytes, byteorder="big"))
    except OverflowError:
        print("\n[FEIL] Nøklene matcher ikke krypteringen.")
        return False

    t1 = time.perf_counter()
    decrypted_text = decrypted_bytes.decode("utf-8", errors="replace")

    with open(FILE_DECRYPTED, "w", encoding="utf-8") as f:
        f.write(decrypted_text)

    print(f"\nDekryptering fullført på {(t1 - t0) * 1000:.2f} ms.")
    print(f"Lagret til: '{FILE_DECRYPTED}'")

    if os.path.exists(FILE_PLAIN):
        with open(FILE_PLAIN, "r", encoding="utf-8") as f:
            original = f.read()
        match = original == decrypted_text
        print(f"Integritet: {'MATCH! 100% identisk med file.txt' if match else 'FEIL: Teksten avviker!'}")
    return True


def crack_and_decrypt(keys: dict):
    if not os.path.exists(FILE_ENCRYPTED):
        print(f"Mangler '{FILE_ENCRYPTED}'. Krypterer automatisk...")
        if not encrypt_file(keys):
            return

    n = keys["n"]
    p, q = keys["p"], keys["q"]
    diff = abs(p - q)

    a_target = (p + q) // 2
    a_base = math.isqrt(n)
    if a_base * a_base < n:
        a_base += 1
    est_steps = a_target - a_base

    cpus = mp.cpu_count()
    print(f"\nStarter optimalisert Fermat-knekking over {cpus} kjerner...")
    print(f"Modulus N:       {n.bit_length()} bits")
    print(f"Avstand |p - q|: {diff.bit_length()} bits (~{diff:.2e})")
    print(f"Mål-steg:        ca. {est_steps:,} steg")

    t0 = time.perf_counter()
    res = parallel_fermat_factorization(
        n, max_steps=est_steps + 10_000_000, est_steps=max(1, est_steps), num_workers=cpus
    )
    t1 = time.perf_counter()

    if not res:
        print("\nFaktorisering feilet eller nådde ikke målet.")
        return

    found_p, found_q = res
    print(f"\nFaktorisering fullført på {t1 - t0:.2f} sekunder!")

    recovered_phi = (found_p - 1) * (found_q - 1)
    recovered_d = pow(keys["e"], -1, recovered_phi)
    print(f"Rekonstruert d:  {recovered_d.bit_length()} bits")

    print("\nDekrypterer filen med den knekte nøkkelen...")
    decrypt_file(keys, custom_d=recovered_d)


# --- Hovedmeny ---

def main():
    ensure_input_file(FILE_PLAIN)
    keys = load_or_create_keys()

    print("=" * 70)
    print("RSA 1024-BIT FERMAT BENCHMARK")
    print("=" * 70)
    print(f"Modulus n:       {keys['n'].bit_length()} bits")
    print(f"Primtall p:      {keys['p'].bit_length()} bits")
    print(f"Primtall q:      {keys['q'].bit_length()} bits")
    print(f"Avstand |p - q|: {keys['diff'].bit_length()} bits")
    print("-" * 70)
    print("1: Krypter 'file.txt' -> 'encrypted.txt'")
    print("2: Dekrypter 'encrypted.txt' -> 'decrypted.txt' (med nøkkel d)")
    print("3: KNEKK 'encrypted.txt' (Estimert tid: ~1 minutt)")
    print("4: GENERER NYE NØKLER (Velg måltid i sekunder)")
    print("q: Avslutt")
    print("-" * 70)

    valg = input("Velg handling (1/2/3/4/q): ").strip().lower()

    if valg == "1":
        encrypt_file(keys)
    elif valg == "2":
        decrypt_file(keys)
    elif valg == "3":
        crack_and_decrypt(keys)
    elif valg == "4":
        sec_str = input("Oppgi ønsket beregningstid i sekunder (f.eks. 60, 90, 120): ").strip()
        sec = int(sec_str) if sec_str.isdigit() else 60
        new_keys = generate_keys_for_duration(target_seconds=sec)
        print(f"\nNye 1024-bit nøkler generert og lagret til '{FILE_KEYS}'!")
        print(f"Differanse satt for ca. {sec} sekunders kjøretid.")
        print("Husk å kjøre valg '1' (Krypter) før du tester knekkingen.")
    elif valg == "q":
        print("Avslutter.")
    else:
        print("Ugyldig valg.")


if __name__ == "__main__":
    main()
