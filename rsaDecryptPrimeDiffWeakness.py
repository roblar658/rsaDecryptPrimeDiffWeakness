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
CHUNK_SIZE = 7


# --- Primtallsgenerering (ikke-Mersenne) ---

def is_prime(n: int, k: int = 25) -> bool:
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
    """Sjekker om tallet er på formen 2^k - 1."""
    return (p + 1 & p) == 0


def generate_random_prime(bits: int = 32) -> int:
    """Genererer et tilfeldig primtall som garantert ikke er Mersenne."""
    while True:
        cand = secrets.randbits(bits) | 1 | (1 << (bits - 1))
        if is_mersenne(cand):
            continue
        if is_prime(cand):
            return cand


def generate_keys_with_distance(target_diff: int = 25_000_000):
    """
    Genererer et nytt, tilfeldig nøkkelsett der |p - q| er rundt target_diff.
    Ingen av faktorene er Mersenne-primtall.
    """
    p = generate_random_prime(bits=32)
    
    # Velg tilfeldig om q ligger over eller under p
    sign = 1 if secrets.randbits(1) == 1 else -1
    q_cand = p + sign * (target_diff + secrets.randbelow(50_000))
    if q_cand % 2 == 0:
        q_cand += 1

    while not (is_prime(q_cand) and not is_mersenne(q_cand)):
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
    """Laster eksisterende nøkkelsett, eller lager nytt om det mangler."""
    if os.path.exists(FILE_KEYS):
        with open(FILE_KEYS, "r", encoding="utf-8") as f:
            return json.load(f)
    return generate_keys_with_distance()


# --- Parallell Fermat-arbeider ---

def _fermat_worker(
    worker_id: int,
    num_workers: int,
    n: int,
    a_base: int,
    max_steps: int,
    stop_event,
    result_val,
    global_counter,
    counter_lock,
):
    valid_mod64 = {0, 1, 4, 9, 16, 17, 25, 33, 36, 41, 49, 57}
    stride = num_workers
    local_steps = 0
    BATCH = 16384

    for step in range(worker_id, max_steps, stride):
        local_steps += 1
        if (local_steps & (BATCH - 1)) == 0:
            with counter_lock:
                global_counter.value += BATCH
            if stop_event.is_set():
                return

        a = a_base + step
        b2 = a * a - n

        if (b2 & 63) not in valid_mod64:
            continue

        b = math.isqrt(b2)
        if b * b == b2:
            with result_val.get_lock():
                if not stop_event.is_set():
                    result_val.value = a
                    stop_event.set()
            with counter_lock:
                global_counter.value += (local_steps % BATCH)
            return

    with counter_lock:
        global_counter.value += (local_steps % BATCH)


def _progress_printer(global_counter, est_steps, stop_event, t0):
    bar_length = 30
    while not stop_event.is_set():
        time.sleep(0.25)
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
            f"\rProgress: [{bar}] {pct:5.1f}% | {current/1e6:6.2f}M/{est_steps/1e6:6.2f}M | "
            f"{speed/1e6:4.2f} Mstep/s | Tid: {int(elapsed)}s | ETA: {int(rem_sec)}s  "
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

    stop_event = mp.Event()
    result_val = mp.Value("q", 0)
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
                result_val,
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

    if result_val.value > 0:
        a = result_val.value
        b2 = a * a - n
        b = math.isqrt(b2)
        return a - b, a + b

    return None


# --- Kryptering og Dekryptering ---

def ensure_input_file(filepath: str):
    if not os.path.exists(filepath):
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(
                "Hemmelig fil generert for RSA-test.\n"
                "Primtallene her er tilfeldig valgte ikke-Mersenne-tall!\n"
            )
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
        print("\n[FEIL] OverflowError: 'encrypted.txt' matcher ikke nåværende nøkler.")
        print("Løsning: Kjør valg '1' (Krypter) for å oppdatere kryptert fil.")
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
    print(f"\nStarter parallell Fermat-knekking over {cpus} kjerner...")
    print(f"Avstand |p - q|: {diff:,}")
    print(f"Mål-steg:        ca. {est_steps:,} iterasjoner")

    t0 = time.perf_counter()
    res = parallel_fermat_factorization(
        n, max_steps=est_steps + 2_000_000, est_steps=est_steps, num_workers=cpus
    )
    t1 = time.perf_counter()

    if not res:
        print("\nFaktorisering feilet eller ble avbrutt.")
        return

    found_p, found_q = res
    print(f"\nFaktorisering vellykket på {t1 - t0:.2f} sekunder!")
    print(f"Funnet faktorer: {found_p} og {found_q}")

    recovered_phi = (found_p - 1) * (found_q - 1)
    recovered_d = pow(keys["e"], -1, recovered_phi)
    print(f"Rekonstruert d:  {recovered_d}")

    print("\nDekrypterer med den knekte nøkkelen...")
    decrypt_file(keys, custom_d=recovered_d)


# --- Hovedmeny ---

def main():
    ensure_input_file(FILE_PLAIN)
    keys = load_or_create_keys()

    print("=" * 65)
    print("RSA DEMO MED TILFELDIGE, IKKE-MERSENNE PRIMTALL")
    print("=" * 65)
    print(f"Aktiv Modulus n: {keys['n']} ({int(keys['n']).bit_length()} bits)")
    print(f"Aktiv p:         {keys['p']} (Mersenne: {is_mersenne(keys['p'])})")
    print(f"Aktiv q:         {keys['q']} (Mersenne: {is_mersenne(keys['q'])})")
    print(f"Avstand |p - q|: {keys['diff']:,}")
    print("-" * 65)
    print("1: Krypter 'file.txt' -> 'encrypted.txt'")
    print("2: Dekrypter 'encrypted.txt' -> 'decrypted.txt' (med nøkkel d)")
    print("3: KNEKK 'encrypted.txt' med Fermat (viser progress) -> 'decrypted.txt'")
    print("4: GENERER NYE TILFELDIGE NØKLER (ikke-Mersenne)")
    print("q: Avslutt")
    print("-" * 65)

    valg = input("Velg handling (1/2/3/4/q): ").strip().lower()

    if valg == "1":
        encrypt_file(keys)
    elif valg == "2":
        decrypt_file(keys)
    elif valg == "3":
        crack_and_decrypt(keys)
    elif valg == "4":
        dist_str = input("Oppgi ønsket avstand |p - q| (trykk Enter for ~25 mill): ").strip()
        dist = int(dist_str) if dist_str.isdigit() else 25_000_000
        new_keys = generate_keys_with_distance(dist)
        print(f"\nNye nøkler generert og lagret til '{FILE_KEYS}'!")
        print(f"Ny p: {new_keys['p']}, Ny q: {new_keys['q']}, Avstand: {new_keys['diff']:,}")
    elif valg == "q":
        print("Avslutter.")
    else:
        print("Ugyldig valg.")


if __name__ == "__main__":
    main()
