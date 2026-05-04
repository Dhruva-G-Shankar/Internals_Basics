import random
import requests

URL = "http://127.0.0.1:8500/infer"

random.seed(42)


def send(payload):
    r = requests.post(URL, json=payload)
    return r.json()


def normal_payload():
    return {
        "code_context_lines": random.randint(5, 200),
        "language_complexity": random.randint(1, 5),
        "prompt_length": random.randint(10, 500),
        "is_inline": random.randint(0, 1),
    }


def drifted_payload():
    return {
        "code_context_lines": random.randint(140, 200),
        "language_complexity": random.randint(1, 5),
        "prompt_length": random.randint(350, 500),
        "is_inline": random.randint(0, 1),
    }


def main():
    total = 0

    for _ in range(40):
        send(normal_payload())
        total += 1

    for _ in range(10):
        send(drifted_payload())
        total += 1

    print(f"Sent {total} requests")


if __name__ == "__main__":
    main()