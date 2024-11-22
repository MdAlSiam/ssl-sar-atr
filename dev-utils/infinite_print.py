import sys
import time

def print_forever():
    if len(sys.argv) < 2:
        print("Usage: python script.py <text_to_print>")
        sys.exit(1)

    text = sys.argv[1]
    count = 1

    try:
        while True:
            print(f"{count}: {text}")
            count += 1
            time.sleep(0.5)  # Optional: add a delay to make output readable
    except KeyboardInterrupt:
        print("\nProgram terminated by user.")

if __name__ == "__main__":
    print_forever()

# Usage: python common-utils/infinite_print.py "Hello from local-1!"
