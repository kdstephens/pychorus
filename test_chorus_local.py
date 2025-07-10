import os
from pychorus.helpers import find_and_output_chorus

def run_test():
    input_path = "pychorus/tests/local_data/test.wav"
    if not os.path.exists(input_path):
        print("File not found: ", input_path, 
              " Make sure to place your own file")
        return

    chorus_starts = find_and_output_chorus(
        input_file=input_path,
        output_file=None,
        clip_length=10,
        top_n=3
    )

    if not chorus_starts:
        print("No choruses detected.")
    else:
        print("Detected chorus start times (in seconds):")
        for i, t in enumerate(chorus_starts):
            print(f"  Chorus {i+1}: {int(t // 60)}:{t % 60:.2f} (mm:ss)")

if __name__ == "__main__":
    run_test()