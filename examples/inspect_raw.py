import pickletools
import joblib
import io

# Assuming 'my_subclass_model.joblib' exists
file_path = 'examples/preg.joblib'

try:
    with open(file_path, 'rb') as f:
        pickle_stream = f.read()

    # Use io.BytesIO to treat the byte string as a file-like object
    # for pickletools.dis
    print(f"--- Disassembling {file_path} ---")
    pickletools.dis(io.BytesIO(pickle_stream))

except FileNotFoundError:
    print(f"Error: File '{file_path}' not found.")
except Exception as e:
    print(f"An error occurred: {e}")