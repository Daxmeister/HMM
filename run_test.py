import subprocess

# The exact input you provided (including line breaks)
input_data = """4 4 0.2 0.5 0.3 0.0 0.1 0.4 0.4 0.1 0.2 0.0 0.4 0.4 0.2 0.3 0.0 0.5
4 3 1.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 1.0 0.2 0.6 0.2
1 4 0.0 0.0 0.0 1.0
"""

# Run task0.py with the test input fed to stdin
result = subprocess.run(
    ["python3", "task0.py"],
    input=input_data,
    text=True,
    capture_output=True
)

print("=== OUTPUT ===")
print(result.stdout)

print("=== ERRORS (if any) ===")
print(result.stderr)
