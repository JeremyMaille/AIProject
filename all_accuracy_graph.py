import matplotlib.pyplot as plt
import subprocess
import os

# Define paths for the programs
programs = {
    "Train Model 100": "reseau_neurone/train_model_100.py",
    "Random Forest": "randomForest/random_forest.py",
    "Logistic Regression": "logisticRegression/logistic_regression.py",
    "Optimized Logistic Regression": "logisticRegression/optimized_logistic_regression.py"
}

# Run each program and collect its output
results = {}

def run_program(program_name, script_path):
    try:
        print(f"Running {program_name}...")
        result = subprocess.run(["python", script_path], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error running {program_name}: {result.stderr}")
            return None

        # Extract accuracy data from the program's standard output
        lines = result.stdout.splitlines()
        accuracy = []
        epochs = []
        for line in lines:
            if "Epoch" in line and "Accuracy:" in line:
                parts = line.split()
                epoch = int(parts[1].strip(':'))
                acc = float(parts[-1])
                epochs.append(epoch)
                accuracy.append(acc)

        results[program_name] = (epochs, accuracy)

    except Exception as e:
        print(f"Failed to run {program_name}: {e}")

# Execute all programs
for program_name, script_path in programs.items():
    run_program(program_name, script_path)

# Initialize plot
plt.figure(figsize=(10, 6))

# Plot the results
for label, (epochs, accuracy) in results.items():
    plt.plot(epochs, accuracy, label=label, marker='o')

# Configure plot
plt.title("Test Accuracy Comparison")
plt.xlabel("Epochs")
plt.ylabel("Test Accuracy")
plt.legend()
plt.grid(True)

# Save and show the plot
plt.savefig("combined_test_accuracy.png")
plt.show()
