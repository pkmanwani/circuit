import matplotlib.pyplot as plt
import csv
import os
import pandas as pd
# Get the current working directory (directory of the Python file)
current_directory = os.getcwd()

# Set the relative path to the CSV files within the workspace
csv_directory = os.path.join(current_directory, "experimentaldata")

# Create a directory on the desktop to save the plots
desktop_directory = current_directory
plots_directory = os.path.join(desktop_directory, "plots")
os.makedirs(plots_directory, exist_ok=True)

# Get a list of all CSV files in the directory
csv_files = [file for file in os.listdir(csv_directory) if file.endswith(".csv")]

# Iterate over each CSV file
for file in csv_files[1:2]:
    # Create the full path to the CSV file
    file_path = os.path.join(csv_directory, file)

    # Open the CSV file
    with open(file_path, 'r') as csv_file:
        # Read the CSV data
        csv_reader = csv.reader(csv_file)

        # Determine the number of rows in the CSV file
        num_rows = sum(1 for _ in csv_reader)
        print(f"Total rows: {num_rows}")
        csv_file.seek(0)  # Reset the file pointer to the beginning of the file

        # Extract the float value from row 9 column 2
        for _ in range(7):  # Skip the first 8 rows
            next(csv_reader)

        row = next(csv_reader)
        line1_unit = float(row[1][:-2])
        #print(f"Voltage per ADC value: {line1_unit}")
        print(row)
        print("We assume it is in mV for now")
        if len(row) > 2:
            line2_unit= float(row[2][:-2])
        if len(row) > 3:
            line3_unit= float(row[3][:-2])
        row_time = next(csv_reader)
        time_interval = float(row_time[1][:-2])  # Extract float value without the characters at the end
        print(f'Assuming time interval is same: {time_interval}')

        # Skip the remaining rows
        for _ in range(2):  # Skip 3 more rows
            next(csv_reader)

        # Extract the data from columns 1 and 2
        time= []
        line1 = []
        line2 = []  # Array to store the third column values
        line3 = []  # Array to store the third column values
        for row in csv_reader:
            time.append(float(row[0]) * time_interval)  # Multiply time by time_interval and row value
            line1.append(float(row[1])/1000)  # Multiply y values by 1000. 0.001 V = 1 V

            # Check if the third column exists
            if len(row) > 2:
                line2.append(float(row[2])/1000)  # Multiply the third column values by 10. 0.1 V = 1 A
            else:
                line2.append(0)  # Assign a default value if the third column is missing

                # Check if the third column exists
            if len(row) > 3:
                line3.append(float(row[3])/1000)  # Divide the third column values by 1000
            else:
                line3.append(0)  # Assign a default value if the third column is missing

        # Create a line plot with linewidth=0.8 for y
        import scipy.signal as signal

        # First, design the Buterworth filter
        N = 3  # Filter order
        Wn = 0.1  # Cutoff frequency
        B, A = signal.butter(N, Wn, output='ba')
        smooth_line2 = signal.filtfilt(B, A, line2)
        plt.plot(time, line1, label='Capacitor (kV)', linewidth=0.5)

        # Plot x against z only if the third column exists
        if len(line2) > 0:
            plt.plot(time, line2, label='Current Probe (10 A)', linewidth=0.5)

        if len(line3) > 0:
            plt.plot(time, line3, label='Interferometer signal (V)', linewidth=0.5)

        plt.xlabel("Time (us)")
        #plt.ylabel("Am")
        plt.title(file[:-4])  # Set the title as the name of the file
        plt.legend()

        # Save the plot in the plots directory on the desktop
        plot_filename = os.path.join(plots_directory, f"{file[:-4]}.png")
        plt.savefig(plot_filename, dpi=600)
        plt.show()
        plt.close()

# Show a message when all plots are saved
print("Plots saved successfully in the 'plots' folder on your desktop.")
