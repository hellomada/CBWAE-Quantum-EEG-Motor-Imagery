import os

# Path where your EDF files are stored
data_path = "/content"  # <-- change if needed

missing_files = []

for i in range(1, 101):  # Subjects S001 to S100
    subject = f"S{i:03d}"
    for run in ["R04", "R06"]:
        filename = f"{subject}{run}.edf"
        if not os.path.exists(os.path.join(data_path, filename)):
            missing_files.append(filename)

if missing_files:
    print(" Missing files:")
    for f in missing_files:
        print(f)
else:
    print(" All 200 EDF files are present!")
