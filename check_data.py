import pandas as pd

csv_path = r"C:\Users\eqela\Desktop\fiber_coupling\Data\2026-06-11\fc_large_misalignment_17-37-27\broad_scan\datasets\broad_global_scan_dataset.csv"

df = pd.read_csv(csv_path)

top10 = df.nlargest(20, "voltage_mV")

print("\nTop 10 voltage values:\n")
print(top10[["m0", "m1", "m2", "m3", "z", "voltage_mV", "std_mV"]])
