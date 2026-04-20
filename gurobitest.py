import gurobipy as gp

# Create a model
m = gp.Model("test")

# Check if it recognized your license
print(f"License type: {m.getParamInfo('LicenseID')[2]}")