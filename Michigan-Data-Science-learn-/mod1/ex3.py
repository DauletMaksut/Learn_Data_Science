import pandas as pd
p1 = pd.Series({"Name": "Daulet",
		  "Age": 20,
		  "Self-lover": "Absolutely"
})
p2 = pd.Series({"Name":"Maia","Age": 24,"Self-lover": "Only"})
dts = pd.DataFrame([p1, p2], index = ["Shaize", "Maize"])
print(dts)
print("Fucking experiment")
print(dts.loc['Maize'])
print("Let me")
print(dts.loc['Maize']['Age'])
