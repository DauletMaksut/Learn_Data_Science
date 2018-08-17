import pandas as pd
p1 = pd.Series({"Name": "Daulet",
		  "Age": 20,
		  "Loves": "Absolutely"
})
p2 = pd.Series({"Name":"Maia","Age": 24,"Loves": "Only"})
dts = pd.DataFrame([p1, p2], index = ["haize", "Maize"])
print(dts)
print("Fucking experiment")
print(dts.loc['Maize'])
print("Let me")
print(dts.loc['Maize']['Age'])
