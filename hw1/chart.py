import matplotlib.pyplot as plt

methods = [0.5552322253985197,0.4347440455274725,0.4867832932856734,0.6014318125855583,0.5994970580650516,0.6184462613145231,0.5960986238876629,0.654067440322969,0.6211150379170536,0.5915113134537067,0.6713683551110319,0.7349046094021524,0.728922083450112,0.7081949410435889,0.724465272893239,0.7596360083116614,0.7721144025818151,0.7932328489828333,0.8039384843252966,0.8004774982489462,0.7871689940184119,0.7887079080061282,0.8048730336911665,0.8018864499974275,0.8482453448552589,0.8516095833307258,0.8431948452683862,0.8716635503721663]

plt.figure(figsize=(10, 6))

# 繪製 methods 列表中的數值
plt.plot(methods, marker='o', linestyle='-')

plt.xlabel('Index', fontsize=12)
plt.ylabel('Value', fontsize=12)
plt.title('Reporjection Error', fontsize=14)

plt.show()