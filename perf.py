import pickle
import numpy as np
import matplotlib.pyplot as plt

with open("gait_results.pkl", "rb") as f:
    gait = pickle.load(f)

# Convert values from pickle (int64 → int)
x = np.array([int(i) for i in gait["view_regulation"]])
y = np.array([int(i) for i in gait["graphical_regression"]])

plt.figure(figsize=(9,5))

# ----- Scatter -----
plt.scatter(x, y, color="magenta", s=50)

# ----- Regression Line (Least Squares Fit) -----
coeff = np.polyfit(x, y, 1)     # slope + intercept
reg_line = np.poly1d(coeff)    

plt.plot(x, reg_line(x), color="green", linewidth=2)

# Labels
plt.xlabel("View Regulation")
plt.ylabel("Graphical Regression")
plt.grid(True)

plt.show()


plt.figure(figsize=(7,4))
plt.plot(gait["epochs"], gait["accuracy"], '-.o', linewidth=1.5)
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.grid(True)
plt.show()

plt.figure(figsize=(7,4))
plt.plot(gait["epochs"], gait["recall"], '-.o', linewidth=1.5)
plt.xlabel("Epoch")
plt.ylabel("Recall (%)")
plt.grid(True)
plt.show()

plt.figure(figsize=(7,4))
plt.plot(gait["epochs"], gait["f1_score"], '-.o', linewidth=1.5)
plt.xlabel("Epoch")
plt.ylabel("F1-Score (%)")
plt.grid(True)
plt.show()

plt.figure(figsize=(7,4))
plt.plot(gait["epochs"], gait["similarity_ratio"], '-.o', linewidth=1.5)
plt.xlabel("Epoch")
plt.ylabel("Similarity Ratio (%)")
plt.grid(True)
plt.show()

plt.figure(figsize=(7,4))
plt.plot(gait["accuracy_models"], '-.o', linewidth=1.5)
plt.xticks([0,1,2,3], ['Fine tree','ELM','KELM','Proposed'])
plt.ylabel("Accuracy (%)")
plt.grid(True)
plt.show()

plt.figure(figsize=(7,4))
plt.plot(gait["error_models"], '-.o', linewidth=1.5)
plt.xticks([0,1,2,3], ['Fine tree','ELM','KELM','Proposed'])
plt.ylabel("Error (%)")
plt.grid(True)
plt.show()

plt.figure(figsize=(7,4))
plt.plot(gait["computational_time"], '-.o', linewidth=1.5)
plt.xticks([0,1,2,3], ['Fine tree','ELM','KELM','Proposed'])
plt.ylabel("Computational Time (s)")
plt.grid(True)
plt.show()

plt.figure(figsize=(7,4))
plt.plot(gait["fnr_models"], '-.o', linewidth=1.5)
plt.xticks([0,1,2,3], ['Fine tree','ELM','KELM','Proposed'])
plt.ylabel("FNR (%)")
plt.grid(True)
plt.show()



