import tkinter as tk
from tkinter import messagebox
import joblib
import numpy as np

# Load trained model
model = joblib.load("models/gui_model.pkl")

def predict_churn():

    try:

        tenure = float(tenure_entry.get())
        monthly = float(monthly_entry.get())
        total = float(total_entry.get())

        data = np.array([
            [
                tenure,
                monthly,
                total
            ]
        ])

        prediction = model.predict(data)[0]

        probability = model.predict_proba(data)[0][1]

        probability_percent = round(
            probability * 100,
            2
        )

        if prediction == 1:

            risk_label.config(
                text=f"HIGH RISK ({probability_percent}%)",
                fg="#d90429"
            )

            recommendation_label.config(
                text="Recommendation: Contact customer and offer retention benefits."
            )

        else:

            risk_label.config(
                text=f"LOW RISK ({probability_percent}%)",
                fg="#2b9348"
            )

            recommendation_label.config(
                text="Recommendation: Customer is likely to remain with the company."
            )

    except Exception as e:

        messagebox.showerror(
            "Input Error",
            str(e)
        )

root = tk.Tk()

root.title(
    "Customer Churn Prediction Dashboard"
)

root.geometry("750x550")

root.configure(
    bg="#edf2f4"
)

# Title

title = tk.Label(
    root,
    text="Customer Churn Prediction Dashboard",
    font=("Arial",22,"bold"),
    bg="#edf2f4",
    fg="#1d3557"
)

title.pack(pady=20)

# Input Frame

input_frame = tk.Frame(
    root,
    bg="white",
    bd=2,
    relief="groove"
)

input_frame.pack(
    padx=20,
    pady=10,
    fill="x"
)

tk.Label(
    input_frame,
    text="Tenure (Months)",
    font=("Arial",12),
    bg="white"
).pack(pady=5)

tenure_entry = tk.Entry(
    input_frame,
    width=35,
    font=("Arial",12)
)

tenure_entry.pack()

tk.Label(
    input_frame,
    text="Monthly Charges",
    font=("Arial",12),
    bg="white"
).pack(pady=5)

monthly_entry = tk.Entry(
    input_frame,
    width=35,
    font=("Arial",12)
)

monthly_entry.pack()

tk.Label(
    input_frame,
    text="Total Charges",
    font=("Arial",12),
    bg="white"
).pack(pady=5)

total_entry = tk.Entry(
    input_frame,
    width=35,
    font=("Arial",12)
)

total_entry.pack(pady=(0,10))

predict_btn = tk.Button(
    input_frame,
    text="Predict Churn",
    font=("Arial",12,"bold"),
    width=25,
    bg="#457b9d",
    fg="white",
    command=predict_churn
)

predict_btn.pack(pady=15)

# Results Frame

result_frame = tk.Frame(
    root,
    bg="white",
    bd=2,
    relief="groove"
)

result_frame.pack(
    padx=20,
    pady=20,
    fill="x"
)

tk.Label(
    result_frame,
    text="Prediction Result",
    font=("Arial",18,"bold"),
    bg="white"
).pack(pady=10)

risk_label = tk.Label(
    result_frame,
    text="Waiting for Prediction...",
    font=("Arial",20,"bold"),
    bg="white"
)

risk_label.pack(pady=10)

recommendation_label = tk.Label(
    result_frame,
    text="",
    font=("Arial",12),
    bg="white",
    wraplength=500
)

recommendation_label.pack(pady=10)

root.mainloop()