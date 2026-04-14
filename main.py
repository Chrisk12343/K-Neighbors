# Create a GUI app for a simple ML model using Tkinter and scikit-learn

# Import tkinter library for building the GUI
import tkinter as tk

# Import ttk for more modern styled widgets like dropdowns
from tkinter import ttk

# Import pandas for handling and processing data
import pandas as pd
print("This application uses the Titanic dataset to predict survival based on passenger features." \
" It trains a Decision Tree and KNN model, " \
"then allows users to input passenger details and see predictions from both models "
"in a user-friendly interface.")

# Import Decision Tree model from sklearn
from sklearn.tree import DecisionTreeClassifier

# Import function to split dataset into training and testing sets
from sklearn.model_selection import train_test_split

# Import accuracy_score to evaluate model performance
from sklearn.metrics import accuracy_score

# Import K-Nearest Neighbors model
from sklearn.neighbors import KNeighborsClassifier

########### LOAD AND PREPARE DATA ###########

# Load the Titanic dataset from CSV file into a pandas DataFrame
df = pd.read_csv('Titanic-Dataset.csv')

# Fill missing Age values with the median age to clean the dataset
df['Age'] = df['Age'].fillna(df['Age'].median())

# Convert categorical "Sex" column into numeric values (male=0, female=1)
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

# Define the features (input variables) used for training
features = ['Pclass','Sex','Age','SibSp']

# Create X (input data) using selected features
X = df[features]

# Create y (target variable) using Survived column
y = df['Survived']

# Split dataset into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create Decision Tree model with max depth of 4
decision_tree_model = DecisionTreeClassifier(max_depth=4, random_state=42)

# Create KNN model with 5 neighbors
knn_model = KNeighborsClassifier(n_neighbors=5)

# Train Decision Tree model using training data
decision_tree_model.fit(X_train, y_train)

# Train KNN model using training data
knn_model.fit(X_train, y_train)

# Calculate accuracy of Decision Tree model using test data
decision_tree_accuracy = accuracy_score(y_test, decision_tree_model.predict(X_test))

# Calculate accuracy of KNN model using test data
knn_accuracy = accuracy_score(y_test, knn_model.predict(X_test))

########### PREDICT FUNCTION ###########

# Define function that runs when user clicks the Predict button
def predict():
    try:
        # Create a DataFrame from user input values
        passenger = pd.DataFrame([[
            # Get passenger class input and convert to integer
            int(pclass_entry.get()),

            # Convert sex input to numeric (male=0, female=1)
            0 if sex_entry.get().lower() == 'male' else 1,

            # Get age input and convert to integer
            int(age_entry.get()),

            # Get siblings/spouses input and convert to integer
            int(sibsp_entry.get())
        ]], columns=features)

        # Predict survival using Decision Tree model
        decision_tree_result = decision_tree_model.predict(passenger)[0]

        # Predict survival using KNN model
        knn_result = knn_model.predict(passenger)[0]

        # Convert Decision Tree result to readable text
        dt_text = "Survived" if decision_tree_result == 1 else "Did not survive"

        # Convert KNN result to readable text
        knn_text = "Survived" if knn_result == 1 else "Did not survive"

        # Update result label to show BOTH model predictions
        result_label.config(
            text=f"Decision Tree: {dt_text}\nKNN: {knn_text}",
            fg="white",
            font=("Arial", 16, "bold")
        )

    # Catch errors if user enters invalid input
    except:
        # Display error message to user
        result_label.config(text="Invalid input! Please check your values.", fg="yellow")

########### BUILD GUI ###########

# Create main application window
root = tk.Tk()

# Set window title
root.title("Titanic Survival Predictor using ML")

# Set window size
root.geometry("420x480")

# Set background color of window
root.configure(bg="blue")

# Create title label at the top of the window
tk.Label(root, text="Titanic Survival Predictor",
         font=("Arial", 24, "bold"),
         bg="blue", fg="white").pack(pady=20)

# Display Decision Tree model accuracy
tk.Label(root,
         text=f"Decision Tree Accuracy: {decision_tree_accuracy:.1%}",
         font=("Arial", 14),
         bg="blue", fg="white").pack(pady=10)

# Display KNN model accuracy
tk.Label(root,
         text=f"KNN Accuracy: {knn_accuracy:.1%}",
         font=("Arial", 14),
         bg="blue", fg="white").pack(pady=10)

########### INPUT FIELDS ###########

# Create a frame to hold input fields
frame = tk.Frame(root, bg="blue")

# Add frame to window
frame.pack(pady=15)

# Function to create labeled input fields
def add_field(label_text, variable, options=None):
    # Create a row frame for each input field
    row = tk.Frame(frame, bg="blue")

    # Add row to frame
    row.pack(fill="x", pady=4)

    # Create label for the input field
    tk.Label(row,
             text=label_text,
             width=18,
             anchor='w',
             bg="blue",
             font=("Arial", 12),
             fg="white").pack(side=tk.LEFT)

    # If dropdown options exist, create a combobox
    if options:
        ttk.Combobox(row,
                     textvariable=variable,
                     values=options,
                     state="readonly",
                     width=15).pack(side=tk.LEFT)
    else:
        # Otherwise create a text entry field
        tk.Entry(row,
                 textvariable=variable,
                 width=18).pack(side=tk.LEFT)

# Create variables to store user input values

# Default passenger class = 3
pclass_entry = tk.StringVar(value="3")

# Default sex = male
sex_entry = tk.StringVar(value="male")

# Default age = 25
age_entry = tk.StringVar(value="25")

# Default siblings/spouses = 0
sibsp_entry = tk.StringVar(value="0")

# Create input fields using add_field function
add_field("Passenger Class (1-3)", pclass_entry, ["1", "2", "3"])
add_field("Sex", sex_entry, ["male", "female"])
add_field("Age", age_entry)
add_field("Siblings/Spouses", sibsp_entry, ["0","1","2","3","4","5","6","7","8"])

########### BUTTON AND OUTPUT ###########

# Create button to trigger prediction function
tk.Button(root,
          text="Predict Survival",
          command=predict,
          font=("Arial", 14, "bold"),
          fg="green",
          bg="white",
          activebackground="lightgreen").pack(pady=15)

# Create label to display prediction results
result_label = tk.Label(root,
                        text="Enter a passenger profile and click predict",
                        bg="blue",
                        fg="white",
                        font=("Arial", 14))

# Place result label on window
result_label.pack(pady=10)

########### FEATURE IMPORTANCE ###########

# Get feature importance from Decision Tree model and sort it
imp = sorted(zip(features, decision_tree_model.feature_importances_),
             key=lambda x: -x[1])

# Format feature importance into readable text
imp_text = " | ".join(f"{name}: {score:.0%}" for name, score in imp)

# Display feature importance on GUI
tk.Label(root,
         text=f"Top factors for survival: {imp_text}",
         bg="blue",
         fg="white",
         font=("Arial", 10)).pack(pady=10)

########### RUN APP ###########

# Start the GUI event loop so the app runs
root.mainloop()








