import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import pickle
import matplotlib.pyplot as plt


# Function to load data from CSV
def load_data(filename):
    return pd.read_csv(filename)


# Function to preprocess data
def preprocess_data(data, label_encoders=None, fit_encoders=True):
    imputer = SimpleImputer(strategy='mean')
    scaler = StandardScaler()
    if label_encoders is None:
        label_encoders = {}

    X = data.iloc[:, :-1]  # Features (all columns except the last one)
    y = data.iloc[:, -1]  # Target (last column)

    # Handle missing values for X
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    # Encode categorical variables in X
    for col in X.columns:
        if X[col].dtype == 'object':
            if fit_encoders:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col])
                label_encoders[col] = le
            else:
                le = label_encoders[col]
                X[col] = le.transform(X[col])

    # Encode target variable if it's categorical
    if y.dtype == 'object':
        if fit_encoders:
            le = LabelEncoder()
            y = le.fit_transform(y)
            label_encoders['target'] = le
        else:
            le = label_encoders['target']
            y = le.transform(y)

    # Scale numerical features in X
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    return X, y, label_encoders


# Function to split data into train and test sets
def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    return X_train, X_test, y_train, y_test


# Function to train a Random Forest classifier
def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(n_estimators=50, max_depth=2, random_state=0, class_weight='balanced')
    model.fit(X_train, y_train)
    return model


# Function to train a XGBoost classifier
def train_xgboost(X_train, y_train):
    model = XGBClassifier(learning_rate=0.1, n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    return model


# Function to train a Support Vector Machine classifier
def train_svm(X_train, y_train):
    model = SVC(kernel='linear', random_state=0)
    model.fit(X_train, y_train)
    return model


# Function to predict using a trained model
def predict(model, X_test):
    y_pred = model.predict(X_test)
    return y_pred


# Function to calculate accuracy
def calculate_accuracy(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred) * 100
    return accuracy


# Function to perform cross-validation
def cross_validation(model, X, y):
    scores = cross_val_score(model, X, y, cv=5)
    return scores

""
# Function to display feature importance
def display_feature_importance(model, X):
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importance})
        return feature_importance_df.sort_values(by='Importance', ascending=False)
    else:
        return pd.DataFrame(columns=['Feature', 'Importance'])


# Main Streamlit app
def main():
    st.title("Smart Machine Learning")

    # Upload CSV file
    uploaded_file = st.file_uploader("Upload Dataset", type=['csv'])

    if uploaded_file is not None:
        data = load_data(uploaded_file)

        try:
            X, y, label_encoders = preprocess_data(data)
        except Exception as e:
            st.error(f"Error in data preprocessing: {e}")
            return

        st.subheader("Model Selection")
        selected_model = st.selectbox("Choose a Classification Algorithm", ["Random Forest", "XGBoost", "SVM"])

        # Option for cross-validation
        cross_val = st.checkbox("Perform Cross-Validation")

        # Button to generate train & test model
        if st.button("Generate Train & Test Model"):
            X_train, X_test, y_train, y_test = split_data(X, y)

            if selected_model == "Random Forest":
                model = train_random_forest(X_train, y_train)
                if cross_val:
                    scores = cross_validation(model, X, y)
                    st.write(f"Cross-Validation Scores: {scores}")
                    st.write(f"Mean Accuracy: {np.mean(scores) * 100:.2f}%")
            elif selected_model == "XGBoost":
                model = train_xgboost(X_train, y_train)
                if cross_val:
                    scores = cross_validation(model, X, y)
                    st.write(f"Cross-Validation Scores: {scores}")
                    st.write(f"Mean Accuracy: {np.mean(scores) * 100:.2f}%")
            elif selected_model == "SVM":
                model = train_svm(X_train, y_train)
                if cross_val:
                    scores = cross_validation(model, X, y)
                    st.write(f"Cross-Validation Scores: {scores}")
                    st.write(f"Mean Accuracy: {np.mean(scores) * 100:.2f}%")

            # Save train/test data and model to session state
            st.session_state.X_train = X_train
            st.session_state.X_test = X_test
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test
            st.session_state.model = model
            st.session_state.label_encoders = label_encoders

            st.success("Train & Test Model Generated!")

            # Display feature importance for Random Forest
            if selected_model == "Random Forest":
                feature_importance_df = display_feature_importance(model, X_train)
                st.subheader("Feature Importance")
                st.write(feature_importance_df)

        # Button to run selected algorithm
        if st.button("Run Selected Algorithm"):
            if 'model' not in st.session_state:
                st.error("Please generate and select a model first.")
            else:
                X_train = st.session_state.get('X_train', None)
                X_test = st.session_state.get('X_test', None)
                y_train = st.session_state.get('y_train', None)
                y_test = st.session_state.get('y_test', None)

                if X_test is None or y_test is None:
                    st.error("Please generate train & test data first.")
                else:
                    model = st.session_state.model
                    y_train_pred = predict(model, X_train)
                    y_test_pred = predict(model, X_test)

                    train_accuracy = calculate_accuracy(y_train, y_train_pred)
                    test_accuracy = calculate_accuracy(y_test, y_test_pred)

                    st.subheader("Prediction Results")
                    st.write(f"Training Accuracy: {train_accuracy:.2f}%")
                    st.write(f"Testing Accuracy: {test_accuracy:.2f}%")

                    # Display confusion matrix
                    cm = confusion_matrix(y_test, y_test_pred)
                    fig, ax = plt.subplots()
                    cm_display = ConfusionMatrixDisplay(cm).plot(ax=ax)
                    st.pyplot(fig)

                    # Determine if model is underfitting, overfitting, or well-fitted
                    if train_accuracy > test_accuracy + 5:
                        st.warning("The model is overfitting.")
                    elif test_accuracy > train_accuracy + 5:
                        st.warning("The model is underfitting.")
                    else:
                        st.success("The model is well-fitted.")




        # Button to download the trained model
        if st.button("Download Trained Model"):
            if 'model' not in st.session_state:
                st.error("Please generate and train a model first.")
            else:
                model = st.session_state.model
                with open("trained_model.pkl", "wb") as f:
                    pickle.dump(model, f)
                with open("trained_model.pkl", "rb") as f:
                    st.download_button(label="Download Model", data=f, file_name="trained_model.pkl")

        # Button to download training data
        if st.button("Download Training Data"):
            if 'X_train' not in st.session_state or 'y_train' not in st.session_state:
                st.error("Please generate train & test data first.")
            else:
                X_train = st.session_state.X_train
                y_train = st.session_state.y_train
                train_data = pd.concat([X_train, pd.Series(y_train, name='target')], axis=1)
                csv = train_data.to_csv(index=False).encode()
                st.download_button(label="Download Train Data", data=csv, file_name='train_data.csv', mime='text/csv')


if __name__ == "__main__":
    main()
