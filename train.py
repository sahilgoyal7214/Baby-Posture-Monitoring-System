import os
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import validation_curve
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV


class InfantPoseCustomDataset():
    def __init__(self, dataset_dir, transform=None):
        self.dataset_dir = dataset_dir
        self.transform = transform
        self.fold_names = ["0", "1", "2", "3", "4"]
        self.split_names = ["train", "val", "test"]

        # Load data for all folds and splits
        self.data = []
        for fold_name in self.fold_names:
            fold_dir = os.path.join(self.dataset_dir, fold_name)
            for split_name in self.split_names:
                split_dir = os.path.join(fold_dir, split_name)

                # Check if data files exist for the current split
                if os.path.exists(split_dir):
                    data_point = {
                        "uncovered": np.load(os.path.join(split_dir, "uncovered.npy")),
                        "cover1": np.load(os.path.join(split_dir, "cover1.npy")),
                        "cover2": np.load(os.path.join(split_dir, "cover2.npy")),
                        "joints": np.load(os.path.join(split_dir, "joints.npy")),
                        "labels": np.load(os.path.join(split_dir, "labels.npy")),
                    }

                    self.data.append(data_point)

    def __getitem__(self, idx):
        sample = self.data[idx]

        return sample

    def __len__(self):
        # Return the total number of samples in the dataset
        return len(self.data)

dataset_dir = r"/content/drive/MyDrive/dataverse_files/SMaL-224"
custom_dataset = InfantPoseCustomDataset(dataset_dir)

with open("infant_pose_dataset.pkl", "wb") as f:
   pickle.dump(custom_dataset, f)

with open("infant_pose_dataset.pkl", "rb") as f:
      custom_dataset = pickle.load(f)


# Extract HOG features from image data
def extract_hog_features(images):
    hog_features = []
    for image in images:
        fd, _ = hog(image[:, :, 0], pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
        hog_features.append(fd)
    return np.array(hog_features)

def train_and_evaluate_svm(X_train, y_train, X_val, y_val):
    # Initialize an SVM classifier
    svm_classifier = SVC(kernel='linear')

    # Train the SVM classifier
    SVM = svm_classifier.fit(X_train, y_train)

    # Predict on the validation set
    y_val_pred = svm_classifier.predict(X_val)

    # Calculate accuracy
    val_accuracy = accuracy_score(y_val, y_val_pred)

    return svm_classifier, val_accuracy ,SVM

def train_and_evaluate_random_forest(X_train, y_train, X_val, y_val):
    # Initialize a Random Forest classifier
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train the Random Forest classifier
    RF = rf_classifier.fit(X_train, y_train)

    # Predict on the validation set
    y_val_pred = rf_classifier.predict(X_val)

    # Calculate accuracy
    val_accuracy = accuracy_score(y_val, y_val_pred)

    return rf_classifier, val_accuracy ,RF

def train_and_evaluate_knn(X_train, y_train, X_val, y_val):
    # Initialize a k-NN classifier
    knn_classifier = KNeighborsClassifier(n_neighbors=3)

    # Train the k-NN classifier
    Knn=knn_classifier.fit(X_train, y_train)

    # Predict on the validation set
    y_val_pred = knn_classifier.predict(X_val)
    # Calculate accuracy
    val_accuracy = accuracy_score(y_val, y_val_pred)

    parameter_range = np.arange(1, 10, 1)

    train_score, val_score = validation_curve(KNeighborsClassifier(),X_train, y_train,
                                           param_name="n_neighbors",
                                           param_range=parameter_range,
                                           cv=5, scoring="accuracy")

    # Calculating mean and standard deviation of training score
    mean_train_score = np.mean(train_score, axis=1)
    std_train_score = np.std(train_score, axis=1)

    # Calculating mean and standard deviation of valing score
    mean_val_score = np.mean(val_score, axis=1)
    std_test_score = np.std(val_score, axis=1)

    # Plot mean accuracy scores for training and valing scores
    plt.plot(parameter_range, mean_train_score,
            label="Training Score", color='b')
    plt.plot(parameter_range, mean_val_score,
            label="Cross Validation Score", color='g')

    # Creating the plot
    plt.title("Validation Curve with KNN Classifier")
    plt.xlabel("Number of Neighbours")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.legend(loc='best')
    plt.show()

    return knn_classifier, val_accuracy , Knn

def tune_svm_hyperparameters(X_train, y_train):
    # Define the hyperparameter grid to search
    param_grid = {
        'C': [0.1, 1, 10],  # Regularization strength
        'kernel': ['linear', 'rbf', 'poly'],  # Kernel type
    }

    # Initialize an SVM classifier
    svm_classifier = SVC()

    # Perform grid search with cross-validation
    grid_search = GridSearchCV(svm_classifier, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    # Get the best hyperparameters
    best_params = grid_search.best_params_
    best_accuracy = grid_search.best_score_

    return best_params, best_accuracy

def tune_random_forest_hyperparameters(X_train, y_train):
    # Define the hyperparameter grid to search
    param_grid = {
        'n_estimators': [50, 100, 200],  # Number of trees in the forest
        'max_depth': [None, 10, 20],  # Maximum depth of trees
        'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split a node
    }

    # Initialize a Random Forest classifier
    rf_classifier = RandomForestClassifier(random_state=42)

    # Perform grid search with cross-validation
    grid_search = GridSearchCV(rf_classifier, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    # Get the best hyperparameters
    best_params = grid_search.best_params_
    best_accuracy = grid_search.best_score_

    return best_params, best_accuracy

def tune_knn_hyperparameters(X_train, y_train):
    # Define the hyperparameter grid to search
    param_grid = {
        'n_neighbors': [3, 5, 7],  # Number of neighbors
        'weights': ['uniform', 'distance'],  # Weight function used in prediction
    }

    # Initialize a k-NN classifier
    knn_classifier = KNeighborsClassifier()

    # Perform grid search with cross-validation
    grid_search = GridSearchCV(knn_classifier, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    # Get the best hyperparameters
    best_params = grid_search.best_params_
    best_accuracy = grid_search.best_score_

    return best_params, best_accuracy

def evaluate_svm_on_test_set(svm_classifier, X_test, y_test):
    # Predict on the test set
    y_test_pred = svm_classifier.predict(X_test)

    # Calculate accuracy on the test set
    test_accuracy = accuracy_score(y_test, y_test_pred)

    return test_accuracy

def evaluate_random_forest_on_test_set(rf_classifier, X_test, y_test):
    # Predict on the test set
    y_test_pred = rf_classifier.predict(X_test)

    # Calculate accuracy on the test set
    test_accuracy = accuracy_score(y_test, y_test_pred)

    return test_accuracy

def evaluate_knn_on_test_set(knn_classifier, X_test, y_test):
    # Predict on the test set
    y_test_pred = knn_classifier.predict(X_test)

    # Calculate accuracy on the test set
    test_accuracy = accuracy_score(y_test, y_test_pred)

    return test_accuracy


# Iterate through batches of data during training
for idx, batch in enumerate(custom_dataset):
    if idx >= len(custom_dataset):
      break
    uncovered_images = batch["uncovered"]
    cover1_images = batch["cover1"]
    cover2_images = batch["cover2"]
    joints = batch["joints"]
    labels = batch["labels"]

    # Flatten the joints data to 2D
    num_samples, num_joints, num_coords = joints.shape
    flattened_joints = joints.reshape(num_samples, -1)

    # Standardize joint data
    scaler = StandardScaler()
    scaled_joint_data = scaler.fit_transform(flattened_joints)

    # Reshape it back to the original shape if needed
    #scaled_joint_data = scaled_joint_data.reshape(num_samples, num_joints, num_coords)


    # # Standardize joint data (optional, depending on your data)
    scaler = StandardScaler()
    scaled_joint_data = scaler.fit_transform(joints)

    # Combine features (HOG + joint coordinates)
    hog_features_uncovered = np.hstack((extract_hog_features(uncovered_images), scaled_joint_data))
    hog_features_cover1 = np.hstack((extract_hog_features(cover1_images), scaled_joint_data))
    hog_features_cover2 = np.hstack((extract_hog_features(cover2_images), scaled_joint_data))


    labels_uncovered = labels
    labels_cover1 = labels
    labels_cover2 = labels

    # Define the train-validation-test split ratios (adjust as needed)
    train_ratio = 0.6  # 60% of data for training
    val_ratio = 0.2    # 20% of data for validation
    test_ratio = 0.2   # 20% of data for testing

    # Perform the train-validation-test split for each modality
    X_train_uncovered, X_temp_uncovered, y_train_uncovered, y_temp_uncovered = train_test_split(
        hog_features_uncovered, labels_uncovered, test_size=(val_ratio + test_ratio), random_state=42)

    X_val_uncovered, X_test_uncovered, y_val_uncovered, y_test_uncovered = train_test_split(X_temp_uncovered, y_temp_uncovered, test_size=(test_ratio / (val_ratio + test_ratio)), random_state=42)

    # Repeat the same split process for cover1 and cover2 modalities
    X_train_cover1, X_temp_cover1, y_train_cover1, y_temp_cover1 = train_test_split(hog_features_cover1, labels_cover1, test_size=(val_ratio + test_ratio), random_state=42)

    X_val_cover1, X_test_cover1, y_val_cover1, y_test_cover1 = train_test_split(X_temp_cover1, y_temp_cover1, test_size=(test_ratio / (val_ratio + test_ratio)), random_state=42)

    X_train_cover2, X_temp_cover2, y_train_cover2, y_temp_cover2 = train_test_split(hog_features_cover2, labels_cover2, test_size=(val_ratio + test_ratio), random_state=42)

    X_val_cover2, X_test_cover2, y_val_cover2, y_test_cover2 = train_test_split(X_temp_cover2, y_temp_cover2, test_size=(test_ratio / (val_ratio + test_ratio)), random_state=42)

    svm_classifier_uncovered, val_accuracy_svm_uncovered, svm_uncovered  = train_and_evaluate_svm(X_train_uncovered, y_train_uncovered, X_val_uncovered, y_val_uncovered)
    rf_classifier_uncovered, val_accuracy_rf_uncovered,rf_uncovered = train_and_evaluate_random_forest(X_train_uncovered, y_train_uncovered, X_val_uncovered, y_val_uncovered)
    knn_classifier_uncovered, val_accuracy_knn_uncovered, knn_uncovered = train_and_evaluate_knn(X_train_uncovered, y_train_uncovered, X_val_uncovered, y_val_uncovered)

    svm_classifier_cover1, val_accuracy_svm_cover1,svm_cover1 = train_and_evaluate_svm(X_train_cover1, y_train_cover1, X_val_cover1, y_val_cover1)
    rf_classifier_cover1, val_accuracy_rf_cover1 ,rf_cover1= train_and_evaluate_random_forest(X_train_cover1, y_train_cover1, X_val_cover1, y_val_cover1)
    knn_classifier_cover1, val_accuracy_knn_cover1 ,knn_cover1= train_and_evaluate_knn(X_train_cover1, y_train_cover1, X_val_cover1, y_val_cover1)

    svm_classifier_cover2, val_accuracy_svm_cover2,svm_cover2 = train_and_evaluate_svm(X_train_cover2, y_train_cover2, X_val_cover2, y_val_cover2)
    rf_classifier_cover2, val_accuracy_rf_cover2 ,rf_cover2= train_and_evaluate_random_forest(X_train_cover2, y_train_cover2, X_val_cover2, y_val_cover2)
    knn_classifier_cover2, val_accuracy_knn_cover2 , knn_cover2 = train_and_evaluate_knn(X_train_cover2, y_train_cover2, X_val_cover2, y_val_cover2)

    print(f'Validation Accuracy (SVM) - Uncovered: {val_accuracy_svm_uncovered:.2f}')
    print(f'Validation Accuracy (Random Forest) - Uncovered: {val_accuracy_rf_uncovered:.2f}')
    print(f'Validation Accuracy (k-NN) - Uncovered: {val_accuracy_knn_uncovered:.2f}')
    print("---------------------------------------------------------------------------------")

    print(f'Validation Accuracy (SVM) - cover1: {val_accuracy_svm_cover1:.2f}')
    print(f'Validation Accuracy (Random Forest) - cover1: {val_accuracy_rf_cover1:.2f}')
    print(f'Validation Accuracy (k-NN) - cover1: {val_accuracy_knn_cover1:.2f}')
    print("---------------------------------------------------------------------------------")

    print(f'Validation Accuracy (SVM) - cover2: {val_accuracy_svm_cover2:.2f}')
    print(f'Validation Accuracy (Random Forest) - cover2: {val_accuracy_rf_cover2:.2f}')
    print(f'Validation Accuracy (k-NN) - cover2: {val_accuracy_knn_cover2:.2f}')
    print("---------------------------------------------------------------------------------")
    print("")
    print("================================================================================================")


    # Hyperparameter tuning for the uncovered modality
    best_params_svm_uncovered, best_accuracy_svm_uncovered = tune_svm_hyperparameters(X_train_uncovered, y_train_uncovered)
    best_params_rf_uncovered, best_accuracy_rf_uncovered = tune_random_forest_hyperparameters(X_train_uncovered, y_train_uncovered)
    best_params_knn_uncovered, best_accuracy_knn_uncovered = tune_knn_hyperparameters(X_train_uncovered, y_train_uncovered)

    # Print the best hyperparameters and their corresponding accuracies
    print('Best SVM Hyperparameters - Uncovered:', best_params_svm_uncovered)
    print('Best SVM Accuracy - Uncovered:', best_accuracy_svm_uncovered)
    print("---------------------------------------------------------------------------------")

    print('Best Random Forest Hyperparameters - Uncovered:', best_params_rf_uncovered)
    print('Best Random Forest Accuracy - Uncovered:', best_accuracy_rf_uncovered)
    print("---------------------------------------------------------------------------------")

    print('Best k-NN Hyperparameters - Uncovered:', best_params_knn_uncovered)
    print('Best k-NN Accuracy - Uncovered:', best_accuracy_knn_uncovered)
    print("---------------------------------------------------------------------------------")
    print("")
    print("================================================================================================")

# Repeat the same process for cover1 and cover2 modalities
    best_params_svm_cover1, best_accuracy_svm_cover1 = tune_svm_hyperparameters(X_train_cover1, y_train_cover1)
    best_params_rf_cover1, best_accuracy_rf_cover1 = tune_random_forest_hyperparameters(X_train_cover1, y_train_cover1)
    best_params_knn_cover1, best_accuracy_knn_cover1 = tune_knn_hyperparameters(X_train_cover1, y_train_cover1)

    # Print the best hyperparameters and their corresponding accuracies
    print('Best SVM Hyperparameters - cover1:', best_params_svm_cover1)
    print('Best SVM Accuracy - cover1:', best_accuracy_svm_cover1)
    print("---------------------------------------------------------------------------------")

    print('Best Random Forest Hyperparameters - cover1:', best_params_rf_cover1)
    print('Best Random Forest Accuracy - cover1:', best_accuracy_rf_cover1)
    print("---------------------------------------------------------------------------------")

    print('Best k-NN Hyperparameters - cover1:', best_params_knn_cover1)
    print('Best k-NN Accuracy - cover1:', best_accuracy_knn_cover1)
    print("---------------------------------------------------------------------------------")
    print("")
    print("================================================================================================")

    best_params_svm_cover2, best_accuracy_svm_cover2 = tune_svm_hyperparameters(X_train_cover2, y_train_cover2)
    best_params_rf_cover2, best_accuracy_rf_cover2 = tune_random_forest_hyperparameters(X_train_cover2, y_train_cover2)
    best_params_knn_cover2, best_accuracy_knn_cover2 = tune_knn_hyperparameters(X_train_cover2, y_train_cover2)

    # Print the best hyperparameters and their corresponding accuracies
    print('Best SVM Hyperparameters - cover2:', best_params_svm_cover2)
    print('Best SVM Accuracy - cover2:', best_accuracy_svm_cover2)
    print("---------------------------------------------------------------------------------")

    print('Best Random Forest Hyperparameters - cover2:', best_params_rf_cover2)
    print('Best Random Forest Accuracy - cover2:', best_accuracy_rf_cover2)
    print("---------------------------------------------------------------------------------")

    print('Best k-NN Hyperparameters - cover2:', best_params_knn_cover2)
    print('Best k-NN Accuracy - cover2:', best_accuracy_knn_cover2)
    print("---------------------------------------------------------------------------------")
    print("")
    print("================================================================================================")

        # Evaluate SVM on the test set for the uncovered modality
    test_accuracy_svm_uncovered = evaluate_svm_on_test_set(svm_classifier_uncovered, X_test_uncovered, y_test_uncovered)
    print(f'Test Accuracy (SVM) - Uncovered: {test_accuracy_svm_uncovered:.2f}')
    print("---------------------------------------------------------------------------------")

    # Evaluate Random Forest on the test set for the uncovered modality
    test_accuracy_rf_uncovered = evaluate_random_forest_on_test_set(rf_classifier_uncovered, X_test_uncovered, y_test_uncovered)
    print(f'Test Accuracy (Random Forest) - Uncovered: {test_accuracy_rf_uncovered:.2f}')
    print("---------------------------------------------------------------------------------")

    # Evaluate k-NN on the test set for the uncovered modality
    test_accuracy_knn_uncovered = evaluate_knn_on_test_set(knn_classifier_uncovered, X_test_uncovered, y_test_uncovered)
    print(f'Test Accuracy (k-NN) - Uncovered: {test_accuracy_knn_uncovered:.2f}')
    print("---------------------------------------------------------------------------------")
    print("")
    print("================================================================================================")
    # Repeat the same process for cover1 and cover2 modalities
    test_accuracy_svm_cover1 = evaluate_svm_on_test_set(svm_classifier_cover1, X_test_cover1, y_test_cover1)
    print(f'Test Accuracy (SVM) - cover1: {test_accuracy_svm_cover1:.2f}')
    print("---------------------------------------------------------------------------------")

    # Evaluate Random Forest on the test set for the cover1 modality
    test_accuracy_rf_cover1 = evaluate_random_forest_on_test_set(rf_classifier_cover1, X_test_cover1, y_test_cover1)
    print(f'Test Accuracy (Random Forest) - cover1: {test_accuracy_rf_cover1:.2f}')
    print("---------------------------------------------------------------------------------")

    # Evaluate k-NN on the test set for the cover1 modality
    test_accuracy_knn_cover1 = evaluate_knn_on_test_set(knn_classifier_cover1, X_test_cover1, y_test_cover1)
    print(f'Test Accuracy (k-NN) - cover1: {test_accuracy_knn_cover1:.2f}')
    print("---------------------------------------------------------------------------------")
    print("")
    print("================================================================================================")

    test_accuracy_svm_cover2 = evaluate_svm_on_test_set(svm_classifier_cover2, X_test_cover2, y_test_cover2)
    print(f'Test Accuracy (SVM) - cover2: {test_accuracy_svm_cover2:.2f}')
    print("---------------------------------------------------------------------------------")

    # Evaluate Random Forest on the test set for the cover2 modality
    test_accuracy_rf_cover2 = evaluate_random_forest_on_test_set(rf_classifier_cover2, X_test_cover2, y_test_cover2)
    print(f'Test Accuracy (Random Forest) - cover2: {test_accuracy_rf_cover2:.2f}')
    print("---------------------------------------------------------------------------------")

    # Evaluate k-NN on the test set for the cover2 modality
    test_accuracy_knn_cover2 = evaluate_knn_on_test_set(knn_classifier_cover2, X_test_cover2, y_test_cover2)
    print(f'Test Accuracy (k-NN) - cover2: {test_accuracy_knn_cover2:.2f}')
    print("---------------------------------------------------------------------------------")
    print("")
    print("================================================================================================")
    print("*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-")

    



