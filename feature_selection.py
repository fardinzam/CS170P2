import numpy as np

def load_data(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                data.append([float(x) for x in line.strip().split()])
    return np.array(data)

def euclidean_distance(x, y):
    return np.sqrt(np.sum(np.square(x - y)))

def forward_selection(data):
    num_features = data.shape[1] - 1  # exclude class
    selected_features = [] # start empty
    best_accuracy = 0.0
    best_feature_subset = []

    print("\nBeginning forward selection search.\n")

    for _ in range(1, num_features + 1): #add feature per iteration
        add_feature = None
        curr_best_accuracy = 0.0

        for feature in range(1, num_features + 1): # try to add each feature not already selected
            if feature in selected_features: # if added then skip
                continue
        
            curr_features = selected_features + [feature]
            accuracy = leave_one_out(data, curr_features)

            subset_str = "{" + ", ".join(map(str, curr_features)) + "}"
            print(f"\tUsing feature(s) {subset_str} accuracy is {accuracy:.1f}%")
            
            if accuracy > curr_best_accuracy: # find highest accuracy feature and add it
                curr_best_accuracy = accuracy
                add_feature = feature
        
        if add_feature:
            selected_features = selected_features + [add_feature]
            print(f"\nFeature set {add_feature} was best, accuracy is {curr_best_accuracy:.1f}%\n")
            
            if curr_best_accuracy > best_accuracy: # check if new accuracy is better
                best_accuracy = curr_best_accuracy
                best_feature_subset = [i for i in selected_features]
            else:
                print("(Warning, accuracy has decreased! Continuing search in case of local maxima)\n")
        
    print(f"Finish search! The best feature subset is {best_feature_subset}, which has an accuracy of {best_accuracy:.1f}%\n")

def backward_elimination(data):
    num_features = data.shape[1] - 1  # exclude class
    curr_features = list(range(1, num_features + 1))  # start with all features
    best_accuracy = leave_one_out(data, curr_features)
    best_feature_subset = [i for i in curr_features]

    print(f"\nStarting with all features, accuracy is {best_accuracy}%")
    print("Beginning backward elimination search\n")

    for _ in range(num_features, 1, -1):
        remove_feature = None
        curr_best_accuracy = 0

        for feature in curr_features:
            temp_features = [f for f in curr_features if f != feature] # copy of current set minus the feature being tested
            accuracy = leave_one_out(data, temp_features)

            subset_str = "{" + ", ".join(map(str, temp_features)) + "}"
            print(f"\tUsing feature(s) {subset_str} accuracy is {accuracy}%")

            if accuracy > curr_best_accuracy:
                curr_best_accuracy = accuracy
                remove_feature = feature
        
        if remove_feature is not None:
            curr_features.remove(remove_feature)
            print(f"\nFeature set {remove_feature} was removed, accuracy is {curr_best_accuracy}%\n")

            if curr_best_accuracy > best_accuracy:
                best_accuracy = curr_best_accuracy
                best_feature_subset = [i for i in curr_features]
            else:
                print("(Warning, Accuracy has decreased! Continuing search in case of local maxima)\n")
    
    print(f"Finished search!! The best feature subset is {best_feature_subset}, which has an accuracy of {best_accuracy}%\n")


def leave_one_out(data, selected_features):
    accuracy = []
    for i in range(data.shape[0]):
        # select one instance as test and take only the selected features
        test_instance = data[i, np.array([0] + selected_features)]
        # select every instance except i as train and take only the selected features
        train_instances = np.vstack([data[:i], data[i+1:]]) # remove test instance
        train_instances = train_instances[:, np.array([0] + selected_features)] # take only selected features
        predicted_class = nearest_neighbor(train_instances, test_instance)
        actual_class = test_instance[0]
        accuracy.append(int(predicted_class == actual_class))
    return round(100 * ( sum(accuracy) / len(accuracy) ), 1)

def nearest_neighbor(training_data, test_instance):
    nearest = None
    min_dist = float('inf')
    for instance in training_data:
        dist = euclidean_distance(instance[1:], test_instance[1:])
        if dist < min_dist:
            min_dist = dist
            nearest = instance
    return nearest[0]  # return the class label of the nearest neighbor


# for leave-one-out 
def main():
    print("Welcome to Feature Selection Algorithm (by Fardin Zaman)\n")
    file_path = input("Type in the name of the file to load: ")
    data = load_data(file_path)

    print(f"\nThis dataset has {data.shape[1] - 1} features (excluding the class label), with {data.shape[0]} instances.\n")

    print("Type the number of the algorithm you want to run:")
    print("1) Forward Selection")
    print("2) Backward Elimination")
    choice = input()

    if choice == '1':
        forward_selection(data)
    elif choice == '2':
        backward_elimination(data)
    else:
        print("Invalid choice. Exiting.")

if __name__ == "__main__":
    main()

# On small dataset 84 the error rate can be 0.938 when using only features 2  4  5
# On large dataset 90 the error rate can be 0.947 when using only features 2  26  32
