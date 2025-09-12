from sklearn.ensemble import RandomForestClassifier

class RandomForestClassifierModel:
    """
    A Random Forest Classifier model using scikit-learn.
    
    This class defines an RF model with customizable parameters.
    
    Args:
        n_estimators (int): Number of decision trees in the forest.
        max_depth (int): Maximum depth of the trees.
        min_samples_split (int): Minimum number of samples required to split an internal node.
        min_samples_leaf (int): Minimum number of samples required to be at a leaf node.
        n_classes (int): Number of classes in the classification task.
    """
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1, n_classes=2):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            criterion='gini',  # Gini impurity
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf
        )
        self.n_classes = n_classes
    
    def train(self, X_train, y_train):
        """
        Train the Random Forest classifier.
        
        Args:
            X_train (np.array or torch.Tensor): Training feature data.
            y_train (np.array or torch.Tensor): Training label data.
        """
        self.model.fit(X_train, y_train)
    
    def predict(self, X):
        """
        Predict using the trained Random Forest classifier.
        
        Args:
            X (np.array or torch.Tensor): Feature data to predict.
        
        Returns:
            np.array: Predicted class labels.
        """
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Predict class probabilities using the trained Random Forest classifier.
        
        Args:
            X (np.array or torch.Tensor): Feature data to predict.
        
        Returns:
            np.array: Predicted class probabilities.
        """
        return self.model.predict_proba(X)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the classifier on the test data.
        
        Args:
            X_test (np.array or torch.Tensor): Test feature data.
            y_test (np.array or torch.Tensor): Test label data.
        
        Returns:
            float: Accuracy of the model on the test data.
        """
        return self.model.score(X_test, y_test)

