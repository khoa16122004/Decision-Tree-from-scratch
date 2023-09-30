import numpy as np
import pandas as pd


class Node():
    

    def __init__(self, feature=None, threshold=None, left=None, right=None, gain=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.gain = gain
        self.value = value

class DecisionTree():

    def __init__(self, min_samples=2, max_depth=2):
        
        # min_samples của một node khi tiến hành split
        # max_dept : chiều cao tối đa của cây quyết định
        
        self.min_samples = min_samples
        self.max_depth = max_depth

    # So với ngưỡng để split node
    def split_data(self, dataset, feature_index, threshold):
        
        # feature_index: index của feature trong data
        # threshold: dùng để split
        
        
        left_dataset = []
        right_dataset = []
        
        for row in dataset:
            if row[feature_index] <= threshold: # left: True 
                left_dataset.append(row)
            else:
                right_dataset.append(row) # right: False

        left_dataset = np.array(left_dataset)
        right_dataset = np.array(right_dataset)
        return left_dataset, right_dataset

    # Tính entropy của một tập D
    # E(D) =  sigma(p_i * log(1 / p_i)) = -sigma(p_i * log(1 / p_i))
    def entropy(self, y):
        
        # y_lable
        
        entropy = 0
        y_unique = np.unique(y)
        for label in y_unique:
            label_examples = y_unique[y_unique == label]
            p_i = len(label_examples) / len(y) # xác xuất của một biến cố i
            entropy += -p_i * np.log2(p_i) # được đề xuất là đổi log_n theo n field

        return entropy

    
    # Tính information gain 
    # E(D|feature) = -sigma(p_i(D|feature) * log(p_i(D|feature))) 
    # I_g(D) = E(D) - E(D|A)
    def information_gain(self, parent, left, right):
       

        # Có thể không cần tính information gain mà sẽ so sánh mean Entropy
        information_gain = 0
        parent_entropy = self.entropy(parent)
        weight_left = len(left) / len(parent)
        weight_right= len(right) / len(parent)
        entropy_left, entropy_right = self.entropy(left), self.entropy(right)
        weighted_entropy = weight_left * entropy_left + weight_right * entropy_righ
        information_gain = parent_entropy - weighted_entropy # Option
        return information_gain

   # Tìm feautre có information_gain cao nhất
    def best_split(self, dataset, num_samples, num_features):
       
       
        best_split = {'gain':- 1, 'feature': None, 'threshold': None}
        for feature_index in range(num_features):
            feature_values = dataset[:, feature_index]
            
            # Lưu ý nên số hóa các cột không rời rạc
            thresholds = np.unique(feature_values)
            
            # pp này chọn threshold bằng cách duyệt qua tất cả các value của một feature và tìm value nào cho entropy thấp nhất hay information_gain cao nhất
            for threshold in thresholds:
                left_dataset, right_dataset = self.split_data(dataset, feature_index, threshold)
                if len(left_dataset) and len(right_dataset):
                    y, left_y, right_y = dataset[:, -1], left_dataset[:, -1], right_dataset[:, -1]
                    information_gain = self.information_gain(y, left_y, right_y)
                    if information_gain > best_split["gain"]:                        
                        # lưu best_split
                        best_split["feature"] = feature_index
                        best_split["threshold"] = threshold
                        best_split["left_dataset"] = left_dataset
                        best_split["right_dataset"] = right_dataset
                        best_split["gain"] = information_gain



            # Có thể dùng mean để làm threshold
        
            ## threshold = np.mean(feature_values) 
            ## left_dataset, right_dataset = self.split_data(dataset, feature_index, threshold)
            ##     if len(left_dataset) and len(right_dataset):
            ##         y, left_y, right_y = dataset[:, -1], left_dataset[:, -1], right_dataset[:, -1]
            ##         information_gain = self.information_gain(y, left_y, right_y)
            ##         if information_gain > best_split["gain"]:                        
            ##             best_split["feature"] = feature_index
            ##             best_split["threshold"] = threshold
            ##             best_split["right_dataset"] = right_dataset
            ##             best_split["gain"] = information_gain

        
        return best_split

    
    def calculate_leaf_value(self, y):
        
        # y_lable
        
        y = list(y)
        most_occuring_value = max(y, key=y.count)
        return most_occuring_value
    
    
    def build_tree(self, dataset, current_depth=0):
        
        X, y = dataset[:, :-1], dataset[:, -1]
        n_samples, n_features = X.shape
        if n_samples >= self.min_samples and current_depth <= self.max_depth: # điều kiện dừng
            best_split = self.best_split(dataset, n_samples, n_features)
            if best_split["gain"]:
                left_node = self.build_tree(best_split["left_dataset"], current_depth + 1)
                right_node = self.build_tree(best_split["right_dataset"], current_depth + 1)
                return Node(best_split["feature"], best_split["threshold"],
                            left_node, right_node, best_split["gain"])

        leaf_value = self.calculate_leaf_value(y)
        return Node(value=leaf_value)

    # train
    def fit(self, X, y):
      
        dataset = np.concatenate((X, y), axis=1)  
        self.root = self.build_tree(dataset)
        
    # 
    def predict(self, X):
       
        predictions = []
        for x in X:
            prediction = self.make_prediction(x, self.root)
            predictions.append(prediction)
        np.array(predictions)
        return predictions
    
    def make_prediction(self, x, node):
        
        if node.value != None: 
            return node.value
        else:
            feature = x[node.feature]
            if feature <= node.threshold:
                return self.make_prediction(x, node.left)
            else:
                return self.make_prediction(x, node.right)
