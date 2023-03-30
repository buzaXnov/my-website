import torch
import torch.nn as nn
import numpy as np
from itertools import combinations

# # Device configuration
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_size) 
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.3)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.3)
        self.l3 = nn.Linear(hidden_size, hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(p=0.3)
        self.l4 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        x = x.to(torch.float32)
        out = self.l1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout1(out)
        out = self.l2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        out = self.l3(out)
        out = self.bn3(out)
        out = self.relu3(out)
        out = self.dropout3(out)
        out = self.l4(out)
        
        return out
    
    def predict(self, x):
        with torch.no_grad():
            out = self.forward(x)
            _, predicted = torch.max(out.data, 1)
            return predicted

def load_model():
    # Hyper-parameters 
    input_size = 40 # 40 points from the normalized points
    hidden_size = 128 
    num_classes = 5 # 5 symbols

    PATH = 'utils/symbol-recognizer.pt'

    checkpoint = torch.load(PATH)

    model = NeuralNet(input_size, hidden_size, num_classes).eval()
    model.load_state_dict(checkpoint['model_state_dict'])

    return model

def preprocess_symbol_vector(points):    
    
    M = 20

    # You do not have labels here beacuse this is used for prediction
    
    # Removing information about location of the symbol on the canvas
    symb_mean = sum( [el[0]+el[1] for el in points] ) / len(points)   # Arithmetic mean value for the current symbol.
    new_p = [ (p_[0]-symb_mean, p_[1]-symb_mean) for p_ in points]
    
    # Scaling the symbol
    scaled_points = list()
    
    square_distance = lambda x,y: sum([(xi-yi)**2 for xi, yi in zip(x,y)])    
    max_square_dist = 0
    for p in combinations(new_p, 2):
        dist = square_distance(*p)
        if dist > max_square_dist:
            max_square_dist = dist
            max_pair = p
                
    mx = max( [abs(el[0]) for el in max_pair] )
    my = max( [abs(el[1]) for el in max_pair] )
    m = max(mx, my)

    scaled_points = [(el[0]/m, el[1]/m) for el in new_p]
    
    representative_points = list()
    D = 0
    for i in range(0, len(scaled_points)-1):
        D += square_distance(x=scaled_points[i], y=scaled_points[i+1])
        
    for k in range(M):
        dist_ = (k * D) / (M - 1)
        all_distances = [ abs( dist_ - square_distance(x=p_, y=scaled_points[0]) ) for p_ in scaled_points]  # Subtract the distance of each point from the start from the representative distance and the smallest value is the clossest
        closset_point = scaled_points[ np.argmin( np.array(all_distances) ) ]
        
        representative_points.append( closset_point )
    
    array = []
    for x in representative_points:
        array.append(x[0])
        array.append(x[1])
        
    array = np.array(array)
    
    return array

# model = NeuralNet(input_size, hidden_size, num_classes).to(device)

# # Loss and optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# # Define the scheduler
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)