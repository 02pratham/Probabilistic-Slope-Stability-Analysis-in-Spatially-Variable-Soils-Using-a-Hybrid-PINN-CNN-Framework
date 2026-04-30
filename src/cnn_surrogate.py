# src/cnn_surrogate.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
from datetime import datetime

if torch.cuda.is_available():
    torch.cuda.init()

class UniversalFoSSurrogate(nn.Module):
    def __init__(self):
        super(UniversalFoSSurrogate, self).__init__()
        
        self.register_buffer('S_mean', torch.zeros(3))
        self.register_buffer('S_std', torch.ones(3))
        
        # -------------------------------------------------
        # HEAD 1: Convolutional Feature Extractor (Images)
        # -------------------------------------------------
        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
            )
            
        self.image_features = nn.Sequential(
            conv_block(4, 32),    # Output: 32x32x32
            conv_block(32, 64),   # Output: 16x16x64
            conv_block(64, 128),  # Output: 8x8x128
            conv_block(128, 256), # Output: 4x4x256
            conv_block(256, 256)  # Output: 2x2x256
        )
        
        # FIX: Squeeze the massive 4096 image tensor down to match the scalars
        self.image_bottleneck = nn.Sequential(
            nn.Linear(4096, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True)
        )
        
        # -------------------------------------------------
        # HEAD 2: Dense Feature Extractor (Scalars)
        # -------------------------------------------------
        # FIX: Expand the 3 physics scalars up to match the image dimensions
        self.scalar_features = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 256),
            nn.ReLU(inplace=True)
        )
        
        # -------------------------------------------------
        # FUSION: Combined Regressor
        # -------------------------------------------------
        # Inputs = 256 (Image) + 256 (Physics) = 512 (Perfect 50/50 Balance)
        self.regressor = nn.Sequential(
            nn.Linear(256 + 256, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1), 
            nn.Softplus() 
        )
        
        self._initialize_weights()

    def set_scaler(self, mean, std):
        self.S_mean.copy_(torch.tensor(mean, dtype=torch.float32))
        self.S_std.copy_(torch.tensor(std, dtype=torch.float32))

    def forward(self, x, s):
        s_norm = (s - self.S_mean) / self.S_std
        
        # Process image
        img_out = self.image_features(x)
        img_out = img_out.view(img_out.size(0), -1) 
        img_out = self.image_bottleneck(img_out) # Squeeze to 256
        
        # Process scalars
        scalar_out = self.scalar_features(s_norm) # Expand to 256
        
        # Concatenate side-by-side (256 + 256 = 512)
        combined = torch.cat((img_out, scalar_out), dim=1)
        
        out = self.regressor(combined)
        return out
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

# --- Training Logic ---
def train_surrogate_model(train_dir="data/train", test_dir="data/test", epochs=5000, batch_size=32):
    
    X_train_path = os.path.join(train_dir, "X_train.npy")
    S_train_path = os.path.join(train_dir, "S_train.npy")
    y_train_path = os.path.join(train_dir, "y_train.npy")
    
    X_test_path = os.path.join(test_dir, "X_test.npy")
    S_test_path = os.path.join(test_dir, "S_test.npy")
    y_test_path = os.path.join(test_dir, "y_test.npy")
    
    X_np = np.load(X_train_path)
    S_np = np.load(S_train_path)
    y_np = np.load(y_train_path)
    
    X_tensor = torch.tensor(X_np, dtype=torch.float32)
    S_tensor = torch.tensor(S_np, dtype=torch.float32)
    y_tensor = torch.tensor(y_np, dtype=torch.float32).view(-1, 1)
    
    dataset = TensorDataset(X_tensor, S_tensor, y_tensor)
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UniversalFoSSurrogate().to(device)
    
    mean_S = np.mean(S_np, axis=0)
    std_S = np.std(S_np, axis=0)
    std_S[std_S == 0] = 1.0 
    model.set_scaler(mean_S, std_S)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
    
    patience = 1000 
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    os.makedirs("data/models", exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped_model_path = f"data/models/cnn_model_{timestamp}.pth"
    
    print(f"Starting training on {device}...")
    for epoch in range(1, epochs+1):
        model.train()
        train_loss = 0.0
        
        for batch_X, batch_S, batch_y in train_loader:
            batch_X, batch_S, batch_y = batch_X.to(device), batch_S.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X, batch_S)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_X.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_S, batch_y in val_loader:
                batch_X, batch_S, batch_y = batch_X.to(device), batch_S.to(device), batch_y.to(device)
                outputs = model(batch_X, batch_S)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_X.size(0)
                
        val_loss /= len(val_loader.dataset)
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch:04d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), timestamped_model_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch}! Best Val Loss: {best_val_loss:.4f}")
                break
                
    print("Training complete.")
    
    print("\n" + "="*50)
    print("   Evaluating Model on Unseen Testing Data   ")
    print("="*50)
    
    try:
        model.load_state_dict(torch.load(timestamped_model_path))
        model.eval()
        
        X_test_np = np.load(X_test_path)
        S_test_np = np.load(S_test_path)
        y_test_np = np.load(y_test_path)
        
        X_test_tensor = torch.tensor(X_test_np, dtype=torch.float32).to(device)
        S_test_tensor = torch.tensor(S_test_np, dtype=torch.float32).to(device)
        y_test_tensor = torch.tensor(y_test_np, dtype=torch.float32).view(-1, 1).to(device)
        
        with torch.no_grad():
            test_predictions = model(X_test_tensor, S_test_tensor)
            test_mse = criterion(test_predictions, y_test_tensor).item()
            test_mae = torch.mean(torch.abs(test_predictions - y_test_tensor)).item()
            
        print(f"Total Test Samples : {len(y_test_np)}")
        print(f"Test MSE (Loss)    : {test_mse:.6f}")
        print(f"Test MAE           : {test_mae:.6f}")
        print(f"Archived Model     : {timestamped_model_path}")
        print("="*50)
        
    except FileNotFoundError:
        print("Warning: Test data not found! Could not perform final evaluation.")

    return timestamped_model_path

if __name__ == "__main__":
    train_surrogate_model()