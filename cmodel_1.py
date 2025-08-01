import math
import random
import re
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    OrdinalEncoder,
    OneHotEncoder,
    StandardScaler,
    LabelEncoder,
)
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# ---------------------------------------------------------------------
# 1. DATA --------------------------------------------------------------
# ---------------------------------------------------------------------
csv_path = r"C:\Users\user\Documents\neural_network\flood_risk_dataset_final.csv"  # <- adapt if necessary
df = pd.read_csv(csv_path)

# 1.1 Clean & transform numerical outliers
for col in ["Elevation", "Rainfall"]:
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    df[col] = df[col].clip(lower=max(q1 - 1.5 * iqr, 0), upper=q3 + 1.5 * iqr)
    mn = df[col].min()
    df[col] = np.log1p(df[col] + abs(mn) + 1e-6)

# 1.2 Feature groups ---------------------------------------------------
text_feature = "ProximityToWaterBody"
numerical_features = ["Elevation", "Rainfall"]
ordinal_features = [
    "Vegetation",
    "Urbanization",
    "Drainage",
    "Slope",
    "StormFrequency",
    "Deforestation",
    "Infrastructure",
    "Encroachment",
    "Season",
]
onehot_features = ["Soil", "Wetlands"]
label_column = "FloodRisk"

# 1.3 Fill NA ----------------------------------------------------------
for col in ordinal_features + onehot_features + [text_feature]:
    df[col] = df[col].fillna("Missing")

# 1.4 Encode y ---------------------------------------------------------
label_enc = LabelEncoder()
df[label_column] = label_enc.fit_transform(df[label_column])

# 1.5 Train/val/test split --------------------------------------------
train_df, tmp_df = train_test_split(
    df, test_size=0.2, stratify=df[label_column], random_state=42
)
val_df, test_df = train_test_split(
    tmp_df, test_size=0.5, stratify=tmp_df[label_column], random_state=42
)

# 1.6 Encoders ---------------------------------------------------------
ord_enc = OrdinalEncoder(
    handle_unknown="use_encoded_value", unknown_value=-1
)
text_enc = OrdinalEncoder(
    handle_unknown="use_encoded_value", unknown_value=-1
)
onehot_enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
scaler = StandardScaler()

ord_enc.fit(train_df[ordinal_features])
text_enc.fit(train_df[[text_feature]])
onehot_enc.fit(train_df[onehot_features])
scaler.fit(train_df[numerical_features])

# Store valid categories for validation
valid_categories = {}
for i, col in enumerate(ordinal_features):
    valid_categories[col] = list(ord_enc.categories_[i])
for i, col in enumerate(onehot_features):
    valid_categories[col] = list(onehot_enc.categories_[i])
valid_categories[text_feature] = list(text_enc.categories_[0])

# ---------------------------------------------------------------------
# 2. TORCH DATASET -----------------------------------------------------
# ---------------------------------------------------------------------
class FloodDataset(Dataset):
    def __init__(self, pdf, train=True):
        self.text_code = (
            text_enc.transform(pdf[[text_feature]]).astype(np.int64).squeeze(1)
        )
        self.ordinals = ord_enc.transform(pdf[ordinal_features]).astype(np.int64)
        self.onehots = onehot_enc.transform(pdf[onehot_features]).astype(np.float32)
        self.numerical = scaler.transform(pdf[numerical_features]).astype(np.float32)
        self.labels = pdf[label_column].astype(np.int64).values
        self.train = train

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "text": torch.tensor(self.text_code[idx], dtype=torch.long),
            "ord": torch.tensor(self.ordinals[idx], dtype=torch.long),
            "onehot": torch.tensor(self.onehots[idx], dtype=torch.float),
            "num": torch.tensor(self.numerical[idx], dtype=torch.float),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }

train_set = FloodDataset(train_df, train=True)
val_set = FloodDataset(val_df, train=False)
test_set = FloodDataset(test_df, train=False)

# 2.1 Weighted sampler to fight imbalance -----------------------------
class_counts = np.bincount(train_set.labels)
class_weights = 1.0 / (class_counts + 1e-4)
sample_weights = class_weights[train_set.labels]
sampler = WeightedRandomSampler(
    weights=torch.tensor(sample_weights, dtype=torch.double),
    num_samples=len(sample_weights),
    replacement=True,
)

# 2.2 DataLoaders ------------------------------------------------------
train_loader = DataLoader(train_set, batch_size=256, sampler=sampler)
val_loader = DataLoader(val_set, batch_size=512, shuffle=False)
test_loader = DataLoader(test_set, batch_size=512, shuffle=False)

# ---------------------------------------------------------------------
# 3. MODEL -------------------------------------------------------------
# ---------------------------------------------------------------------
def make_embedding(num_categories: int, dim: int) -> nn.Embedding:
    emb = nn.Embedding(num_embeddings=num_categories + 1, embedding_dim=dim)
    nn.init.xavier_uniform_(emb.weight)#initialize weights using xavier uniform initialization
    return emb
    #initialize encoders and scalers
ord_enc=OrdinalEncoder(handle_unknown="use_encoded_value",unknown_value=-2)
text_enc=OrdinalEncoder(handle_unknown="use_encoded_value",unknown_value=-2)
onehot_enc=OneHotEncoder(sparse_output=False,handle_unknown="ignore")
scaler=StandardScaler()
#fit encoders and scaler on training data
ord_enc.fit(train_df[ordinal_features])
text_enc.fit(train_df[[text_feature]])
onehot_enc.fit(train_df[onehot_features])
scaler.fit(train_df[numerical_features])
      
    

class FloodNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_emb = make_embedding(
            num_categories=len(text_enc.categories_[0]), dim=8
        )
        self.ord_embs = nn.ModuleList(
            [
                make_embedding(len(cat), dim=4)
                for cat in ord_enc.categories_
            ]
        )
        onehot_dim = train_set.onehots.shape[1]
        num_dim = len(numerical_features)
        concat_dim = 8 + 4 * len(ordinal_features) + onehot_dim + num_dim
        self.net = nn.Sequential(
            nn.Linear(concat_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, len(label_enc.classes_)),
        )

    def forward(self, text, ord, onehot, num):
        x = [self.text_emb(text)]
        for i, emb in enumerate(self.ord_embs):
            x.append(emb(ord[:, i]))
        x.append(onehot)
        x.append(num)
        x = torch.cat(x, dim=1)
        return self.net(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FloodNet().to(device)

# ---------------------------------------------------------------------
# 6. Helper functions for input validation ----------------------------
# ---------------------------------------------------------------------
def show_valid_options():
    print("\n📋 Valid options for each feature:")
    print("=" * 50)
    for feature, options in valid_categories.items():
        print(f"{feature}: {', '.join(options)}")
    print("=" * 50)

def validate_input(feature, value):
    """Check if input value is valid for the given feature"""
    if feature in valid_categories:
        return value in valid_categories[feature]
    return True

def get_closest_match(feature, value):
    """Find closest match for invalid input"""
    if feature not in valid_categories:
        return value
    valid_options = valid_categories[feature]
    value_lower = value.lower()
    for option in valid_options:
        if option.lower() == value_lower:
            return option
    for option in valid_options:
        if value_lower in option.lower() or option.lower() in value_lower:
            return option
    return None

# ---------------------------------------------------------------------
# 7. Getting user input with validation -------------------------------
# ---------------------------------------------------------------------
def get_user_input():
    print("\nEnter environmental details to predict flood risk:")
    print("(Type 'help' to see valid options for any field)")
    user_input_data = {}
    prompts = {
        'Vegetation': "Vegetation: ",
        'Urbanization': "Urbanization: ",
        'Drainage': "Drainage: ",
        'Slope': "Slope: ",
        'StormFrequency': "Storm Frequency: ",
        'Deforestation': "Deforestation: ",
        'Infrastructure': "Infrastructure: ",
        'Encroachment': "Encroachment: ",
        'Season': "Season: ",
        'Soil': "Soil: ",
        'Wetlands': "Wetlands: ",
        'ProximityToWaterBody': "Proximity to water body: ",
        'Rainfall': "Rainfall (numeric): ",
        'Elevation': "Elevation (numeric): "
    }
    for feature, prompt in prompts.items():
        while True:
            value = input(prompt).strip()
            if value.lower() == 'help':
                if feature in valid_categories:
                    print(f"Valid options for {feature}: {', '.join(valid_categories[feature])}")
                continue
            if feature in ['Rainfall', 'Elevation']:
                try:
                    user_input_data[feature] = [float(value)]
                    break
                except ValueError:
                    print("Please enter a valid number.")
                    continue
            else:
                value = value.title()
                if validate_input(feature, value):
                    user_input_data[feature] = [value]
                    break
                else:
                    closest = get_closest_match(feature, value)
                    if closest:
                        print(f"Did you mean '{closest}'? Using that instead.")
                        user_input_data[feature] = [closest]
                        break
                    else:
                        print(f"❌ Invalid input for {feature}.")
                        if feature in valid_categories:
                            print(f"Valid options: {', '.join(valid_categories[feature])}")
                        print("Please try again or type 'help' for options.")
    return pd.DataFrame(user_input_data)

# ---------------------------------------------------------------------
# 8. Preprocess user input --------------------------------------------
# ---------------------------------------------------------------------
def preprocess_user_input(user_df):
    try:
        for col in ordinal_features + onehot_features + [text_feature]:
            if col in user_df.columns:
                user_df[col] = user_df[col].fillna("Missing")
        ordinals = ord_enc.transform(user_df[ordinal_features]).astype(np.int64)
        onehots = onehot_enc.transform(user_df[onehot_features]).astype(np.float32)
        text_code = text_enc.transform(user_df[[text_feature]]).astype(np.int64).squeeze(1)
        numerical = user_df[numerical_features].copy()
        for col in ["Elevation", "Rainfall"]:
            q1, q3 = df[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            numerical[col] = numerical[col].clip(lower=max(q1 - 1.5 * iqr, 0), upper=q3 + 1.5 * iqr)
            mn = numerical[col].min()
            numerical[col] = np.log1p(numerical[col] + abs(mn) + 1e-6)
        numerical = scaler.transform(numerical).astype(np.float32)
        return (
            torch.tensor(text_code, dtype=torch.long).to(device),
            torch.tensor(ordinals, dtype=torch.long).to(device),
            torch.tensor(onehots, dtype=torch.float).to(device),
            torch.tensor(numerical, dtype=torch.float).to(device),
        )
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        raise

def predict_from_input():
    model.eval()
    user_df = get_user_input()
    text_tensor, ord_tensor, onehot_tensor, num_tensor = preprocess_user_input(user_df)
    with torch.no_grad():
        out = model(text_tensor, ord_tensor, onehot_tensor, num_tensor)
        probabilities = torch.softmax(out, dim=1)
        pred = out.argmax(1).item()
    risk_level = label_enc.inverse_transform([pred])[0]
    confidence = probabilities[0][pred].item()
    print(f"\n🌊 Predicted Flood Risk Level: {risk_level}")
    print(f"📊 Confidence: {confidence:.1%}")
    print("\n📈 Risk Breakdown:")
    for i, class_name in enumerate(label_enc.classes_):
        prob = probabilities[0][i].item()
        print(f"   {class_name}: {prob:.1%}")

# ---------------------------------------------------------------------
# Main execution block (training, testing, and interactive loop) -------
# ---------------------------------------------------------------------
if __name__== "__main__":
    # 4. TRAINING ----------------------------------------------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=2e-3, weight_decay=0)
    epochs = 40
    patience = 6
    best_f1, best_state, epochs_without_improve = 0.0, None, 0

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(
                batch["text"].to(device),
                batch["ord"].to(device),
                batch["onehot"].to(device),
                batch["num"].to(device),
            )
            loss = criterion(out, batch["label"].to(device))
            loss.backward()
            optimizer.step()
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                out = model(
                    batch["text"].to(device),
                    batch["ord"].to(device),
                    batch["onehot"].to(device),
                    batch["num"].to(device),
                )
                val_preds.extend(out.argmax(1).cpu().numpy())
                val_labels.extend(batch["label"].numpy())
        val_f1 = f1_score(val_labels, val_preds, average="weighted")
        print(f"Epoch {epoch:02d}  Val-F1 = {val_f1:.4f}")
        if val_f1 > best_f1 + 1e-3:
            best_f1, best_state = val_f1, model.state_dict()
            epochs_without_improve = 0
            torch.save(best_state, "best_floodnet_model.pth")
            print("  ⤷ new best model saved")
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= patience:
                print("  ⤷ early-stopping")
                break

    # 5. TEST --------------------------------------------------------------
    model.load_state_dict(best_state)
    model.eval()
    test_preds, test_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            out = model(
                batch["text"].to(device),
                batch["ord"].to(device),
                batch["onehot"].to(device),
                batch["num"].to(device),
            )
            test_preds.extend(out.argmax(1).cpu().numpy())
            test_labels.extend(batch["label"].numpy())
    print(
        f"Best Val-F1 = {best_f1:.4f} | "
        f"Test-F1 = {f1_score(test_labels, test_preds, average='weighted'):.4f}"
    )

    from sklearn.metrics import ConfusionMatrixDisplay
    import matplotlib.pyplot as plt
    ConfusionMatrixDisplay.from_predictions(
        test_labels,
        test_preds,
        display_labels=label_enc.classes_,
        cmap="Blues",
        xticks_rotation=45,
    )
    plt.tight_layout()
    plt.show()

    # Interactive prediction loop
    print("\n🌊 FLOOD RISK PREDICTION SYSTEM 🌊")
    show_valid_options()
    while True:
        try:
            predict_from_input()
            questionnaire = input("\nWould you like to predict again? (Yes/No): ").strip().lower()
            if questionnaire not in ["yes", "y"]:
                print("Goodbye! 👋")
                break
        except KeyboardInterrupt:
            print("\n\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please try again with valid inputs.")