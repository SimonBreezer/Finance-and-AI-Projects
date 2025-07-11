from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd

# Load model and encoders
model = joblib.load("qa_model.pkl")
encoders = joblib.load("qa_encoder.pkl")

# Must match training script
categorical_cols = [
    'Classification', 'Primary Platform', 'Game Engine', 'Genre',
    'Game World Scale', 'Gameplay Complexity', 'Multiplayer', 'Game Modes'
]

# Define input schema
class GameAttributes(BaseModel):
    Classification: str
    Primary_Platform: str
    Game_Engine: str
    Genre: str
    Game_World_Scale: str
    Gameplay_Complexity: str
    Multiplayer: str
    Game_Modes: str

app = FastAPI()

# Enable CORS (so your frontend can call this from localhost:5173 or similar)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or use ["http://localhost:5173"] for stricter config
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
def predict(attrs: GameAttributes):
    input_dict = {
        'Classification': attrs.Classification,
        'Primary Platform': attrs.Primary_Platform,
        'Game Engine': attrs.Game_Engine,
        'Genre': attrs.Genre,
        'Game World Scale': attrs.Game_World_Scale,
        'Gameplay Complexity': attrs.Gameplay_Complexity,
        'Multiplayer': attrs.Multiplayer,
        'Game Modes': attrs.Game_Modes
    }

    encoded_input = []
    for col in categorical_cols:
        val = input_dict[col]
        val_encoded = encoders[col].transform([val])[0]
        encoded_input.append(val_encoded)

    input_df = pd.DataFrame([encoded_input], columns=categorical_cols)
    prediction = model.predict(input_df)[0]

    # Calculate person-months
    person_months = [round(h / 150, 0) for h in prediction]

    return {
        "QA_Curve": [
            {
                "Months_to_release": i,
                "Predicted_QA_Hours": round(h, 0),
                "Person_Month": person_months[i]
            } for i, h in enumerate(prediction)
        ]
    }
