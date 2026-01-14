import torch
import numpy as np
from command import Command
from buttons import Buttons
from torch import nn
import joblib  # for loading StandardScaler if saved

# Neural network architecture (same as in training)
class BotModel(nn.Module):
    def __init__(self):
        super(BotModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(15, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 10),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

class Bot:
    def __init__(self):
        # Load trained model
        self.model = BotModel()
        self.model.load_state_dict(torch.load("improved_bot_model.pt"))
        self.model.eval()

        # If using standardization:
        # self.scaler = joblib.load("scaler.pkl")

    def fight(self, current_game_state, player):
        # Select bot and opponent player
        if player == "1":
            self_player = current_game_state.player1
            opp_player = current_game_state.player2
        else:
            self_player = current_game_state.player2
            opp_player = current_game_state.player1

        # Input features: 15 dimensions
        features = np.array([[
            self_player.x_coord,
            self_player.y_coord,
            self_player.health,
            int(self_player.is_jumping),
            int(self_player.is_crouching),
            int(self_player.is_player_in_move),
            self_player.move_id,
            opp_player.x_coord,
            opp_player.y_coord,
            opp_player.health,
            int(opp_player.is_jumping),
            int(opp_player.is_crouching),
            int(opp_player.is_player_in_move),
            opp_player.move_id,
            current_game_state.timer
        ]], dtype=np.float32)

        # Optional: normalize if you saved a scaler
        # features = self.scaler.transform(features)

        input_tensor = torch.tensor(features)

        # Inference
        with torch.no_grad():
            prediction = self.model(input_tensor)[0].numpy()
            print("🎯 Model predictions:", prediction)

        # Threshold for multi-label activation
        threshold = 0.5
        btn = Buttons()
        btn.up    = bool(prediction[0] > threshold)
        btn.down  = bool(prediction[1] > threshold)
        btn.left  = bool(prediction[2] > threshold)
        btn.right = bool(prediction[3] > threshold)
        btn.Y     = bool(prediction[4] > threshold)
        btn.B     = bool(prediction[5] > threshold)
        btn.X     = bool(prediction[6] > threshold)
        btn.A     = bool(prediction[7] > threshold)
        btn.L     = bool(prediction[8] > threshold)
        btn.R     = bool(prediction[9] > threshold)

        # Build command
        command = Command()
        if player == "1":
            command.player_buttons = btn
        else:
            command.player2_buttons = btn

        return command