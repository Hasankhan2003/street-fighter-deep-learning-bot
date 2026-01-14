# Street Fighter II ML Bot (Python API)

This repository contains only the **Python API** portion of the AI Project for training and running a machine learning–based bot for Street Fighter II Turbo using the BizHawk emulator. The original project zip (provided by the course) contains the emulator, ROM, and surrounding folders; this repo focuses on the Python side (bot logic, data collection, and model training).      

***

## Original project structure

In the zip, the relevant folder is:

```text
AI Project/
  gamebot-competition-master/
    Python API/
      # <-- All the Python files go here
```

This repository is essentially that `Python API` folder extracted and version-controlled. To run the project end‑to‑end, you still need the original zip from the course (BizHawk, ROM, etc.).  

**Download the full project zip (emulator + assets)**  
> Link: `https://drive.google.com/file/d/18SN8e_XqJFEPZ0wcWXQ8GnzuZk58cn-2/view?usp=sharing`

After downloading and extracting the zip, copy the contents of this repo into the `Python API` folder inside the original structure, replacing or adding files as needed.  

***

## Contents of this repository

```text
python-api/
├─ bot.py
├─ buttons.py
├─ player.py
├─ game_state.py
├─ command.py
├─ controller.py
├─ train_model_pytorch.py
├─ gameplay_data.csv              
├─ gameplay_data_cleaned.csv      
├─ game_log.csv                   
└─ README.md                   
```

- `bot.py`: ML-based bot, loads a trained PyTorch model and decides which buttons to press each frame using the current `GameState`.  
- `controller.py`: Handles socket connection with BizHawk, main game loop, and logging gameplay data to CSV for training. 
- `buttons.py`, `player.py`, `game_state.py`, `command.py`: Core API classes defining controller buttons, player state, game state, and the command sent back to the emulator.[4][6][7][11]  
- `train_model_pytorch.py`: Training script that reads `gameplay_data_cleaned.csv`, trains a neural network (`ImprovedBotModel`), and saves `improved_bot_model.pt`.  
- `gameplay_data.csv` / `gameplay_data_cleaned.csv`: Recorded gameplay data (per frame state + button presses) and a cleaned version without “no-action” frames.[10]  

***

## Setup

1. **Get the full project zip**

   - Download and extract the original course zip from:  
     `https://drive.google.com/file/d/18SN8e_XqJFEPZ0wcWXQ8GnzuZk58cn-2/view?usp=sharing`  
   - Inside it, locate: `AI Project/Application/Python API/`.  

2. **Copy this Python API folder**

   - Clone this repository, then copy its contents into the `Python API` folder from the zip (or replace the existing Python files there).    

3. **Install Python and dependencies**

   - Use Python 3.8+ (original API was tested on 3.6.3 but works on newer versions).  
   - From inside the `Python API` folder:

   ```bash
   pip install pandas torch scikit-learn
   ```

   - These are used for data loading, preprocessing, and model training/inference. 

***

## How the ML bot works (brief)

1. **Data collection**

   - When you play manually using `controller.py`, each frame’s game state and your button presses are logged into `gameplay_data.csv`.  
   - A cleaning step (example in the notebook that produced `gameplay_data_cleaned.csv`) removes frames where no buttons are pressed.[10] 

2. **Model training**

   - `train_model_pytorch.py`:
     - Loads `gameplay_data_cleaned.csv`.  
     - Splits into features (game state) and labels (buttons). 
     - Normalizes features with `StandardScaler`. 
     - Trains a neural network (`ImprovedBotModel`) with multi-label outputs for each button and saves `improved_bot_model.pt`. 

3. **Inference during gameplay**

   - `bot.py`:
     - Loads `improved_bot_model.pt` and the saved scaler.  
     - Converts the current `GameState` into a 15‑dimensional feature vector (positions, health, jump/crouch/move flags, timer).[6] 
     - Runs the model, applies a threshold to each output, and sets the corresponding buttons in a `Buttons` object inside a `Command`.[7][4] 
   - `controller.py` sends this `Command` back to BizHawk each frame. 

***

## How to run matches with the bot

After placing this folder into the original zip’s `Python API` location and installing dependencies:

1. **Launch BizHawk and load the ROM**

   - Run `EmuHawk.exe` from the course zip.  
   - `File` → `Open ROM` → select `Street Fighter II Turbo (U).smc`.  

2. **Open the Tool Box**

   - `Tools` → `Tool Box` (or `Shift+T`).  

3. **Start the Python controller**

   In a terminal opened inside the `Python API` folder:

   ```bash
   # Bot controls Player 1 (left side)
   python controller.py 1

   # Bot controls Player 2 (right side)
   python controller.py 2
   ```

   The argument `1` or `2` selects which in‑game player is controlled by the bot.   

4. **Connect from BizHawk**

   - Once the terminal shows `Connected to game!`, click the **Gyroscope Bot** icon in BizHawk to start the connection.   

5. **Play or watch**

   - Choose a mode (e.g., Normal or VS Battle), select characters, and start the round; the bot will now play automatically based on the model’s decisions.    

***

## Reproducing training (optional)

If you want to retrain the model or adapt it:

1. Generate or reuse `gameplay_data.csv` by playing matches with `controller.py`. 
2. Clean it to remove frames with all-zero button labels, producing `gameplay_data_cleaned.csv` (example logic is in the original notebook used for this repo).[10] 
3. Run:

```bash
python train_model_pytorch.py
```

This will train `ImprovedBotModel` and save `improved_bot_model.pt` in the same folder. 

