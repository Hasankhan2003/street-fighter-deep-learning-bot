import socket
import json
import csv
import sys
from game_state import GameState
from bot import Bot

def connect(port):
    # For making a connection with the game
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(("127.0.0.1", port))
    server_socket.listen(5)
    (client_socket, _) = server_socket.accept()
    print("Connected to game!")
    return client_socket

def send(client_socket, command):
    # Send your updated command to Bizhawk so that game reacts accordingly
    command_dict = command.object_to_dict()
    pay_load = json.dumps(command_dict).encode()
    client_socket.sendall(pay_load)

def receive(client_socket):
    # Receive the game state and return it
    pay_load = client_socket.recv(4096)
    input_dict = json.loads(pay_load.decode())
    game_state = GameState(input_dict)
    return game_state
import os

def save_data(game_state, buttons, player_id):
    file_exists = os.path.isfile("gameplay_data.csv")
    with open("gameplay_data.csv", "a", newline="") as file:
        writer = csv.writer(file)
        
        if not file_exists or os.stat("gameplay_data.csv").st_size == 0:
            # Write header once
            writer.writerow([
                "player_id",
                "p1_x", "p1_y", "p1_health", "p1_jumping", "p1_crouching", "p1_in_move", "p1_move_id",
                "p2_x", "p2_y", "p2_health", "p2_jumping", "p2_crouching", "p2_in_move", "p2_move_id",
                "timer",
                "up", "down", "left", "right",
                "Y", "B", "X", "A",
                "L", "R"
            ])

        # Now write actual data row
        p1 = game_state.player1
        p2 = game_state.player2
        row = [
            player_id,
            p1.x_coord, p1.y_coord, p1.health, int(p1.is_jumping), int(p1.is_crouching), int(p1.is_player_in_move), p1.move_id,
            p2.x_coord, p2.y_coord, p2.health, int(p2.is_jumping), int(p2.is_crouching), int(p2.is_player_in_move), p2.move_id,
            game_state.timer,
            int(buttons.up), int(buttons.down), int(buttons.left), int(buttons.right),
            int(buttons.Y), int(buttons.B), int(buttons.X), int(buttons.A),
            int(buttons.L), int(buttons.R)
        ]
        writer.writerow(row)

def main():
    if sys.argv[1] == '1':
        client_socket = connect(9999)
    elif sys.argv[1] == '2':
        client_socket = connect(10000)

    current_game_state = None
    bot = Bot()

    while (current_game_state is None) or (not current_game_state.is_round_over):
        current_game_state = receive(client_socket)

        # Extract human player button data for training
        if sys.argv[1] == '1':
            buttons = current_game_state.player1.player_buttons
        else:
            buttons = current_game_state.player2.player_buttons

        # Save each frame of gameplay
        #save_data(current_game_state, buttons, sys.argv[1])

        # Call bot (still works if you later test bot vs CPU)
        bot_command = bot.fight(current_game_state, sys.argv[1])
        send(client_socket, bot_command)

if __name__ == '__main__':
    main()
