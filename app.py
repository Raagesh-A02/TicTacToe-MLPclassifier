import streamlit as st
import numpy as np
import pickle

# Load ML model
with open("tictactoe_model.pkl", "rb") as f:
    model = pickle.load(f)

# Encode board for ML
def encode_board(board):
    mapping = {'x': 0, 'o': 1, 'b': 2}
    return [mapping[c] for c in board]

# Check winner
def check_winner(board, player):
    combos = [(0,1,2), (3,4,5), (6,7,8),
              (0,3,6), (1,4,7), (2,5,8),
              (0,4,8), (2,4,6)]
    return any(all(board[i] == player for i in combo) for combo in combos)

# Reset game
def reset_game():
    st.session_state.board = ['b'] * 9
    st.session_state.game_over = False
    st.session_state.message = ""

# Initialize state
if 'board' not in st.session_state:
    reset_game()

# Symbol mapping
def symbol(cell):
    return {"x": "❌", "o": "⭕", "b": "⬜"}[cell]

st.title("TicTacToe - ML Powered AI")
st.write("You play as ❌. AI plays as ⭕.")

# Track clicked position
clicked = -1
for row in range(3):
    cols = st.columns(3)
    for col in range(3):
        idx = row * 3 + col
        cell_symbol = symbol(st.session_state.board[idx])
        key = f"{idx}_{st.session_state.board[idx]}"

        if st.session_state.board[idx] == 'b' and not st.session_state.game_over:
            if cols[col].button("⬜", key=key):
                clicked = idx
        else:
            cols[col].button(cell_symbol, key=key, disabled=True)

# Player move
if clicked != -1 and not st.session_state.game_over:
    st.session_state.board[clicked] = 'x'

    if check_winner(st.session_state.board, 'x'):
        st.session_state.message = "You Win!"
        st.session_state.game_over = True
    elif 'b' not in st.session_state.board:
        st.session_state.message = "It's a Draw!"
        st.session_state.game_over = True
    else:
        # AI move
        encoded = np.array([encode_board(st.session_state.board)])
        ai_move = model.predict(encoded)[0]
        while st.session_state.board[ai_move] != 'b':
            ai_move = (ai_move + 1) % 9
        st.session_state.board[ai_move] = 'o'

        if check_winner(st.session_state.board, 'o'):
            st.session_state.message = "AI Wins!"
            st.session_state.game_over = True
        elif 'b' not in st.session_state.board:
            st.session_state.message = "It's a Draw!"
            st.session_state.game_over = True

# Show message
st.markdown(f"### {st.session_state.message}")

# Restart
if st.button("🔁 Restart Game"):
    reset_game()
