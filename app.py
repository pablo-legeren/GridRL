import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import deque
import random
import time
from environment import GridEnvironment
from agent import DQNAgent


def get_cell_icon(pos, env, obstacle_types, reward_types):
    if pos == env.start_pos:
        return "🚗"  # Car at start
    elif pos == env.end_pos:
        return "🏁"  # Finish flag
    elif pos == env.current_pos:
        return "🚗"  # Moving car
    elif pos in env.obstacles:
        obs_type = env.obstacles[pos]
        if obs_type == 1:
            return "🧱"  # Wall
        elif obs_type == 2:
            return "🌋"  # Lava
        else:
            return "🏖️"  # Sand
    elif pos in env.rewards:
        rew_type = env.rewards[pos]
        if abs(rew_type) == 0.5:
            return "💰"  # Coin
        elif abs(rew_type) == 1:
            return "💎"  # Treasure
        else:
            return "🧪"  # Potion
    return "⬜"  # Empty cell

def show_intro():
    st.title("Bienvenido al Grid World RL")
    st.write("""
    ## Descripción del Juego
    
    Este es un entorno de Reinforcement Learning donde un agente debe aprender a navegar desde un punto inicial hasta un punto final en un grid de 8x8.
    
    ### Características:
    - El punto inicial y final se generan aleatoriamente en los bordes del grid
    - Puedes colocar diferentes tipos de obstáculos y recompensas haciendo clic en el grid
    - Los obstáculos penalizan al agente (suman tiempo)
    - Las recompensas benefician al agente (restan tiempo)
    - El agente aprende a través de Deep Q-Learning
    
    ### Controles:
    - Usa los controles laterales para configurar el entrenamiento
    - Haz clic en una celda del grid para colocar obstáculos y recompensas
    - El botón "Reiniciar" genera nuevas posiciones inicial y final
    """)
    
    if st.button("Comenzar"):
        st.session_state.page = "main"
        st.rerun()

def render_grid(env, obstacle_types, reward_types, timestamp=None):
    grid_placeholder = st.empty()
    with grid_placeholder.container():
        for i in range(env.size):
            cols = st.columns(env.size)
            for j, col in enumerate(cols):
                pos = (i, j)
                icon = get_cell_icon(pos, env, obstacle_types, reward_types)
                key = f"train_cell_{i}_{j}_{timestamp}" if timestamp else f"cell_{i}_{j}"
                col.button(icon, key=key, use_container_width=True, disabled=timestamp is not None)

def handle_cell_click(pos, env, selected_type, obstacle_types, reward_types):
    if selected_type is None:
        return
    if pos != env.start_pos and pos != env.end_pos:
        if selected_type in obstacle_types:
            if pos in env.rewards:
                del env.rewards[pos]
            env.obstacles[pos] = obstacle_types[selected_type]
        else:
            if pos in env.obstacles:
                del env.obstacles[pos]
            env.rewards[pos] = reward_types[selected_type]
    st.rerun()

def update_charts(times, rewards_history):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Evolución de Pasos")
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        ax1.plot(times, label='Pasos por episodio')
        ax1.set_xlabel("Episodio")
        ax1.set_ylabel("Número de pasos")
        ax1.axhline(y=64, color='r', linestyle='--', label='Máximo')
        ax1.fill_between(range(len(times)), 64, color='red', alpha=0.1)
        ax1.legend()
        st.pyplot(fig1)
        plt.close(fig1)
    
    with col2:
        st.subheader("Evolución de Recompensas")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.plot(rewards_history, label='Recompensa por episodio', color='green')
        ax2.set_xlabel("Episodio")
        ax2.set_ylabel("Recompensa total")
        if len(rewards_history) > 1:
            z = np.polyfit(range(len(rewards_history)), rewards_history, 1)
            p = np.poly1d(z)
            ax2.plot(range(len(rewards_history)), p(range(len(rewards_history))), 
                    "r--", alpha=0.8, label='Tendencia')
        ax2.legend()
        st.pyplot(fig2)
        plt.close(fig2)

def train_agent(env, agent, episodes, batch_size, obstacle_types, reward_types):
    grid_container = st.empty()
    progress_bar = st.progress(0)
    status_text = st.empty()
    metrics_container = st.empty()
    charts_container = st.empty()

    for e in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            agent.replay(batch_size)

            timestamp = time.time()
            with grid_container:
                render_grid(env, obstacle_types, reward_types, timestamp)
            time.sleep(0.2)

        st.session_state.times.append(env.time_step)
        st.session_state.rewards_history.append(total_reward)
        progress_bar.progress((e + 1) / episodes)
        
        with metrics_container:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Pasos", env.time_step)
            with col2:
                st.metric("Recompensas", f"{env.total_rewards:.2f}")
            with col3:
                st.metric("Obstáculos", f"{env.total_obstacles:.2f}")
            with col4:
                score = 64 - env.time_step - env.total_rewards + env.total_obstacles
                st.metric("Score Final", f"{score:.2f}")
        
        status_text.text(f"Episodio: {e+1}/{episodes}, Score: {score:.2f}, Recompensa Total: {total_reward:.2f}")
        
        with charts_container:
            update_charts(st.session_state.times, st.session_state.rewards_history)

def main_page():
    st.title("Grid World RL Dashboard")
    
    # Add custom CSS
    st.markdown("""
        <style>
        /* Main grid buttons style */
        div[data-testid="column"] > div > div > div > div > div > button {
            width: 60px !important;
            height: 60px !important;
            padding: 0px !important;
            line-height: 60px !important;
            text-align: center !important;
            font-size: 24px !important;
            margin: 1px !important;
            border-radius: 4px !important;
            background-color: white !important;
            border: 1px solid #e0e0e0 !important;
        }
        
        div[data-testid="column"] > div > div > div > div > div > button:hover {
            background-color: #f0f0f0 !important;
            border-color: #d0d0d0 !important;
            transform: scale(1.05);
            transition: transform 0.2s;
        }
        
        /* Sidebar buttons style */
        .sidebar .stButton > button {
            width: 100% !important;
            height: auto !important;
            line-height: 1.6 !important;
            padding: 10px 15px !important;
            margin: 5px 0 !important;
            font-size: 16px !important;
            border-radius: 5px !important;
            background-color: #f0f2f6 !important;
            border: 1px solid #e0e0e0 !important;
        }
        
        .sidebar .stButton > button:hover {
            background-color: #e0e2e6 !important;
            border-color: #d0d0d0 !important;
        }
        
        /* Selected button style */
        .sidebar .stButton > button.selected {
            background-color: #00acee !important;
            color: white !important;
        }
        
        /* Adjust grid container spacing */
        div[data-testid="column"] {
            padding: 0px 1px !important;
        }
        
        /* Make grid buttons container full width */
        div[data-testid="stHorizontalBlock"] {
            gap: 0 !important;
            padding: 0 !important;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Initialize environment and agent
    if 'env' not in st.session_state:
        st.session_state.env = GridEnvironment()
    if 'agent' not in st.session_state:
        state_size = st.session_state.env.size * st.session_state.env.size
        action_size = 4
        st.session_state.agent = DQNAgent(state_size, action_size)
    if 'times' not in st.session_state:
        st.session_state.times = []
    if 'rewards_history' not in st.session_state:
        st.session_state.rewards_history = []
    if 'selected_type' not in st.session_state:
        st.session_state.selected_type = None
    
    # Sidebar controls
    st.sidebar.header("Controles")
    episodes = st.sidebar.slider("Número de episodios", 1, 1000, 100)
    batch_size = st.sidebar.slider("Tamaño del batch", 1, 64, 32)
    
    # Element types setup
    st.sidebar.markdown("---")
    st.sidebar.header("🎲 Elementos del Grid")

    element_types = {
        "🧱 Pared (+1)": 1,
        "🌋 Lava (+2)": 2,
        "🏖️ Arena (+0.5)": 0.5,
        "💰 Moneda (-0.5)": -0.5,
        "💎 Tesoro (-1)": -1,
        "🧪 Poción (-0.2)": -0.2
    }

    obstacle_types = {k: v for k, v in element_types.items() if v > 0}
    reward_types = {k: v for k, v in element_types.items() if v < 0}

    # Obstacles section
    st.sidebar.subheader("🚧 Obstáculos")
    for label, value in obstacle_types.items():
        is_selected = st.session_state.selected_type == label
        if st.sidebar.button(label, key=f"btn_{label}"):
            st.session_state.selected_type = label if not is_selected else None

    # Rewards section
    st.sidebar.subheader("🎁 Recompensas")
    for label, value in reward_types.items():
        is_selected = st.session_state.selected_type == label
        if st.sidebar.button(label, key=f"btn_{label}"):
            st.session_state.selected_type = label if not is_selected else None

    st.sidebar.markdown("---")
    
    # Control buttons
    st.sidebar.subheader("🎮 Acciones")
    element_count = st.sidebar.slider("N° elementos aleatorios", 5, 50, 15)
    
    if st.sidebar.button("🎲 Azar", help="Genera obstáculos y recompensas aleatoriamente"):
        st.session_state.env.add_random_elements(obstacle_types, reward_types, count=element_count)
        st.rerun()

    if st.sidebar.button("🔄 Reiniciar Grid", help="Limpia el grid y genera nuevas posiciones"):
        st.session_state.env.reset_positions()
        st.session_state.env.obstacles.clear()
        st.session_state.env.rewards.clear()
        st.session_state.times = []
        st.session_state.rewards_history = []
        st.session_state.selected_type = None
        st.rerun()
    
    if st.sidebar.button("▶️ Iniciar Entrenamiento", help="Comienza el proceso de aprendizaje"):
        train_agent(st.session_state.env, st.session_state.agent, episodes, batch_size, obstacle_types, reward_types)

    # Grid visualization
    st.header("Grid World")
    col1, col2 = st.columns([3, 1])

    with col1:
        for i in range(st.session_state.env.size):
            cols = st.columns(st.session_state.env.size)
            for j, col in enumerate(cols):
                pos = (i, j)
                icon = get_cell_icon(pos, st.session_state.env, obstacle_types, reward_types)
                if col.button(icon, key=f"cell_{i}_{j}", 
                            use_container_width=True,
                            help=f"Posición ({i},{j})"):
                    handle_cell_click(pos, st.session_state.env, 
                                    st.session_state.selected_type, 
                                    obstacle_types, reward_types)

def main():
    if 'page' not in st.session_state:
        st.session_state.page = "intro"
    
    if st.session_state.page == "intro":
        show_intro()
    else:
        main_page()

if __name__ == "__main__":
    main() 