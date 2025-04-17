# GridRL

This project is an interactive reinforcement learning environment where an autonomous agent must learn to navigate an 8x8 grid. Its goal is to move from a randomly placed starting position to a randomly placed goal, both located on the edges of the grid. Along the way, the agent must avoid obstacles that penalize its performance and take advantage of rewards that reduce its total travel time.

Despite its visual simplicity, this environment provides an excellent foundation for studying RL techniques applicable to real-world contexts such as logistics, route planning, or decision-making in partially observable environments. The environment's architecture allows real-time observation of the agent's behavior during training and the analysis of its learning progress through visual plots.

---

## Features

The environment generates an 8x8 board with random positions for the start and goal. Through a Streamlit-based interface, users can manually edit the environment or populate it randomly with obstacles and rewards.

- **Obstacles** represent time penalties and simulate unfavorable movement conditions such as walls, lava, or sand.
- **Rewards** provide performance bonuses, simulating elements such as coins, treasures, or potions.

The agent is trained using the **Deep Q-Learning (DQN)** algorithm, implemented in **PyTorch**, and learns to maximize its total reward during each episode by deciding which actions to take based on the environment's state.

---

## Potential Applications

Although designed as an educational and visually accessible environment, this simulator can serve as a foundation for:

- Autonomous navigation systems  
- Route planning and optimization algorithms  
- Logistics problem simulations  

The environment is **extensible** and can be easily adapted to different configurations or more complex dynamics.

---

## Installation and Execution

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/GridRL.git
   cd GridRL
   ```
2. **Create the virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. **Install dependencies:**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Run the application**:
   ```bash
   streamlit run main.py
   ```
Once the application is launched, a web interface will open where you can configure the environment, start the agent's training, and observe the results of the learning process.

---

## Project Structure

The project is organized into several modules that separate the environment logic, agent implementation, and graphical interface:

- `main.py`: Entry point of the application  
- `app.py`: Interaction and visualization logic with Streamlit  
- `environment.py`: Definition of the Grid World environment  
- `agent.py`: Implementation of the Deep Q-Learning-based agent  
- `requirements.txt`: List of dependencies needed to run the project

   
