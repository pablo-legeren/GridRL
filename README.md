# GridRL

Este proyecto consiste en un entorno interactivo de aprendizaje por refuerzo (Reinforcement Learning) donde un agente autónomo debe aprender a navegar en una cuadrícula de 8x8. Su objetivo es desplazarse desde una posición inicial hasta una posición final, ambas ubicadas aleatoriamente en los bordes del grid. En su trayecto, el agente debe evitar obstáculos que penalizan su rendimiento y, al mismo tiempo, aprovechar recompensas que reducen su tiempo total.

A pesar de su simplicidad visual, este entorno representa una excelente base para el estudio de técnicas de RL aplicadas a contextos reales como la logística, la planificación de rutas o la toma de decisiones en entornos parcialmente observables. La arquitectura del entorno permite observar, en tiempo real, el comportamiento del agente durante su entrenamiento, y analizar la evolución de su aprendizaje mediante gráficas.

## Características

El entorno genera un tablero de 8x8 con ubicaciones aleatorias para el inicio y el objetivo. A través de una interfaz desarrollada con Streamlit, es posible editar manualmente el entorno o poblarlo aleatoriamente con obstáculos y recompensas.

Los obstáculos representan penalizaciones de tiempo y simulan condiciones desfavorables para el movimiento del agente, como muros, lava o arena. Por otro lado, las recompensas ofrecen bonificaciones al rendimiento, simulando elementos como monedas, tesoros o pociones.

El agente se entrena utilizando el algoritmo Deep Q-Learning (DQN), implementado en PyTorch, y aprende a maximizar su recompensa total durante cada episodio, decidiendo qué acciones tomar en función del estado del entorno.

## Aplicaciones potenciales

Aunque diseñado como un entorno educativo y visualmente accesible, este simulador puede servir como base para:

- Sistemas de navegación autónoma.
- Algoritmos de planificación y optimización de rutas.
- Simulación de problemas logísticos.

El entorno es extensible y puede adaptarse fácilmente a diferentes configuraciones o dinámicas más complejas.

## Instalación y ejecución

1. Clonar el repositorio:
```bash
git clone https://github.com/tu-usuario/GridRL.git
cd GridRL
```

2. Crear un entorno virtual (opcional):
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

3. Instalar las dependencias:
```bash
pip install -r requirements.txt
```

4. Ejecutar la aplicación:
```bash
streamlit run main.py
```

Una vez lanzada la aplicación, se abrirá una interfaz web donde se puede configurar el entorno, iniciar el entrenamiento del agente y observar los resultados del proceso de aprendizaje.

## Estructura del proyecto

El proyecto está organizado en varios módulos que separan la lógica del entorno, la implementación del agente y la interfaz gráfica:

- `main.py`: Punto de entrada de la aplicación.
- `app.py`: Lógica de interacción y visualización con Streamlit.
- `environment.py`: Definición del entorno Grid World.
- `agent.py`: Implementación del agente basado en Deep Q-Learning.
- `requirements.txt`: Lista de dependencias necesarias para ejecutar el proyecto.
