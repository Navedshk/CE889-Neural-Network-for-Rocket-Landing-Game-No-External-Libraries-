
# Neural Network for Rocket Landing Game (No External Libraries) - CE889

This project features a feed-forward neural network built from scratch with backpropagation, implemented without any external libraries. The goal is to train the neural network to control the landing of a rocket in a simulated game environment.

## Project Overview

The neural network is trained using data collected from manual gameplay to learn how to safely land a rocket. After training, the network is integrated into the game for real-time rocket control.

### Key Components

- **Data Collection**: Data collected from manual gameplay to train the network.
- **Neural Network Architecture**: Implements a basic feed-forward network with backpropagation.
- **Training**: Network learns to predict optimal actions for landing the rocket.
- **Integration**: The trained network is used in the game to control the rocket's landing.

### Files

- **`main.py`**: Contains the neural network implementation and training process.
- **`neuralnetholder.py`**: Integrates the trained neural network with the game, enabling real-time control of the rocket.

### Getting Started

1. **Install Dependencies**: No external libraries are required beyond Python's standard libraries.

2. **Run the Code**: 
   - First, execute the training script:
     ```bash
     python main.py
     ```
   - Then, run the integration script to see the trained network in action:
     ```bash
     python neuralnetholder.py
     ```

### Contact

For any questions or feedback, please reach out to:

- **Name**: Naved Shaikh
- **Email**: navedshaikh77920@gmail.com

### License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

Thank you for exploring this project!
```

This updated `README.md` now includes instructions for both training the model and integrating it with the game. Adjust any details as needed.
