# 🔐 URL Classification using Federated Learning

This project demonstrates the implementation of a simple feed-forward neural network to classify URLs as **benign** or **unsafe**. It serves as the foundation for simulating a **federated learning** environment, allowing decentralized model training without centralized data collection.

---

## 🚀 Project Overview

🔸 **Goal**: Build a neural network for URL classification, then extend it to a federated learning setup to preserve data privacy.

🔸 **Approach**:
- Implemented a basic neural network for binary classification.
- Simulated federated learning by training the model on isolated data partitions across multiple clients.

---

## 🧠 Model Architecture

- **Input**: 72 numerical features extracted from each URL.
- **Network**:
  - Dense (64 units) + ReLU
  - Dense (32 units) + ReLU
  - Output: Dense (1 unit) + Sigmoid (for binary classification)

---

## 🛠 Technologies Used

- Python
- PyTorch / TensorFlow (choose the one you used)
---

## 🔮 Future Improvements

- Expand the system to use real-world, device-distributed data in a true federated setting.
- Integrate secure aggregation and differential privacy techniques to further enhance data security.
- Experiment with more complex architectures such as Convolutional Neural Networks (CNNs), Recurrent Neural Networks (RNNs), or Transformers for improved performance.
- Benchmark federated learning performance against traditional centralized training methods.
- Extend the model to handle multiclass classification (e.g., phishing, malware, spam).
- Add automated logging, evaluation metrics, and model versioning.

