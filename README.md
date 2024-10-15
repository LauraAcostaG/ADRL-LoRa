# ADRL: A Reconfigurable Energy-Efficient Transmission Policy for Mobile LoRa Devices based on Reinforcement Learning

## Description

**ADRL** is a novel transmission mechanism designed for mobile LoRa devices in dynamic and variable environments. This method enhances the adaptability of LoRa devices by using a **k-nearest neighbors (KNN)**-based approximation to predict the received signal strength and a **deep reinforcement learning (DRL)** model to select transmission parameters that minimize energy consumption while ensuring a high probability of successful packet decoding. The system also leverages duty cycle information to maximize the packet delivery ratio and comply with regulatory requirements in LoRaWAN networks. This approach has proven effective in improving both energy efficiency and performance, especially in urban environments with rapidly changing channel conditions.

## Key Features

- Proactive ADR mechanism for LoRaWAN selecting transmission parameters for mobile nodes.
- Channel conditions predicted using k-nearest neighbors on buffered RSSI samples.
- Reconfigurable deep reinforcement learning controller minimizes energy consumption and ensures packet decoding probability.
- Duty cycle information used to reduce energy consumption and decide when to transmit.
- Evaluated in a realistic simulator, achieving up to 57% increase in packet delivery ratio and 15% reduction in energy consumption compared to related works.

## System Requirements

To run this project, you'll need the following:

- **Operating System**: Windows 11
- **Programming Language**: Python 3.10.4
- **Required Libraries**:
  - `numpy`
  - `matplotlib`
  - `scikit-learn`
  - `gymnasium`
  - `sympy`
  - `pandas`
  - `scipy`
  - `stable-baselines3`
 
## Installation

1. Clone the repository:

```bash
git clone https://github.com/your_username/ADRL_LoRa_project.git
cd ADRL_LoRa_project

2. Install the necessary dependencies:

pip install -r requirements.txt
