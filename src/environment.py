from gymnasium import Env, spaces
import numpy as np
import random
import math
import scipy.special as sp
from scipy.special import erf


def calculate_psi(rssi, eta_fi, sigma):
    # Calcular Nik (ruido)
    #N = np.random.normal(0, sigma)  # Valor aleatorio de ruido con media 0 y desviación estándar sigma
    N = 0

    # Calcular zik
    z = rssi + N

    # Calcular psi
    arg = (z - eta_fi) / (np.sqrt(2) * sigma)
    psi = 0.5 * (1 + erf(arg))

    return psi


def normalize_value(value, arr):
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    normalized_value = (value - arr_min) / (arr_max - arr_min)
    return normalized_value


class LoRaEnv(Env):
    ALL_ACTIONS = {
        "no_tx": {'DR': None, 'CR': None, 'SF': None, 'SNR': None, 'BW': None},
        "a1": {'DR': 0, 'CR': 4 / 5, 'SF': 12, 'SE': -137, 'BW': 125},
        "a2": {'DR': 1, 'CR': 4 / 5, 'SF': 11, 'SE': -134.5, 'BW': 125},
        "a3": {'DR': 2, 'CR': 4 / 5, 'SF': 10, 'SE': -132, 'BW': 125},
        "a4": {'DR': 3, 'CR': 4 / 5, 'SF': 9, 'SE': -129, 'BW': 125},
        "a5": {'DR': 4, 'CR': 4 / 5, 'SF': 8, 'SE': -126, 'BW': 125},
        "a6": {'DR': 5, 'CR': 4 / 5, 'SF': 7, 'SE': -123, 'BW': 125}
    }

    def __init__(self):
        """Initialize environment parameters and state."""
        self.rssi = -115  # Initial Signal-to-Noise Ratio (SNR) in dB
        self.prob_th = 0.6
        self.tdc = 0  # Initial Time Duty Cycle (TDC) in seconds
        self.w1 = 2
        self.w2 = 1
        self.w3 = 1
        self.max_energy = 20.52
        self.PACKET_SIZE_BYTES = 26  # Size of the packet in bytes
        self.MAX_BATTERY_LEVEL = 21312  # Maximum battery level in Joules
        self.PTX = 14  # Transmission power in dBm
        self.LAMBDA_VAL = 0.1  # To control the exponential decay rate in the reward function
        self.PL_d0 = 70  # Pérdida de trayectoria a una distancia de referencia en dB
        self.d0 = 1  # Distancia de referencia en metros
        self.gamma = 2.0  # Exponente de la pérdida de trayectoria
        self.sigma = 2  # Desviación estándar del ruido en dB
        self.toa = 0  # Time on air for transmitted packet (calculated during step)
        self.energy = 0  # Energy consumed for transmitted packet (calculated during step)
        self.ber = 0  # Bit Error Rate (BER) for transmitted packet (calculated during step)
        self.sensitivities = {
            7: -123,
            8: -126,
            9: -129,
            10: -132,
            11: -134.5,
            12: -137
        }  # Sensibilidades de la puerta de enlace para diferentes SF
        self.rssi_values = [-140, -90]
        self.thresholds = [0.6, 0.7, 0.8, 0.9]
        self.tdc_values = [0, 36]

        self.action_space = spaces.Discrete(len(self.ALL_ACTIONS))
        self.observation_space = spaces.Box(low=0, high=1, shape=(3,), dtype=np.float64)

        self.state = [normalize_value(self.rssi, self.rssi_values),
                      normalize_value(self.prob_th, self.thresholds),
                      normalize_value(self.tdc, self.tdc_values)]

    def step(self, action):
        """Perform a step in the environment based on the given action."""
        done = False

        if action == 0:
            remaining_tdc = self.tdc
            reward = 0

        else:
            selected_action = list(self.ALL_ACTIONS.values())[action]
            sf = selected_action['SF']
            cr = selected_action['CR']
            sensitivity = selected_action['SE']
            bw = selected_action['BW']

            h, de = self._calculate_h_de(sf)
            self.toa, self.energy = self._model_energy(h, de, sf, cr, bw)

            #self.ber = self._calculate_ber(sf)

            remaining_tdc = self.tdc - self.toa

            if remaining_tdc >= 0:
                g_q = 1
                psi = calculate_psi(self.rssi, sensitivity, self.sigma)
                if psi >= self.prob_th:
                    prob_difference = abs(psi - self.prob_th)
                    max_difference = 1
                    f_q = (1.0 - (prob_difference / max_difference))
                else:
                    f_q = -1.2
            else:
                f_q = -1
                g_q = -1

            normalized_energy = self.energy / self.max_energy
            reward = self.w1 * f_q + self.w2 * g_q - self.w3 * normalized_energy

        # Update distance
        self.rssi = random.uniform(np.min(self.rssi_values), np.max(self.rssi_values))
        # Update threshold
        self.prob_th = random.choice(self.thresholds)
        # Update TDC
        self.tdc = random.uniform(np.min(self.tdc_values), np.max(self.tdc_values))

        self.state = [normalize_value(self.rssi, self.rssi_values),
                      normalize_value(self.prob_th, self.thresholds),
                      normalize_value(self.tdc, self.tdc_values)]

        observation = np.array(self.state)
        return observation, reward, done, False, {}

    def reset(self, *, seed=None, options=None):
        """Reset the environment to a random state."""
        self.rssi = random.uniform(np.min(self.rssi_values), np.max(self.rssi_values))
        self.prob_th = random.choice(self.thresholds)
        self.tdc = random.uniform(np.min(self.tdc_values), np.max(self.tdc_values))

        self.state = [normalize_value(self.rssi, self.rssi_values),
                      normalize_value(self.prob_th, self.thresholds),
                      normalize_value(self.tdc, self.tdc_values)]

        observation = np.array(self.state)
        return observation, {}

    def render(self, mode='human'):
        # Implement the render function to visualize the environment
        pass

    def set_state(self, rssi, threshold, tdc):
        """Set the environment state with given SNR and TDC values."""
        self.rssi = rssi
        self.prob_th = threshold
        self.tdc = tdc

        self.state = [normalize_value(self.rssi, self.rssi_values),
                      normalize_value(self.prob_th, self.thresholds),
                      normalize_value(self.tdc, self.tdc_values)]

        observation = np.array(self.state)
        return observation

    def get_statistics(self):
        """Return statistics related to the last transmitted packet."""
        return [self.toa, self.energy, self.ber]

    @staticmethod
    def _calculate_h_de(sf):
        """Calculate header (h) and coding rate (de) based on Spreading Factor (SF)."""
        if sf in [11, 12]:
            de = 1
        else:
            de = 0

        if sf == 6:
            h = 1
        else:
            h = 0
        return h, de

    def _model_energy(self, h, de, sf, cr, bw):
        """Model energy consumption and Time on Air (ToA) for transmitted packet."""
        n_p = 8
        t_pr = (4.25 + n_p) * pow(2, sf) / (bw*1000)  # Preamble time
        p_sy = 8 + max(((8 * self.PACKET_SIZE_BYTES - 4 * sf + 44 - 20 * h) / (4 * (sf - 2 * de))) * (cr + 4), 0)
        t_pd = p_sy * pow(2, sf) / (bw*1000)  # Payload time
        t = t_pr + t_pd  # Total Time on Air (ToA) in seconds
        e_pkt = self.PTX * t  # Energy consumed for packet transmission
        return t, e_pkt

    def _calculate_ber(self, sf):
        """Calculate Bit Error Rate (BER) for transmitted packet based on SNR and SF."""
        snr_linear = pow(10, self.snr / 10)
        ber = 0.5 * sp.erfc(math.sqrt(2 * (snr_linear / (pow(2, sf) - 1))))
        return ber


# Example usage of the LoRaEnv class
if __name__ == "__main__":
    env = LoRaEnv()

    action = 6
    obs, rw, dn, truncated, info = env.step(action)
    action = 1
    obs, rw, dn, truncated, info = env.step(action)
    action = 3
    obs, rw, dn, truncated, info = env.step(action)
    # Reset the environment
    obs = env.reset()

