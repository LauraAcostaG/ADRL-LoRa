import numpy as np
from itertools import cycle


class AdaptiveDataRate:
    def __init__(self):
        # Initialize the parameters
        self.adr_snr_req = {
            7: -7.5,
            8: -10,
            9: -12.5,
            10: -15,
            11: -17.5,
            12: -20
        }
        self.sf_min = 7
        self.sf_max = 12
        self.pt_min = 2
        self.pt_max = 14
        self.adr_margin = 10

    @staticmethod
    def exponential_moving_average(snr_list, beta):
        """
        Calculate the Exponential Moving Average (EMA) of SNR values.

        Parameters:
        - snr_list (list): List of SNR values.
        - beta (float): Smoothing factor for EMA.

        Returns:
        - ema_value (float): Exponential Moving Average of SNR values.
        """
        #ema_values = [snr_list[0]]  # Initialize EMA with the first SNR value
        #for i in range(1, len(snr_list)):
        #    ema_values.append(beta * snr_list[i] + (1 - beta) * ema_values[i - 1])
        #ema_max = np.mean(ema_values)
        #return ema_max

        ema_value = snr_list[0]
        for snr in snr_list[1:]:
            ema_value = beta * snr + (1 - beta) * ema_value

        return ema_value

    @staticmethod
    def gaussian_filter(snr_list):
        """
        Apply Gaussian filter to SNR values.

        Parameters:
        - snr_list (list): List of SNR values.

        Returns:
        - average_snr (float): Average SNR after applying the Gaussian filter.
        """
        mean = np.mean(snr_list)
        variance = np.var(snr_list, ddof=1)
        sigma = np.sqrt(variance)
        effective_range = (mean - sigma, mean + sigma)
        filtered_values = [x for x in snr_list if effective_range[0] <= x <= effective_range[1]]
        average_snr = sum(filtered_values) / len(filtered_values)
        return average_snr

    @staticmethod
    def linear_regression(time_list, snr_list):
        """
        Perform linear regression on SNR values over time.

        Parameters:
        - time_list (list): List of time values.
        - snr_list (list): List of corresponding SNR values.

        Returns:
        - lr_value (float): Average SNR predicted by linear regression.
        """
        last_10_time = time_list[-10:]
        last_10_snr = snr_list[-10:]

        t_avg = np.mean(last_10_time)
        snr_avg = np.mean(last_10_snr)

        beta = np.sum((last_10_time - t_avg) * (last_10_snr - snr_avg)) / np.sum((last_10_time - t_avg) ** 2)
        alpha = snr_avg - (beta * t_avg)

        next_time = time_list[-1] + 10  # Assuming a fixed transmission period of 10
        next_snr = beta * next_time + alpha

        snr_list_lr = snr_list[1:]
        snr_list_lr = np.append(snr_list_lr, next_snr)
        lr_value = np.mean(snr_list_lr)
        return lr_value

    @staticmethod
    def blind_adr():
        """
        Perform Blind Adaptive Data Rate (ADR) by cycling through a predefined list of Spreading Factors (SF).
        """
        sf_list = [12, 7, 10, 7, 10, 7]
        tx_count = 0
        sf_iterator = cycle(sf_list)
        #for sf in sf_iterator:
            #tx_count += 1
            #print(f"Transmission {tx_count}: SF{sf}")

    @staticmethod
    def adr(snr_list):
        snr_max = max(snr_list)
        return snr_max

    @staticmethod
    def adr_plus(snr_list):
        snr_mean = np.mean(snr_list)
        return snr_mean


    def snr_margin_calculation(self, snr_value, sf):
        """
        Calculate the Signal-to-Noise Ratio (SNR) margin for a given SF.

        Parameters:
        - snr_value (float): Current SNR value.
        - sf (int): Spreading Factor.

        Returns:
        - snr_margin (float): SNR margin.
        """
        return snr_value - self.adr_snr_req[sf] - self.adr_margin

    def adjust_parameters(self, snr_mrg, sf, ptx):
        """
        Adjust SF and transmit power (PTX) based on the SNR margin.

        Parameters:
        - snr_mrg (float): SNR margin.
        - sf (int): Current Spreading Factor.
        - PTX (int): Current transmit power.

        Returns:
        - sf (int): Adjusted Spreading Factor.
        - PTX (int): Adjusted transmit power.
        """
        n_step = round(snr_mrg / 3)
        if n_step == 0:
            pass
        elif n_step > 0:
            while n_step > 0 and sf > self.sf_min:
                sf -= 1
                n_step -= 1
            while n_step > 0 and ptx > self.pt_min:
                ptx -= 3
                n_step -= 1
        elif n_step < 0:
            while n_step < 0 and ptx < self.pt_max:
                ptx += 3
                n_step += 1
            while n_step < 0 and sf < self.sf_max:
                sf += 1
                n_step += 1
        return sf, ptx


if __name__ == "__main__":
    adr = AdaptiveDataRate()
    # Example usage:
    snr_values = [19.45, 17.83, 14.35, 7.45, 12.34, 8.93, -3.12, -8.24, -12.66, -9.51, -15.39, -18.48, -12.53, -5.01,
                  -1.34, 4.23, 11.92, 15.38, 10.23, 5.23]  # Example SNR values
    sf_value = 7
    ptx_value = 11
    ema = adr.exponential_moving_average(snr_values, beta=0.7)
    gaussian = adr.gaussian_filter(snr_values)
    lr = adr.linear_regression(range(0, 210, 10), snr_values)
    snr_margin = adr.snr_margin_calculation(gaussian, sf_value)
    sf_value, ptx_value = adr.adjust_parameters(snr_margin, sf_value, ptx_value)
