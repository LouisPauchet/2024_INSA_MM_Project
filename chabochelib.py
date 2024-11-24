from scipy.stats import linregress
import numpy as np

def kinematic_hardening(p, C, gamma):
    """
    Kinematic hardening rule for the Chaboche model.

    Parameters:
        p (float): Accumulated plastic strain.
        C (float): Kinematic hardening modulus.
        gamma (float): Saturation rate for kinematic hardening.

    Returns:
        float: Backstress (X) from kinematic hardening.
    """
    return (C / gamma) * (1 - np.exp(-gamma * p))


def isotropic_hardening(p, Q, b):
    """
    Isotropic hardening rule for the Chaboche model.

    Parameters:
        p (float): Accumulated plastic strain.
        Q (float): Saturation value for isotropic hardening.
        b (float): Saturation rate for isotropic hardening.

    Returns:
        float: Isotropic hardening (R).
    """
    return Q * (1 - np.exp(-b * p))


def sim_chaboche(E, sigma_y, C, gamma, Q, b, strain_input):
    """
    Simulates the Chaboche model under strain-controlled conditions.

    Parameters:
        E (float): Young's modulus.
        sigma_y (float): Initial yield stress.
        C (float): Kinematic hardening modulus.
        gamma (float): Saturation rate for kinematic hardening.
        Q (float): Saturation value for isotropic hardening.
        b (float): Saturation rate for isotropic hardening.
        strain_input (list or np.array): Sequence of imposed strain values.

    Returns:
        np.array: Corresponding stress values.
    """
    stress_output = []
    X = 0  # Kinematic hardening variable
    R = 0  # Isotropic hardening variable
    p = 0  # Accumulated plastic strain

    sigma_prev = 0
    epsilon_prev = 0

    for i, epsilon in enumerate(strain_input):
        # Trial stress
        sigma_trial = E * (epsilon - epsilon_prev) + sigma_prev

        # Yield condition
        yield_condition = abs(sigma_trial - X) - (sigma_y + R)

        if yield_condition > 0:  # Plastic loading
            dp = yield_condition / (E + C + b * Q)
            p += dp  # Update cumulative plastic strain

            # Update isotropic hardening
            R += b * (Q - R) * dp

            # Update kinematic hardening
            X += C * dp * np.sign(sigma_trial - X) - gamma * X * dp

            # Compute plastic stress
            sigma = X + (sigma_y + R) * np.sign(sigma_trial - X)
        else:  # Elastic loading
            sigma = sigma_trial

        # Save stress and update history variables
        stress_output.append(sigma)
        sigma_prev = sigma
        epsilon_prev = epsilon

    return np.array(stress_output)


def sim_chaboche_stress_control(E, sigma_y, C, gamma, Q, b, stress_input):
    """
    Simulates the Chaboche model under stress-controlled conditions.

    Parameters:
        E (float): Young's modulus.
        sigma_y (float): Initial yield stress.
        C (float): Kinematic hardening modulus.
        gamma (float): Kinematic hardening parameter.
        Q (float): Saturation value for isotropic hardening.
        b (float): Isotropic hardening rate.
        stress_input (list or np.array): Sequence of imposed stress values.

    Returns:
        np.array: Corresponding strain values.
    """
    strain_output = []
    X = 0  # Kinematic hardening variable
    R = 0  # Isotropic hardening variable
    p = 0  # Accumulated plastic strain

    sigma_prev = 0
    epsilon_prev = 0

    for i, sigma in enumerate(stress_input):
        # Trial strain
        epsilon_trial = epsilon_prev + (sigma - sigma_prev) / E

        # Yield condition
        yield_condition = abs(sigma - X) - (sigma_y + R)

        if yield_condition > 0:  # Plastic loading
            dp = yield_condition / (E + C + b * Q)
            p += dp  # Update cumulative plastic strain

            # Update isotropic hardening
            R += b * (Q - R) * dp

            # Update kinematic hardening
            X += C * dp * np.sign(sigma - X) - gamma * X * dp

            # Correct strain for plasticity
            epsilon = epsilon_trial + dp * np.sign(sigma - X)
        else:  # Elastic loading
            epsilon = epsilon_trial

        # Save strain and update history variables
        strain_output.append(epsilon)
        sigma_prev = sigma
        epsilon_prev = epsilon

    return np.array(strain_output)


def get_sigma_y(df, lim=2):
    """
    Determines the yield stress from a stress-strain dataset.

    Parameters:
        df (pd.DataFrame): DataFrame containing stress ('Contrainte (MPa)') and strain ('Déformation').
        lim (float): Deviation limit to detect yield point.

    Returns:
        tuple: Yield stress (sigma_y), strain at yield, original deformation, predicted elastic stress,
               elastic slope, strain plot values, and stress plot values.
    """
    deformation = df["Déformation"]
    stress = df["Contrainte (MPa)"]

    # Linear fit for the elastic region
    elastic_region = df[deformation < 0.002]
    slope, intercept, _, _, _ = linregress(elastic_region["Déformation"], elastic_region["Contrainte (MPa)"])

    # Calculate the predicted stress based on the elastic fit
    predicted_stress = slope * deformation + intercept

    # Find the point where the curve deviates significantly
    deviation = np.abs(stress - predicted_stress)
    yield_index = deviation[deviation > lim].idxmin()

    sigma_y = stress.loc[yield_index]  # Yield stress value
    strain_y = deformation.loc[yield_index]  # Strain at yield

    strain_plot = np.linspace(deformation.min(), deformation.max(), 100)
    stress_plot = slope * strain_plot + intercept

    return sigma_y, strain_y, deformation, predicted_stress, slope, strain_plot, stress_plot


def format_numbers_dynamic(df, min_value=1e-2, max_value=1e3):
    """
    Formats numbers in a DataFrame dynamically based on their range.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        min_value (float): Minimum value for decimal formatting.
        max_value (float): Maximum value for decimal formatting.

    Returns:
        pd.DataFrame: DataFrame with formatted numbers as strings.
    """
    def dynamic_format(x):
        if isinstance(x, (int, float)):
            if abs(x) < min_value or abs(x) > max_value:
                return f"{x:.2e}"  # Scientific notation for values outside the range
            return f"{x:.2f}"  # Decimal format for values within the range
        return x  # Non-numeric values remain unchanged

    return df.applymap(dynamic_format)


def get_sigma_cycle(x, y, strain, stress, slope, lim):
    """
    Determines the yield stress and strain based on the stress-strain curve.

    Parameters:
        x (pd.Series): Subset of strain values for fitting.
        y (pd.Series): Subset of stress values for fitting.
        strain (pd.Series): Full strain values for deviation calculation.
        stress (pd.Series): Full stress values for deviation calculation.
        slope (float): Elastic modulus (slope of the elastic line).
        lim (float): Threshold for significant deviation.

    Returns:
        tuple: Yield stress (sigma_y), strain at yield, and elastic intercept.
    """
    intercept = np.mean(y) - slope * np.mean(x)
    deviation = np.abs(stress - (slope * strain + intercept))
    yield_indices = deviation[deviation > lim].index
    if not yield_indices.empty:
        yield_index = yield_indices[0]
    else:
        raise ValueError("No significant deviation found; adjust the 'lim' parameter.")

    sigma_y = stress.loc[yield_index]
    strain_y = strain.loc[yield_index]

    return sigma_y, strain_y, intercept

def generate_cycle(num_cycles, points_per_cycle=150, amplitude=300, average=0, phase_shift=0.75):
    """
    Generates a cyclic loading signal (stress) based on the specified parameters.

    Parameters:
        num_cycles (int): Number of cycles to generate.
        points_per_cycle (int): Number of data points per cycle.
        amplitude (float): Amplitude of the cyclic stress.
        average (float): Average stress level.
        phase_shift (float): Phase shift to control the starting point of the cycle.

    Returns:
        np.array: Generated stress values.
        np.array: Corresponding time values.
    """
    # Generate time values
    t = np.linspace(0, num_cycles, num_cycles * points_per_cycle)

    # Generate triangular cyclic stress waveform
    stress_given = amplitude * (np.abs(((t + phase_shift) % 1) - 0.5) * 2) - (amplitude / 2 - average)

    return stress_given, t
