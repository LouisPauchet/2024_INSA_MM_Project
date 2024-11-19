def kinematic_hardening(p, C, gamma):
    """
    Kinematic hardening rule for the Chaboche model.
    Parameters:
        p (float): Accumulated plastic strain
        C (float): Kinematic hardening modulus
        gamma (float): Saturation rate for kinematic hardening
    Returns:
        X (float): Backstress
    """
    return (C / gamma) * (1 - np.exp(-gamma * p))


def isotropic_hardening(p, Q, b):
    """
    Isotropic hardening rule for the Chaboche model.
    Parameters:
        p (float): Accumulated plastic strain
        Q (float): Saturation value for isotropic hardening
        b (float): Saturation rate for isotropic hardening
    Returns:
        R (float): Isotropic hardening
    """
    return Q * (1 - np.exp(-b * p))


def simulate_chaboche(E, sigma_y, C, gamma, Q, b, strain_input):
    """
    Simulates the stress-strain response of a material using the Chaboche model.

    Parameters:
        E (float): Young's modulus [MPa]
        sigma_y (float): Initial yield stress [MPa]
        C (float): Kinematic hardening modulus [MPa]
        gamma (float): Saturation rate for kinematic hardening
        Q (float): Saturation value for isotropic hardening [MPa]
        b (float): Saturation rate for isotropic hardening
        strain_input (array): Array of strain values (load path)

    Returns:
        stress_output (array): Simulated stress values [MPa]
    """
    # Initialize outputs and state variables
    stress_output = []
    plastic_strain = 0.0  # Initial plastic strain
    backstress = 0.0  # Initial backstress (X)
    p = 0.0  # Accumulated plastic strain (p)

    # Loop through the strain inputs
    for strain in strain_input:
        # Trial stress from the elastic response
        trial_stress = E * (strain - plastic_strain)

        # Effective stress considering the backstress
        effective_stress = abs(trial_stress - backstress)

        # Yield stress incorporating isotropic hardening
        R = isotropic_hardening(p, Q, b)
        yield_stress = sigma_y + R

        if effective_stress > yield_stress:
            # Plastic deformation occurs
            dp = (effective_stress - yield_stress) / (E + C)  # Increment in plastic strain
            p += dp  # Update cumulative plastic strain

            # Update backstress (kinematic hardening)
            backstress = kinematic_hardening(p, C, gamma)

            # Update plastic strain
            plastic_strain += dp * np.sign(trial_stress - backstress)

            # Correct the stress after plasticity
            stress = trial_stress - (trial_stress - backstress) * dp
        else:
            # Elastic response
            stress = trial_stress

        # Append stress to output
        stress_output.append(stress)

    return np.array(stress_output)