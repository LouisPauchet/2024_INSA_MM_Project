from fontTools.misc.bezierTools import epsilon
from scipy.stats import linregress
import numpy as np


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
    Simule la réponse contrainte-déformation d'un matériau en utilisant le modèle de Chaboche.
    Simulates the stress-strain response of a material using the Chaboche model.

    Paramètres | Parameters :
        E (float): Module de Young [MPa] | Young's modulus [MPa]
        sigma_y (float): Limite d'élasticité initiale [MPa] | Initial yield stress [MPa]
        C (float): Module d'écrouissage cinématique [MPa] | Kinematic hardening modulus [MPa]
        gamma (float): Taux de saturation pour l'écrouissage cinématique | Saturation rate for kinematic hardening
        Q (float): Valeur de saturation pour l'écrouissage isotrope [MPa] | Saturation value for isotropic hardening [MPa]
        b (float): Taux de saturation pour l'écrouissage isotrope | Saturation rate for isotropic hardening
        strain_input (array): Tableau des valeurs de déformation imposée | Array of strain values (load path)

    Retourne | Returns :
        stress_output (array): Valeurs de contrainte simulées [MPa] | Simulated stress values [MPa]
    """
    # Initialisation des variables d'état et des sorties
    # Initialize state variables and outputs
    stress_output = []
    plastic_strain = 0.0  # Déformation plastique initiale | Initial plastic strain
    backstress = 0.0  # Backstress initial (X) | Initial backstress (X)
    p = 0.0  # Déformation plastique cumulée | Accumulated plastic strain (p)

    # Boucle sur les déformations
    # Loop through the strain input values
    for strain in strain_input:
        # Calcul de la contrainte d'essai à partir de la réponse élastique
        # Compute the trial stress based on the elastic response
        trial_stress = E * (strain - plastic_strain)

        # Calcul de la contrainte effective tenant compte de l'écrouissage cinématique
        # Compute the effective stress considering kinematic hardening
        effective_stress = abs(trial_stress - backstress)

        # Calcul de la limite d'élasticité avec l'écrouissage isotrope
        # Compute the yield stress incorporating isotropic hardening
        R = isotropic_hardening(p, Q, b)
        yield_stress = sigma_y + R

        if effective_stress > yield_stress:
            # Une déformation plastique se produit
            # Plastic deformation occurs

            # Calcul de l'incrément de déformation plastique
            # Calculate plastic strain increment
            dp = (effective_stress - yield_stress) / E

            # Mise à jour de la déformation plastique cumulée
            # Update cumulative plastic strain
            p += dp

            # Mise à jour de la contrainte interne (écrouissage cinématique)
            # Update backstress (kinematic hardening)
            backstress += C * dp * np.sign(trial_stress - backstress)

            # Mise à jour de la déformation plastique
            # Update the plastic strain
            plastic_strain += dp * np.sign(trial_stress - backstress)

            # Correction de la contrainte après plasticité
            # Correct the stress after plasticity
            stress = sigma_y + R + backstress * np.sign(trial_stress - backstress)
        else:
            # Réponse élastique
            # Elastic response
            stress = trial_stress

        # Ajouter la contrainte simulée à la sortie
        # Append the simulated stress to the output
        stress_output.append(stress)

    # Retourner les contraintes simulées sous forme de tableau NumPy
    # Return the simulated stresses as a NumPy array
    return np.array(stress_output)


def sim_chaboche(E, sigma_y, C, gamma, Q, b, strain_input):

    stress_output = []

    sigma_max_i = 0
    epsilon_max_i = 0

    epsilon_elastic_i = 0

    sigma_max_elastic = 0

    p = 0


    for i, epsilon in enumerate(strain_input):


        sigma = sigma_max_i + (E * (epsilon - epsilon_max_i))
        sigma_y_i = sigma_y + isotropic_hardening(p, Q, b) #sigma_y + R


        if (np.abs(sigma) > sigma_y_i) and (np.abs(epsilon_max_i) < np.abs(epsilon)):

            p = p + np.abs(epsilon - strain_input[i - 1])

            X = kinematic_hardening(p, C, gamma)

            sigma_plastic = X * np.sign(sigma_max_elastic) + sigma_max_elastic

            stress_output.append(sigma_plastic)

            epsilon_max_i = epsilon
            sigma_max_i = sigma_plastic

        else :
            epsilon_elastic_i = epsilon
            sigma_max_elastic = sigma
            stress_output.append(sigma)

    return np.array(stress_output)




def get_sigma_y(df, lim=2):
    deformation = df["Déformation"]
    stress = df["Contrainte (MPa)"]

    # Linear fit for the elastic region (assuming strain < 0.002 is elastic)
    elastic_region = df[deformation < 0.002]
    slope, intercept, _, _, _ = linregress(elastic_region["Déformation"], elastic_region["Contrainte (MPa)"])

    # Calculate the predicted stress based on the elastic fit
    predicted_stress = slope * deformation + intercept

    # Find the point where the curve deviates significantly (plastic region starts)
    deviation = np.abs(stress - predicted_stress)
    yield_index = deviation[deviation > lim].idxmin()  # Identify index of maximum deviation

    sigma_y = stress.loc[yield_index]  # Yield stress value
    strain_y = deformation.loc[yield_index]  # Strain at yield

    return sigma_y, strain_y, deformation, predicted_stress, slope

def get_sigma_cycle(x, y, strain, stress, slope, lim):
    """
    Calculate the yield point (sigma_y and strain_y) based on the deviation
    of the stress-strain curve from the elastic fit.

    Parameters:
        x (Series): Subset of strain values for fitting (independent variable).
        y (Series): Subset of stress values for fitting (dependent variable).
        strain (Series): Full strain values for deviation calculation.
        stress (Series): Full stress values for deviation calculation.
        slope (float): Elastic modulus (slope of the elastic line).
        lim (float): Threshold for significant deviation to identify the yield point.

    Returns:
        sigma_y (float): Yield stress value.
        strain_y (float): Strain value at the yield point.
        intercept (float): Calculated y-intercept of the elastic fit.
    """
    # Calculate the y-intercept (b) for the elastic fit
    intercept = np.mean(y) - slope * np.mean(x)

    # Calculate the deviation of the stress from the elastic fit
    deviation = np.abs(stress - (slope * strain + intercept))

    # Find the first point where the deviation exceeds the threshold
    yield_indices = deviation[deviation > lim].index
    if not yield_indices.empty:
        yield_index = yield_indices[0]  # Use the first index exceeding the threshold
    else:
        raise ValueError("No significant deviation found; adjust the 'lim' parameter.")

    sigma_y = stress.loc[yield_index]  # Yield stress
    strain_y = strain.loc[yield_index]  # Yield strain

    return sigma_y, strain_y, intercept