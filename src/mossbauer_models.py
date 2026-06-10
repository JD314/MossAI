import lmfit
import numpy as np

def lorentzian(v, center, width, area):
    """
    Normalized Lorentzian profile (returns negative values for absorption).
    
    Args:
        v: np.ndarray - Doppler velocity axis (mm/s).
        center: float - Isomer shift center (mm/s).
        width: float - Full Width at Half Maximum (FWHM) (mm/s).
        area: float - Total integrated area under the peak.
    """
    denom = (v - center)**2 + (width/2.0)**2
    # Prevent division by zero
    denom = np.where(denom == 0.0, 1e-12, denom)
    return -area * (width/2.0)**2 / denom

def doublet_lorentzian(x, delta, quad, gamma, area):
    """
    Symmetric doublet curve for quadrupolar splitting.
    
    Args:
        x: np.ndarray - Doppler velocity axis (mm/s).
        delta: float - Isomer shift (mm/s).
        quad: float - Quadrupolar splitting delta_EQ (mm/s).
        gamma: float - FWHM linewidth (mm/s).
        area: float - Total integrated doublet area.
    """
    # The doublet consists of two lorentzians at delta - quad/2 and delta + quad/2, each sharing half the area.
    return (lorentzian(x, delta - 0.5 * quad, gamma, area / 2.0) +
            lorentzian(x, delta + 0.5 * quad, gamma, area / 2.0))

def sextet_lorentzian(x, delta, q_shift, B_hf, gamma, area):
    """
    Sextet model for magnetic hyperfine field splitting with quadrupolar perturbation.
    
    Args:
        x: np.ndarray - Doppler velocity axis (mm/s).
        delta: float - Isomer shift (mm/s).
        q_shift: float - Quadrupole shift epsilon (mm/s).
        B_hf: float - Magnetic hyperfine field splitting factor (mm/s).
        gamma: float - FWHM linewidth (mm/s).
        area: float - Total integrated sextet area.
    """
    d = q_shift / B_hf if B_hf != 0.0 else 0.0
    # Relative positions of the 6 lines
    positions = np.array([-1.0, -0.6 + d, -0.2, 0.2, 0.6 - d, 1.0])
    # Relative transition probabilities (3:2:1:1:2:3), normalized to sum to 12.0
    intensities = np.array([3.0, 2.0, 1.0, 1.0, 2.0, 3.0]) / 12.0
    
    y = np.zeros_like(x)
    for pos, inten in zip(positions, intensities):
        center = delta + pos * B_hf
        y += lorentzian(x, center, gamma, area * inten)
    return y

def build_model(model_idx: int) -> lmfit.Model:
    """
    Constructs the composite lmfit.Model corresponding to the given model index.
    
    Args:
        model_idx: int (0 to 5)
    """
    if model_idx == 0:    # 1S
        return lmfit.Model(lorentzian, prefix='s1_')
    elif model_idx == 1:  # 1D
        return lmfit.Model(doublet_lorentzian, prefix='d1_')
    elif model_idx == 2:  # 2D
        return lmfit.Model(doublet_lorentzian, prefix='d1_') + lmfit.Model(doublet_lorentzian, prefix='d2_')
    elif model_idx == 3:  # 1X
        return lmfit.Model(sextet_lorentzian, prefix='x1_')
    elif model_idx == 4:  # 1X+1D
        return lmfit.Model(sextet_lorentzian, prefix='x1_') + lmfit.Model(doublet_lorentzian, prefix='d1_')
    elif model_idx == 5:  # 2X
        return lmfit.Model(sextet_lorentzian, prefix='x1_') + lmfit.Model(sextet_lorentzian, prefix='x2_')
    else:
        raise ValueError(f"Invalid model index: {model_idx}")

def get_initial_params(model_idx: int) -> lmfit.Parameters:
    """
    Provides initial bounds and values for the parameters of the chosen sub-model.
    """
    params = lmfit.Parameters()
    
    if model_idx == 0:  # 1S
        params.add('s1_center', value=0.0, min=-2.0, max=2.0)
        params.add('s1_width', value=0.15, min=0.01, max=0.5)
        params.add('s1_area', value=0.1, min=0.0, max=1.0)
        
    elif model_idx == 1:  # 1D
        params.add('d1_delta', value=0.3, min=-0.7, max=1.5)
        params.add('d1_quad', value=1.0, min=0.0, max=2.5)
        params.add('d1_gamma', value=0.15, min=0.01, max=0.35)
        params.add('d1_area', value=0.2, min=0.0, max=1.0)
        
    elif model_idx == 2:  # 2D
        params.add('d1_delta', value=0.3, min=-0.7, max=1.5)
        params.add('d1_quad', value=1.0, min=0.0, max=2.5)
        params.add('d1_gamma', value=0.15, min=0.01, max=0.35)
        params.add('d1_area', value=0.2, min=0.0, max=1.0)
        
        params.add('d2_delta', value=0.4, min=-0.7, max=1.5)
        params.add('d2_quad', value=0.8, min=0.0, max=2.5)
        params.add('d2_gamma', value=0.15, min=0.01, max=0.35)
        params.add('d2_area', value=0.1, min=0.0, max=1.0)
        
    elif model_idx == 3:  # 1X
        params.add('x1_delta', value=0.5, min=-0.7, max=1.5)
        params.add('x1_q_shift', value=0.0, min=-0.5, max=0.5)
        params.add('x1_B_hf', value=2.5, min=0.0, max=8.5)
        params.add('x1_gamma', value=0.25, min=0.01, max=0.35)
        params.add('x1_area', value=0.2, min=0.0, max=1.0)
        
    elif model_idx == 4:  # 1X + 1D
        params.add('x1_delta', value=0.5, min=-0.7, max=1.5)
        params.add('x1_q_shift', value=0.0, min=-0.5, max=0.5)
        params.add('x1_B_hf', value=2.5, min=0.0, max=8.5)
        params.add('x1_gamma', value=0.25, min=0.01, max=0.35)
        params.add('x1_area', value=0.2, min=0.0, max=1.0)
        
        params.add('d1_delta', value=0.3, min=-0.7, max=1.5)
        params.add('d1_quad', value=1.0, min=0.0, max=2.5)
        params.add('d1_gamma', value=0.15, min=0.01, max=0.35)
        params.add('d1_area', value=0.1, min=0.0, max=1.0)
        
    elif model_idx == 5:  # 2X
        params.add('x1_delta', value=0.5, min=-0.7, max=1.5)
        params.add('x1_q_shift', value=0.0, min=-0.5, max=0.5)
        params.add('x1_B_hf', value=2.5, min=0.0, max=8.5)
        params.add('x1_gamma', value=0.25, min=0.01, max=0.35)
        params.add('x1_area', value=0.2, min=0.0, max=1.0)
        
        params.add('x2_delta', value=0.4, min=-0.7, max=1.5)
        params.add('x2_q_shift', value=0.0, min=-0.5, max=0.5)
        params.add('x2_B_hf', value=3.0, min=0.0, max=8.5)
        params.add('x2_gamma', value=0.25, min=0.01, max=0.35)
        params.add('x2_area', value=0.1, min=0.0, max=1.0)
        
    return params

def evaluate_model(params, x, model_idx: int) -> np.ndarray:
    """
    Evaluates the model function directly using numpy to avoid lmfit overhead.
    """
    p = params
    if model_idx == 0:
        return lorentzian(x, p['s1_center'].value, p['s1_width'].value, p['s1_area'].value)
    elif model_idx == 1:
        return doublet_lorentzian(x, p['d1_delta'].value, p['d1_quad'].value, p['d1_gamma'].value, p['d1_area'].value)
    elif model_idx == 2:
        return (doublet_lorentzian(x, p['d1_delta'].value, p['d1_quad'].value, p['d1_gamma'].value, p['d1_area'].value) +
                doublet_lorentzian(x, p['d2_delta'].value, p['d2_quad'].value, p['d2_gamma'].value, p['d2_area'].value))
    elif model_idx == 3:
        return sextet_lorentzian(x, p['x1_delta'].value, p['x1_q_shift'].value, p['x1_B_hf'].value, p['x1_gamma'].value, p['x1_area'].value)
    elif model_idx == 4:
        return (sextet_lorentzian(x, p['x1_delta'].value, p['x1_q_shift'].value, p['x1_B_hf'].value, p['x1_gamma'].value, p['x1_area'].value) +
                doublet_lorentzian(x, p['d1_delta'].value, p['d1_quad'].value, p['d1_gamma'].value, p['d1_area'].value))
    elif model_idx == 5:
        return (sextet_lorentzian(x, p['x1_delta'].value, p['x1_q_shift'].value, p['x1_B_hf'].value, p['x1_gamma'].value, p['x1_area'].value) +
                sextet_lorentzian(x, p['x2_delta'].value, p['x2_q_shift'].value, p['x2_B_hf'].value, p['x2_gamma'].value, p['x2_area'].value))
    return np.zeros_like(x)

def fit_spectrum_model(velocity: np.ndarray, intensity: np.ndarray,
                       model_idx: int) -> lmfit.minimizer.MinimizerResult:
    """
    Fits a single submodel to the velocity and intensity vectors.
    """
    params = get_initial_params(model_idx)
    
    def residual(p, x, y):
        # Since lorentzians return negative values, we sum them to baseline 1.0
        y_fit = 1.0 + evaluate_model(p, x, model_idx)
        return y_fit - y
        
    # Fit using leastsq (Levenberg-Marquardt)
    result = lmfit.minimize(residual, params, args=(velocity, intensity), method='leastsq', max_nfev=150)
    return result

def select_best_model(velocity: np.ndarray,
                      intensity: np.ndarray) -> tuple[int, lmfit.minimizer.MinimizerResult]:
    """
    Evaluates the 6 sub-models on the data and selects the best by minimizing the Bayesian Information Criterion (BIC).
    """
    bic_scores, results = [], []
    for idx in range(6):
        try:
            result = fit_spectrum_model(velocity, intensity, idx)
            bic_scores.append(result.bic)
            results.append(result)
        except Exception:
            # Fallback high BIC value in case of fitting crash
            bic_scores.append(1e12)
            results.append(None)
            
    best_idx = int(np.argmin(bic_scores))
    return best_idx, results[best_idx]

def extract_hyperfine_vector(result: lmfit.minimizer.MinimizerResult, model_idx: int) -> list[float]:
    """
    Extracts the hyperfine vector P of length 15 from the fitted parameters.
    P_vec = [delta_1, Delta_1, B_hf_1, gamma_1, A_1,
             delta_2, Delta_2, B_hf_2, gamma_2, A_2,
             delta_3, Delta_3, B_hf_3, gamma_3, A_3]
    Missing parameters are filled with 0.0.
    """
    p_vec = [0.0] * 15
    if result is None:
        return p_vec
        
    p = result.params
    
    if model_idx == 0:  # 1S
        p_vec[0] = float(p['s1_center'].value)
        p_vec[1] = 0.0
        p_vec[2] = 0.0
        p_vec[3] = float(p['s1_width'].value)
        p_vec[4] = float(p['s1_area'].value)
        
    elif model_idx == 1:  # 1D
        p_vec[0] = float(p['d1_delta'].value)
        p_vec[1] = float(p['d1_quad'].value)
        p_vec[2] = 0.0
        p_vec[3] = float(p['d1_gamma'].value)
        p_vec[4] = float(p['d1_area'].value)
        
    elif model_idx == 2:  # 2D
        # Component 1
        p_vec[0] = float(p['d1_delta'].value)
        p_vec[1] = float(p['d1_quad'].value)
        p_vec[2] = 0.0
        p_vec[3] = float(p['d1_gamma'].value)
        p_vec[4] = float(p['d1_area'].value)
        # Component 2
        p_vec[5] = float(p['d2_delta'].value)
        p_vec[6] = float(p['d2_quad'].value)
        p_vec[7] = 0.0
        p_vec[8] = float(p['d2_gamma'].value)
        p_vec[9] = float(p['d2_area'].value)
        
    elif model_idx == 3:  # 1X
        p_vec[0] = float(p['x1_delta'].value)
        p_vec[1] = float(p['x1_q_shift'].value)
        p_vec[2] = float(p['x1_B_hf'].value)
        p_vec[3] = float(p['x1_gamma'].value)
        p_vec[4] = float(p['x1_area'].value)
        
    elif model_idx == 4:  # 1X + 1D
        # Component 1 (Sextet)
        p_vec[0] = float(p['x1_delta'].value)
        p_vec[1] = float(p['x1_q_shift'].value)
        p_vec[2] = float(p['x1_B_hf'].value)
        p_vec[3] = float(p['x1_gamma'].value)
        p_vec[4] = float(p['x1_area'].value)
        # Component 2 (Doublet)
        p_vec[5] = float(p['d1_delta'].value)
        p_vec[6] = float(p['d1_quad'].value)
        p_vec[7] = 0.0
        p_vec[8] = float(p['d1_gamma'].value)
        p_vec[9] = float(p['d1_area'].value)
        
    elif model_idx == 5:  # 2X
        # Component 1 (Sextet 1)
        p_vec[0] = float(p['x1_delta'].value)
        p_vec[1] = float(p['x1_q_shift'].value)
        p_vec[2] = float(p['x1_B_hf'].value)
        p_vec[3] = float(p['x1_gamma'].value)
        p_vec[4] = float(p['x1_area'].value)
        # Component 2 (Sextet 2)
        p_vec[5] = float(p['x2_delta'].value)
        p_vec[6] = float(p['x2_q_shift'].value)
        p_vec[7] = float(p['x2_B_hf'].value)
        p_vec[8] = float(p['x2_gamma'].value)
        p_vec[9] = float(p['x2_area'].value)
        
    return p_vec
