#!/usr/bin/env python3
"""
================================================================================
TAYLOR SERIES: A VISUAL INTUITION FOR BIOLOGISTS
================================================================================

Author: Daniel Ortiz-Barrientos (with Claude)
Purpose: Build deep intuition for Taylor series through visualization

THE CORE IDEA
-------------
Imagine you're blindfolded at a point on a landscape. You can feel:
  - Where you are (the function value)
  - The slope under your feet (first derivative)  
  - Whether the ground curves up or down (second derivative)
  - Even subtler curvatures (higher derivatives)

Taylor series says: if you know ALL these local properties at one point,
you can reconstruct the ENTIRE landscape (within some radius).

This is profound! Local information → Global reconstruction.

For biologists: Think of it as knowing everything about a population's 
current state (size, growth rate, acceleration of growth, etc.) and using
that to predict future dynamics.

================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from math import factorial
from typing import Callable, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set up beautiful plot styling
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'figure.dpi': 120,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Define a pleasing color palette
COLORS = {
    'true_function': '#2C3E50',      # Dark blue-gray for the true function
    'approximations': ['#E74C3C', '#E67E22', '#F1C40F', '#2ECC71', '#3498DB', '#9B59B6'],
    'error': '#E74C3C',
    'highlight': '#3498DB',
    'biological': '#27AE60',
}


# =============================================================================
# SECTION 1: THE FUNDAMENTAL BUILDING BLOCKS
# =============================================================================

def compute_taylor_approximation(
    f: Callable,           # The function to approximate
    derivatives: List[Callable],  # List of derivative functions [f', f'', f''', ...]
    a: float,              # The expansion point (where we "stand")
    x: np.ndarray,         # Points where we want to evaluate the approximation
    n_terms: int           # How many terms to include (1 = constant, 2 = linear, etc.)
) -> np.ndarray:
    """
    Compute the Taylor series approximation of f(x) around point a.
    
    THE MATHEMATICS
    ---------------
    f(x) ≈ f(a) + f'(a)(x-a) + f''(a)(x-a)²/2! + f'''(a)(x-a)³/3! + ...
    
    Each term adds information:
      - Term 0: f(a)           → "Where am I?"
      - Term 1: f'(a)(x-a)     → "Which way am I heading?"
      - Term 2: f''(a)(x-a)²/2 → "Am I speeding up or slowing down?"
      - Term n: ...            → "What are the subtler dynamics?"
    
    BIOLOGICAL INTUITION
    --------------------
    Think of tracking a population N(t):
      - N(t₀) is the current population size
      - N'(t₀) is the current growth rate
      - N''(t₀) tells us if growth is accelerating (boom) or decelerating (bust)
    
    Parameters
    ----------
    f : callable
        The true function we're approximating
    derivatives : list of callables
        [f', f'', f''', ...] - the derivative functions
    a : float
        Expansion point (where we have "perfect local knowledge")
    x : ndarray
        Points where we evaluate the approximation
    n_terms : int
        Number of terms in the polynomial (degree + 1)
    
    Returns
    -------
    ndarray
        The Taylor polynomial evaluated at each point in x
    """
    # Start with the zeroth term: just the function value at a
    # This is the "constant approximation" - assumes f(x) ≈ f(a) everywhere
    result = np.full_like(x, f(a), dtype=float)
    
    # Add each subsequent term
    for n in range(1, n_terms):
        if n - 1 < len(derivatives):
            # The nth term is: f^(n)(a) * (x-a)^n / n!
            # 
            # Why divide by n! (factorial)?
            # ---------------------------------
            # When you differentiate (x-a)^n repeatedly n times, you get:
            #   d/dx (x-a)^n = n(x-a)^(n-1)
            #   d²/dx² (x-a)^n = n(n-1)(x-a)^(n-2)
            #   ...
            #   d^n/dx^n (x-a)^n = n! (a constant)
            #
            # The n! in the denominator "undoes" this, so when we differentiate
            # the Taylor series n times and evaluate at x=a, we recover f^(n)(a).
            # This is how Taylor series is DERIVED - by matching derivatives!
            
            coefficient = derivatives[n-1](a) / factorial(n)
            term = coefficient * (x - a)**n
            result += term
    
    return result


def numerical_derivative(f: Callable, x: float, order: int = 1, h: float = 1e-5) -> float:
    """
    Compute numerical derivatives using finite differences.
    
    This is useful when we don't have analytical derivatives available.
    
    THE IDEA
    --------
    The derivative is defined as: f'(x) = lim[h→0] (f(x+h) - f(x-h)) / (2h)
    
    We approximate this with small but finite h. The "central difference"
    formula (f(x+h) - f(x-h)) / 2h is more accurate than the "forward 
    difference" (f(x+h) - f(x)) / h because errors cancel symmetrically.
    
    For higher derivatives, we apply this recursively.
    """
    if order == 0:
        return f(x)
    elif order == 1:
        return (f(x + h) - f(x - h)) / (2 * h)
    else:
        # Recursive: derivative of derivative
        def f_lower(t):
            return numerical_derivative(f, t, order - 1, h)
        return (f_lower(x + h) - f_lower(x - h)) / (2 * h)


# =============================================================================
# SECTION 2: VISUALIZATION - BUILDING INTUITION
# =============================================================================

def plot_taylor_convergence(
    f: Callable,
    derivatives: List[Callable],
    f_name: str,
    a: float,
    x_range: Tuple[float, float],
    max_terms: int = 6,
    figsize: Tuple[int, int] = (14, 5)
) -> plt.Figure:
    """
    Visualize how Taylor approximations improve as we add more terms.
    
    This is the KEY visualization for building intuition. Watch how:
    1. The constant term (n=1) is just a horizontal line at f(a)
    2. The linear term (n=2) adds a slope - the tangent line
    3. The quadratic term (n=3) adds curvature - a parabola
    4. Each term captures finer details of the function's shape
    
    The approximations are EXACT at x=a (by construction) and become
    less accurate as we move away. But with more terms, accuracy extends
    further from a.
    """
    x = np.linspace(x_range[0], x_range[1], 500)
    y_true = f(x)
    
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(1, 2, width_ratios=[1.5, 1], wspace=0.3)
    
    # LEFT PANEL: The approximations overlaid on the true function
    ax1 = fig.add_subplot(gs[0])
    
    # Plot the true function with emphasis
    ax1.plot(x, y_true, color=COLORS['true_function'], linewidth=3, 
             label=f'True: {f_name}', zorder=10)
    
    # Mark the expansion point
    ax1.scatter([a], [f(a)], color=COLORS['highlight'], s=150, zorder=15,
                edgecolor='white', linewidth=2, label=f'Expansion point a={a}')
    
    # Plot each approximation
    for n in range(1, max_terms + 1):
        y_approx = compute_taylor_approximation(f, derivatives, a, x, n)
        
        # Clip extreme values for visualization
        y_approx = np.clip(y_approx, y_true.min() - 2, y_true.max() + 2)
        
        color = COLORS['approximations'][(n-1) % len(COLORS['approximations'])]
        label = f'n={n}: ' + ['constant', 'linear', 'quadratic', 
                              'cubic', 'quartic', 'quintic'][n-1] if n <= 6 else f'n={n}'
        ax1.plot(x, y_approx, color=color, linewidth=1.5, alpha=0.8,
                 linestyle='--', label=label)
    
    ax1.set_xlabel('x')
    ax1.set_ylabel('f(x)')
    ax1.set_title(f'Taylor Series Approximations of {f_name} around a={a}')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.set_ylim(y_true.min() - 1, y_true.max() + 1)
    
    # Add annotation explaining the key insight
    ax1.annotate('All approximations\npass through this point!',
                xy=(a, f(a)), xytext=(a + 0.5*(x_range[1]-a), f(a) + 0.5),
                fontsize=9, ha='left',
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
    
    # RIGHT PANEL: Error analysis
    ax2 = fig.add_subplot(gs[1])
    
    # Compute and plot errors for each approximation order
    x_error = np.linspace(x_range[0], x_range[1], 200)
    y_true_error = f(x_error)
    
    for n in range(1, max_terms + 1):
        y_approx = compute_taylor_approximation(f, derivatives, a, x_error, n)
        error = np.abs(y_true_error - y_approx)
        
        # Use log scale, but handle zeros
        error = np.maximum(error, 1e-16)
        
        color = COLORS['approximations'][(n-1) % len(COLORS['approximations'])]
        ax2.semilogy(x_error, error, color=color, linewidth=1.5, 
                     label=f'n={n}', alpha=0.8)
    
    ax2.axvline(x=a, color='gray', linestyle=':', alpha=0.5)
    ax2.set_xlabel('x')
    ax2.set_ylabel('Absolute Error (log scale)')
    ax2.set_title('Error decreases with more terms')
    ax2.legend(loc='upper right', fontsize=9)
    
    plt.tight_layout()
    return fig


def plot_biological_application_population() -> plt.Figure:
    """
    Demonstrate Taylor series in a biological context: population dynamics.
    
    THE BIOLOGICAL SCENARIO
    -----------------------
    Consider a population following logistic growth:
        dN/dt = rN(1 - N/K)
    
    where:
        N = population size
        r = intrinsic growth rate
        K = carrying capacity
    
    The solution is: N(t) = K / (1 + ((K-N₀)/N₀)e^(-rt))
    
    Near the equilibrium at N=K, we can use Taylor series to understand
    local stability. This "linearization" is fundamental to stability analysis
    in ecology, epidemiology, and systems biology.
    
    INTUITION
    ---------
    At N=K, the population is at carrying capacity. Small perturbations
    (N slightly above or below K) will decay back to K if the system is stable.
    The Taylor series (specifically, the linear term) tells us HOW FAST this
    decay happens.
    """
    # Parameters for logistic growth
    r = 0.5      # Intrinsic growth rate
    K = 100      # Carrying capacity
    N0 = 10      # Initial population
    
    # Time array
    t = np.linspace(0, 20, 500)
    
    # True logistic growth solution
    def logistic(t):
        return K / (1 + ((K - N0) / N0) * np.exp(-r * t))
    
    N_true = logistic(t)
    
    # Derivatives for Taylor expansion around t=0
    # N(t) at t=0 is N0
    # dN/dt at t=0 is r*N0*(1 - N0/K)
    # d²N/dt² requires more work but we can compute numerically
    
    dN_dt_0 = r * N0 * (1 - N0/K)
    
    # For second derivative, we differentiate dN/dt = rN(1-N/K) with respect to t
    # d²N/dt² = r(dN/dt)(1-N/K) + rN(-1/K)(dN/dt) = r(dN/dt)(1 - 2N/K)
    d2N_dt2_0 = r * dN_dt_0 * (1 - 2*N0/K)
    
    # Taylor approximations around t=0
    N_linear = N0 + dN_dt_0 * t
    N_quadratic = N0 + dN_dt_0 * t + (d2N_dt2_0 / 2) * t**2
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # LEFT: Full population dynamics
    ax1 = axes[0]
    ax1.plot(t, N_true, color=COLORS['biological'], linewidth=3, label='True logistic growth')
    ax1.plot(t, N_linear, color=COLORS['approximations'][0], linewidth=2, 
             linestyle='--', label='Linear Taylor (tangent)')
    ax1.plot(t, N_quadratic, color=COLORS['approximations'][1], linewidth=2,
             linestyle='--', label='Quadratic Taylor')
    ax1.axhline(y=K, color='gray', linestyle=':', alpha=0.7, label=f'Carrying capacity K={K}')
    ax1.scatter([0], [N0], color=COLORS['highlight'], s=100, zorder=10,
                edgecolor='white', linewidth=2)
    
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Population size N(t)')
    ax1.set_title('Logistic Growth: True vs Taylor Approximations')
    ax1.legend(loc='lower right')
    ax1.set_ylim(0, K * 1.2)
    
    # Add biological interpretation
    ax1.annotate('Early growth phase:\nTaylor works well here',
                xy=(3, logistic(3)), xytext=(5, 30),
                fontsize=9, ha='left',
                arrowprops=dict(arrowstyle='->', color='gray'))
    ax1.annotate('Near carrying capacity:\nLinearization useful for\nstability analysis',
                xy=(15, K), xytext=(12, K*0.7),
                fontsize=9, ha='left',
                arrowprops=dict(arrowstyle='->', color='gray'))
    
    # RIGHT: Stability analysis near equilibrium
    ax2 = axes[1]
    
    # Analyze behavior near N=K (the equilibrium)
    # Let n = N - K (deviation from equilibrium)
    # dN/dt = rN(1-N/K) = r(K+n)(1-(K+n)/K) = r(K+n)(-n/K) = -rn(1 + n/K)/1
    # For small n: dN/dt ≈ -rn (linear approximation)
    # This means perturbations decay exponentially: n(t) ≈ n₀e^(-rt)
    
    # Show perturbation dynamics
    t_pert = np.linspace(0, 10, 200)
    perturbations = [5, 10, -5, -10]  # Different initial deviations from K
    
    for n0 in perturbations:
        # Exact solution starting at N = K + n0
        N_start = K + n0
        N_exact = K / (1 + ((K - N_start) / N_start) * np.exp(-r * t_pert))
        deviation_exact = N_exact - K
        
        # Linear approximation: n(t) ≈ n₀e^(-rt)
        deviation_linear = n0 * np.exp(-r * t_pert)
        
        color = COLORS['biological'] if n0 > 0 else COLORS['approximations'][3]
        ax2.plot(t_pert, deviation_exact, color=color, linewidth=2, 
                 label=f'n₀={n0:+d} (exact)' if n0 in [10, -10] else None)
        ax2.plot(t_pert, deviation_linear, color=color, linewidth=1.5,
                 linestyle='--', alpha=0.7,
                 label=f'n₀={n0:+d} (linear)' if n0 in [10, -10] else None)
    
    ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Deviation from equilibrium (N - K)')
    ax2.set_title('Stability Analysis: Taylor Linearization')
    ax2.legend(loc='upper right')
    
    ax2.annotate('Perturbations decay\nexponentially to equilibrium\n(stable fixed point)',
                xy=(5, 0), xytext=(6, 5),
                fontsize=9, ha='left',
                arrowprops=dict(arrowstyle='->', color='gray'))
    
    plt.tight_layout()
    return fig


def plot_multivariable_taylor() -> plt.Figure:
    """
    Visualize Taylor series for functions of two variables.
    
    BIOLOGICAL CONTEXT
    ------------------
    Many biological phenomena depend on multiple factors:
      - Gene expression depends on both genetic background and environment
      - Fitness depends on multiple traits simultaneously
      - Population dynamics can depend on multiple interacting species
    
    The multivariable Taylor series extends our "local to global" intuition:
    
    f(x,y) ≈ f(a,b) + fₓ(a,b)(x-a) + fᵧ(a,b)(y-b) 
             + ½[fₓₓ(a,b)(x-a)² + 2fₓᵧ(a,b)(x-a)(y-b) + fᵧᵧ(a,b)(y-b)²] + ...
    
    The first-order terms give a tangent PLANE (not line).
    The second-order terms create a quadratic surface that captures curvature.
    """
    # Define a 2D function: a "fitness landscape" with a peak
    def f(x, y):
        return np.exp(-0.5 * (x**2 + y**2)) * np.cos(0.5 * x) * np.cos(0.5 * y)
    
    # Expansion point (at the origin for simplicity)
    a, b = 0, 0
    
    # Create grid
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)
    
    # True function
    Z_true = f(X, Y)
    
    # Taylor approximations
    # At (0,0): f(0,0) = 1, ∂f/∂x = 0, ∂f/∂y = 0 (it's a peak)
    # Second derivatives at (0,0):
    # ∂²f/∂x² = -1 - 0.25 = -1.25 (numerical check shows ≈ -1)
    # ∂²f/∂y² = -1.25
    # ∂²f/∂x∂y = 0
    
    # Constant approximation
    Z_const = np.ones_like(X) * f(0, 0)
    
    # Linear approximation (same as constant here since gradient is zero at peak)
    Z_linear = Z_const  # ∂f/∂x = ∂f/∂y = 0 at origin
    
    # Quadratic approximation using numerical derivatives
    h = 1e-5
    fxx = (f(h, 0) - 2*f(0, 0) + f(-h, 0)) / h**2
    fyy = (f(0, h) - 2*f(0, 0) + f(0, -h)) / h**2
    fxy = (f(h, h) - f(h, -h) - f(-h, h) + f(-h, -h)) / (4*h**2)
    
    Z_quad = (f(0, 0) + 
              0.5 * fxx * X**2 + 
              fxy * X * Y + 
              0.5 * fyy * Y**2)
    
    # Create figure with 3D subplots
    fig = plt.figure(figsize=(15, 5))
    
    # Use a single row of 3D plots
    ax1 = fig.add_subplot(131, projection='3d')
    ax2 = fig.add_subplot(132, projection='3d')
    ax3 = fig.add_subplot(133, projection='3d')
    
    # Plot settings
    plot_kwargs = {'cmap': 'viridis', 'alpha': 0.8, 'rstride': 4, 'cstride': 4}
    
    # True function
    ax1.plot_surface(X, Y, Z_true, **plot_kwargs)
    ax1.set_title('True Function\n(Fitness Landscape)', fontsize=11)
    ax1.set_xlabel('x (Trait 1)')
    ax1.set_ylabel('y (Trait 2)')
    ax1.set_zlabel('Fitness')
    
    # Constant approximation
    ax2.plot_surface(X, Y, Z_const, cmap='Reds', alpha=0.6, rstride=4, cstride=4)
    ax2.plot_surface(X, Y, Z_true, cmap='viridis', alpha=0.3, rstride=8, cstride=8)
    ax2.set_title('Constant Approximation\n(Flat plane at peak value)', fontsize=11)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('f(x,y)')
    
    # Quadratic approximation
    Z_quad_clipped = np.clip(Z_quad, Z_true.min(), Z_true.max() + 0.5)
    ax3.plot_surface(X, Y, Z_quad_clipped, cmap='Oranges', alpha=0.7, rstride=4, cstride=4)
    ax3.plot_surface(X, Y, Z_true, cmap='viridis', alpha=0.3, rstride=8, cstride=8)
    ax3.set_title('Quadratic Approximation\n(Captures curvature at peak)', fontsize=11)
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_zlabel('f(x,y)')
    
    # Set consistent viewing angle
    for ax in [ax1, ax2, ax3]:
        ax.view_init(elev=25, azim=45)
        ax.set_zlim(-0.5, 1.2)
    
    plt.tight_layout()
    return fig


def plot_enzyme_kinetics() -> plt.Figure:
    """
    Michaelis-Menten kinetics: A classic biological application.
    
    THE BIOLOGY
    -----------
    Enzymes catalyze reactions. The rate v depends on substrate concentration [S]:
    
        v = Vmax·[S] / (Km + [S])
    
    where:
        Vmax = maximum reaction rate (all enzyme saturated)
        Km = Michaelis constant (substrate concentration at half-Vmax)
    
    TAYLOR SERIES APPLICATION
    -------------------------
    1. At LOW [S] (S << Km): v ≈ (Vmax/Km)·[S] — linear (first-order kinetics)
    2. At HIGH [S] (S >> Km): v ≈ Vmax — constant (zero-order kinetics)
    
    The Taylor series around [S]=0 reveals the first-order regime.
    This is why we use Taylor expansions in biochemistry!
    """
    # Parameters
    Vmax = 100   # Maximum velocity
    Km = 10      # Michaelis constant
    
    # Substrate concentration range
    S = np.linspace(0, 100, 500)
    
    # Michaelis-Menten equation
    def michaelis_menten(S):
        return Vmax * S / (Km + S)
    
    v_true = michaelis_menten(S)
    
    # Taylor expansion around S=0
    # v = Vmax·S/(Km+S)
    # At S=0: v(0) = 0
    # v'(S) = Vmax·Km/(Km+S)² → v'(0) = Vmax/Km
    # v''(S) = -2·Vmax·Km/(Km+S)³ → v''(0) = -2Vmax/Km²
    
    v_0 = 0
    v_prime_0 = Vmax / Km
    v_double_prime_0 = -2 * Vmax / Km**2
    
    # Taylor approximations
    v_linear = v_0 + v_prime_0 * S  # First-order kinetics
    v_quadratic = v_0 + v_prime_0 * S + (v_double_prime_0 / 2) * S**2
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # LEFT: Full kinetics
    ax1 = axes[0]
    ax1.plot(S, v_true, color=COLORS['biological'], linewidth=3, 
             label='Michaelis-Menten')
    ax1.plot(S, v_linear, color=COLORS['approximations'][0], linewidth=2,
             linestyle='--', label='Linear Taylor (first-order kinetics)')
    ax1.plot(S, np.clip(v_quadratic, 0, Vmax*1.1), color=COLORS['approximations'][1], 
             linewidth=2, linestyle='--', label='Quadratic Taylor')
    ax1.axhline(y=Vmax, color='gray', linestyle=':', alpha=0.7, label=f'Vmax = {Vmax}')
    ax1.axhline(y=Vmax/2, color='gray', linestyle=':', alpha=0.5)
    ax1.axvline(x=Km, color='gray', linestyle=':', alpha=0.5)
    
    ax1.set_xlabel('Substrate Concentration [S]')
    ax1.set_ylabel('Reaction Velocity v')
    ax1.set_title('Michaelis-Menten Kinetics and Taylor Approximations')
    ax1.legend(loc='lower right')
    ax1.set_ylim(0, Vmax * 1.2)
    
    # Annotate key points
    ax1.annotate(f'Km = {Km}\n(half-saturation)',
                xy=(Km, Vmax/2), xytext=(Km+15, Vmax/2-10),
                fontsize=9,
                arrowprops=dict(arrowstyle='->', color='gray'))
    ax1.annotate('First-order region:\nv ∝ [S]',
                xy=(3, michaelis_menten(3)), xytext=(15, 20),
                fontsize=9,
                arrowprops=dict(arrowstyle='->', color='gray'))
    ax1.annotate('Zero-order region:\nv ≈ Vmax',
                xy=(80, Vmax), xytext=(60, Vmax*0.7),
                fontsize=9,
                arrowprops=dict(arrowstyle='->', color='gray'))
    
    # RIGHT: Log-log plot showing regime transitions
    ax2 = axes[1]
    S_log = np.logspace(-1, 2, 500)
    v_true_log = michaelis_menten(S_log)
    v_linear_log = v_prime_0 * S_log
    
    ax2.loglog(S_log, v_true_log, color=COLORS['biological'], linewidth=3,
               label='Michaelis-Menten')
    ax2.loglog(S_log, v_linear_log, color=COLORS['approximations'][0], 
               linewidth=2, linestyle='--', label='First-order: v = (Vmax/Km)·[S]')
    ax2.axhline(y=Vmax, color='gray', linestyle=':', alpha=0.7, label='Zero-order: v = Vmax')
    ax2.axvline(x=Km, color=COLORS['highlight'], linestyle='-', alpha=0.5, linewidth=2)
    
    ax2.set_xlabel('Substrate Concentration [S] (log scale)')
    ax2.set_ylabel('Reaction Velocity v (log scale)')
    ax2.set_title('Log-Log Plot: Kinetic Regimes')
    ax2.legend(loc='lower right')
    ax2.set_ylim(0.1, Vmax * 1.5)
    
    # Annotate regimes
    ax2.annotate('Slope = 1\n(first-order)',
                xy=(1, michaelis_menten(1)), xytext=(0.3, 30),
                fontsize=9,
                arrowprops=dict(arrowstyle='->', color='gray'))
    ax2.annotate('Slope → 0\n(zero-order)',
                xy=(50, michaelis_menten(50)), xytext=(20, 50),
                fontsize=9,
                arrowprops=dict(arrowstyle='->', color='gray'))
    ax2.annotate('Transition\nat [S] = Km',
                xy=(Km, Vmax/2), xytext=(Km*0.1, Vmax*0.7),
                fontsize=9, color=COLORS['highlight'],
                arrowprops=dict(arrowstyle='->', color=COLORS['highlight']))
    
    plt.tight_layout()
    return fig


# =============================================================================
# SECTION 3: PUTTING IT ALL TOGETHER
# =============================================================================

def main():
    """
    Generate all visualizations demonstrating Taylor series concepts.
    
    OUTPUT
    ------
    Four publication-quality figures:
    1. Basic Taylor convergence for exp(x), sin(x), cos(x)
    2. Population dynamics application
    3. Multivariable Taylor (fitness landscape)
    4. Enzyme kinetics (Michaelis-Menten)
    """
    print("="*70)
    print("TAYLOR SERIES: VISUAL INTUITION FOR BIOLOGISTS")
    print("="*70)
    print()
    
    # -------------------------------------------------------------------------
    # FIGURE 1: Classic functions
    # -------------------------------------------------------------------------
    print("Figure 1: Taylor series convergence for classic functions")
    print("-" * 60)
    
    # Define test functions and their derivatives
    # Exponential: all derivatives equal the function itself
    exp_derivs = [np.exp] * 10
    
    # Sine: derivatives cycle sin → cos → -sin → -cos → sin ...
    sin_derivs = [np.cos, lambda x: -np.sin(x), lambda x: -np.cos(x), 
                  np.sin, np.cos, lambda x: -np.sin(x)]
    
    # Cosine: derivatives cycle cos → -sin → -cos → sin → cos ...
    cos_derivs = [lambda x: -np.sin(x), lambda x: -np.cos(x),
                  np.sin, np.cos, lambda x: -np.sin(x), lambda x: -np.cos(x)]
    
    fig1 = plot_taylor_convergence(
        f=np.exp, derivatives=exp_derivs, f_name='exp(x)',
        a=0, x_range=(-2, 3), max_terms=6
    )
    fig1.savefig('taylor_01_exponential.png', dpi=150, bbox_inches='tight',
                 facecolor='white', edgecolor='none')
    print("  → Saved: taylor_01_exponential.png")
    
    fig1b = plot_taylor_convergence(
        f=np.sin, derivatives=sin_derivs, f_name='sin(x)',
        a=0, x_range=(-2*np.pi, 2*np.pi), max_terms=6
    )
    fig1b.savefig('taylor_02_sine.png', dpi=150, bbox_inches='tight',
                  facecolor='white', edgecolor='none')
    print("  → Saved: taylor_02_sine.png")
    
    # -------------------------------------------------------------------------
    # FIGURE 2: Population dynamics
    # -------------------------------------------------------------------------
    print()
    print("Figure 2: Biological application - Population dynamics")
    print("-" * 60)
    
    fig2 = plot_biological_application_population()
    fig2.savefig('taylor_03_population.png', dpi=150, bbox_inches='tight',
                 facecolor='white', edgecolor='none')
    print("  → Saved: taylor_03_population.png")
    
    # -------------------------------------------------------------------------
    # FIGURE 3: Multivariable Taylor (fitness landscape)
    # -------------------------------------------------------------------------
    print()
    print("Figure 3: Multivariable Taylor series - Fitness landscape")
    print("-" * 60)
    
    fig3 = plot_multivariable_taylor()
    fig3.savefig('taylor_04_multivariable.png', dpi=150, bbox_inches='tight',
                 facecolor='white', edgecolor='none')
    print("  → Saved: taylor_04_multivariable.png")
    
    # -------------------------------------------------------------------------
    # FIGURE 4: Enzyme kinetics
    # -------------------------------------------------------------------------
    print()
    print("Figure 4: Michaelis-Menten enzyme kinetics")
    print("-" * 60)
    
    fig4 = plot_enzyme_kinetics()
    fig4.savefig('taylor_05_enzyme.png', dpi=150, bbox_inches='tight',
                 facecolor='white', edgecolor='none')
    print("  → Saved: taylor_05_enzyme.png")
    
    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print()
    print("="*70)
    print("KEY TAKEAWAYS")
    print("="*70)
    print("""
    1. TAYLOR SERIES = LOCAL INFORMATION → GLOBAL APPROXIMATION
       Know the derivatives at one point, reconstruct the function nearby.
    
    2. MORE TERMS = BETTER APPROXIMATION
       But accuracy still degrades as you move far from the expansion point.
    
    3. BIOLOGICAL POWER:
       - Linearization: Simplify nonlinear dynamics near equilibria
       - Stability analysis: Determine if systems return to equilibrium
       - Regime identification: First-order vs zero-order kinetics
       - Fitness landscapes: Quadratic approximation captures local shape
    
    4. THE FACTORIAL (n!) IS NOT ARBITRARY
       It normalizes the terms so that derivatives match exactly at the
       expansion point. This is how the series is DERIVED.
    
    5. CONVERGENCE MATTERS
       Taylor series don't always converge everywhere. In biology,
       we typically care about local behavior, where convergence is assured.
    """)
    
    plt.show()


if __name__ == "__main__":
    main()
