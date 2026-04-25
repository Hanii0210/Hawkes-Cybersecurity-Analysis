import numpy as np

def simulate_hawkes(mu, alpha, beta, T, rng=None):
    if rng is None:
        rng = np.random.default_rng()
        
    events = []
    t = 0.0
    g = 0.0
    while t < T:
        lambda_bar = mu + alpha * g
        dt = rng.exponential(1.0 / lambda_bar)
        t += dt
        if t > T: break
        g *= np.exp(-beta * dt)
        if rng.random() < (mu + alpha * g) / lambda_bar:
            events.append(t)
            g += 1.0
    return np.array(events)

def generate_mixture_data(T_mix=500, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
        
    e1 = simulate_hawkes(mu=0.3, alpha=0.2, beta=1.0, T=T_mix, rng=rng)
    l1 = np.zeros(len(e1), dtype=int)
    
    e2 = simulate_hawkes(mu=1.5, alpha=0.6, beta=2.0, T=T_mix, rng=rng)
    l2 = np.ones(len(e2), dtype=int)
    
    all_events = np.concatenate([e1, e2])
    all_labels = np.concatenate([l1, l2])
    order = np.argsort(all_events)
    
    return all_events[order], all_labels[order]