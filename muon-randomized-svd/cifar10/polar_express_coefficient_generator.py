from polar_express import PolarExpress, optimal_composition


my_coeffs = optimal_composition(
    l=1e-3,
    num_iters=9,                   
    safety_factor_eps=0.0,      
    cushion=0.02407327424182761, 
)

print(my_coeffs)