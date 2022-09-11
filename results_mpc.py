from do_mpc.data import load_results

results = load_results('./results/results.pkl')

ysults = results['mpc']['_u']
print(ysults)