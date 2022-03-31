probability = 0.92641234
probhex = hex(int(probability*100))
probround = round(100*probability)

print(2**10*"–")

print('\nPrediction is "…" ({:.0f}%)\t expected "…"'.format(100 * probability))

print(probround)