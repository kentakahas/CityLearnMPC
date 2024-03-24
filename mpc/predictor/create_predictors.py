import sys
from lgb_predictor import LGBPredictor

if __name__ == "__main__":
    building_index = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 16, 17]
    buildings = [f"Building_{b}" for b in building_index]
    lookback = 12
    horizon = 24
    features = ['non_shiftable_load', 'solar_generation']

    for building in buildings:
        for feature in features:
            sys.stdout.write(f'Starting training for {building}: {feature}\n')
            sys.stdout.flush()
            model = LGBPredictor(building, feature, lookback, horizon)
            model.search(n_iter=3, n_jobs=3)
            model.save()

    sys.stdout.write("Training complete")
    sys.stdout.flush()
