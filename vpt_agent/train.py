# Train one model for each task and for each model type
from behavioural_cloning import behavioural_cloning_train

if __name__ == "__main__":
    
    # train the model for each task
    model_types = ["3x", "2x", "1x"]

    # Train 1x, 2x and 3x model
    for model in model_types:
        print("===Training FindCave model - {} ===".format(model))
        behavioural_cloning_train(
            data_dir="data/MineRLBasaltFindCave-v0",
            in_model="data/VPT-models/foundation-model-{}.model".format(model),
            in_weights="data/VPT-models/foundation-model-{}.weights".format(model),
            out_weights="train/MineRLBasaltFindCave-model-{}.weights".format(model),
        )

        print("===Training MakeWaterfall model - {}===".format(model))
        behavioural_cloning_train(
            data_dir="data/MineRLBasaltMakeWaterfall-v0",
            in_model="data/VPT-models/foundation-model-{}.model".format(model),
            in_weights="data/VPT-models/foundation-model-{}.weights".format(model),
            out_weights="train/MineRLBasaltMakeWaterfall-model-{}.weights".format(model),
        )

        print("===Training CreateVillageAnimalPen model - {}===".format(model))
        behavioural_cloning_train(
            data_dir="data/MineRLBasaltCreateVillageAnimalPen-v0",
            in_model="data/VPT-models/foundation-model-{}.model".format(model),
            in_weights="data/VPT-models/foundation-model-{}.weights".format(model),
            out_weights="train/MineRLBasaltCreateVillageAnimalPen-model-{}.weights".format(model),
        )
        
        print("===Training BuildVillageHouse model - {}===".format(model))
        behavioural_cloning_train(
            data_dir="data/MineRLBasaltBuildVillageHouse-v0",
            in_model="data/VPT-models/foundation-model-{}.model".format(model),
            in_weights="data/VPT-models/foundation-model-{}.weights".format(model),
            out_weights="train/MineRLBasaltBuildVillageHouse-model-{}.weights".format(model),
        )

    