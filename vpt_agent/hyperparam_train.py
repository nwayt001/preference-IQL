from behavioural_cloning import behavioural_cloning_train

if __name__ == "__main__":

    # grid search over different KL loss weights
    kl_weights = [1, 0.8, 0.6, 0.4, 0.2, 0.0]
    
    for kl_w in kl_weights:
        # train from the bc house fine tunred model
        print("===Training BuildVillageHouse model KL {}===".format(kl_w))
        behavioural_cloning_train(
            data_dir="data/MineRLBasaltBuildVillageHouse-v0",
            in_model="data/VPT-models/foundation-model-3x.model",
            in_weights="data/VPT-models/bc-house-3x.weights",
            out_weights="train/MineRLBasaltBuildVillageHouse-model-BC_HOUSE_KL_{}.weights".format(kl_w),
        )

