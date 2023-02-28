# from sc_model import build_state_classifier_model
import sc_model
import sc_data_generator
# from sc_data_generator import StateClassifierDataGenerator


train_data_generator = sc_data_generator.StateClassifierDataGenerator("MineRLBasaltBuildVillageHouse-v0", is_validation=False)
validation_data_generator = sc_data_generator.StateClassifierDataGenerator("MineRLBasaltBuildVillageHouse-v0", is_validation=True)
print(train_data_generator.n_classes)

model = sc_model.build_state_classifier_model(train_data_generator.n_classes)

x, y = train_data_generator[0]
pred = model.predict(x)
print(model.summary())
# print(pred)
# print(y)
# exit()

model.fit_generator(
    generator=train_data_generator,
    validation_data=validation_data_generator,
    use_multiprocessing=False,
    workers=6,
)
