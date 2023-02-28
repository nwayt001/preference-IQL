import tkinter
import tkinter.ttk

import numpy as np

data_labels = [
    # General
    ("facing_wall", "Facing Wall"),
    ("danger_ahead", "Danger Ahead"),
    ("has_open_space", "Has Open Space"),
    ("exploring", "Player is Exploring"),
    ("moving_to_goal", "Player is moving to goal"),
    ("inventory_open", "Inventory is open"),

    # find_cave task
    ("has_cave", "Has Cave"),
    ("inside_cave", "Inside Cave"),

    # make_waterfall task
    ("has_mountain", "Has Mountain"),
    ("at_top_of_mountain", "At Top of Mountain"),
    ("good_waterfall_view", "Good Waterfall View"),

    # build_animal_pen tasks
    ("good_pen_view", "Good View of Pen"),
    ("has_carrot_animal", "Has Horse, Pig, or Rabbit"),
    ("has_wheat_animal", "Has Sheep, Cow, or Mooshroom"),
    ("has_seed_animal", "Has Chicken"),

    # build_village_house task items
    ("good_house_view", "Good View of House"),
    ("good_house_location", "Good Location for House"),
    ("inside_house", "Inside House"),
    ("building_house", "Player is building house"),

    ("in_desert_biome", "In Desert Biome"),
    ("in_plains_biome", "In Plains Biome"),
    ("in_savanna_biome", "In Savanna Biome"),
    ("in_taiga_biome", "In Taiga Biome"),
    ("in_snowy_biome", "In Snowy Biome"),
]


class DataLabelerFrame(tkinter.ttk.Frame):

    def __init__(self):
        super(DataLabelerFrame, self).__init__()
        self.master.title("KAIROS Data Labeler")
        self.style = tkinter.ttk.Style()
        self.style.theme_use("default")

        frame = tkinter.ttk.Frame(self, relief=tkinter.RAISED, borderwidth=1)
        frame.pack(fill=tkinter.BOTH, expand=True)

        self.pack(fill=tkinter.BOTH, expand=True)

        # Buttons
        next_button = tkinter.Button(self, text="Next")
        next_button.pack(side=tkinter.RIGHT, padx=5, pady=5)
        previous_button = tkinter.Button(self, text="Previous")
        previous_button.pack(side=tkinter.RIGHT, padx=5, pady=5)

        # Checklist
        options = np.zeros((len(data_labels),))
        for button_index, (name, text) in enumerate(data_labels):
            c = tkinter.Checkbutton(frame, text=text, variable=options[button_index], onvalue=1, offvalue=0)
            c.pack()

        # Title

        # Images


def main():
    root = tkinter.Tk()
    root.geometry("1200x800+300+300")
    app = DataLabelerFrame()
    root.mainloop()


if __name__ == "__main__":
    main()

