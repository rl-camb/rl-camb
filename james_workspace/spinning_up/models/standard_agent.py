import os
import pickle
import pprint

class StandardAgent():

    def __init__(self, experiment_dir):

        self.experiment_dir = "saved_models/" + experiment_dir
        os.makedirs(self.experiment_dir, exist_ok=True)
        self.model_location = self.experiment_dir + "model.h5"
        self.model_dict_file = os.path.join(
            self.experiment_dir, "model_dict.pickle")

        self.load_model()

        if not self.model_dict:
            self.scores = []
        else:
            self.scores = self.model_dict["scores"]

    def save_dict(self):
        if os.path.exists(self.model_dict_file):
            with open(self.model_dict_file, 'rb') as md:
                model_dict = pickle.load(md)
        else:
            model_dict = {"model_location": self.model_location}

        for key in ("scores",):
            if hasattr(self, key):
                model_dict[key] = getattr(self, key)

        with open(self.model_dict_file, 'wb') as md:
            pickle.dump(model_dict, md)
    
    def load_dict(self):
        if os.path.exists(self.model_dict_file):
            with open(self.model_dict_file, 'rb') as md:
                pickled_details = pickle.load(md)
            self.model_dict = pickled_details
        else:
            print("No model dict exists yet!")
            self.model_dict = {"scores": []}

    def view_models_dict(self, view=False):
        """Open the model dict to view what models we have."""
        with open(self.models_dict_file, 'rb') as md:
            model_dict = pickle.load(md)

        if view:
            pprint.pprint(model_dict)

        return model_dict
