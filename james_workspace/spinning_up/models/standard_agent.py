import os
import pickle
import pprint

class StandardAgent():

    def __init__(self, experiment_dir):

        self.experiment_dir = "saved_models/" + experiment_dir + "/"
        os.makedirs(self.experiment_dir, exist_ok=True)
        self.model_location = self.experiment_dir + "model.h5"
        self.model_dict_file = os.path.join(
            self.experiment_dir, "model_dict.pickle")

        self.load_model()
        print("Loaded model:")
        self.view_models_dict()

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

        # Update values
        for key in ("scores",):
            if hasattr(self, key):
                model_dict[key] = getattr(self, key)

        if hasattr(self, "scores"):
            model_dict["trained_episodes"] = len(self.scores)

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

    def view_models_dict(self):
        """Open the model dict to view what models we have."""
        if os.path.exists(self.model_dict_file):
            with open(self.model_dict_file, 'rb') as md:
                model_dict = pickle.load(md)

            pprint.pprint(model_dict, depth=1)
        else:
            print("Model dict file does not exist for viewing, yet")
            model_dict = {}

        return model_dict
