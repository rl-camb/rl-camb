import os
import pickle

class StandardAgent():

    def __init__(self, experiment_dir):

        self.experiment_dir = "saved_models" + os.sep + experiment_dir + os.sep
        self.model_location = self.experiment_dir + "model.h5"
        self.dict_location =  self.experiment_dir + "status.p"

        os.makedirs(self.experiment_dir, exist_ok=True)

    def save_state_to_dict(self, append_dict={}):

        model_dict = {}

        for key in ("model_location", "scores", "total_t"):
            model_dict[key] = getattr(self, key)

        model_dict["trained_episodes"] = len(self.scores)

        for k, v in append_dict.items():
            model_dict[k] = v

        with open(self.dict_location, 'wb') as md:
            pickle.dump(model_dict, md)

        return model_dict

    def load_state_from_dict(self):

        if os.path.exists(self.dict_location):
            with open(self.dict_location, 'rb') as md:
                model_dict = pickle.load(md)
        else:
            print("No model dict exists yet!")
            model_dict = {}

        # Initialise standard state if empty, else flexible
        self.scores = model_dict.get("scores", [])
        self.total_t = model_dict.get("total_t", 0)
        for k, v in model_dict.items():
            if k in ("scores", "total_t"):
                continue 
            else:
                setattr(self, k, v)

        return model_dict

    def return_state_dict(self):
        """Open the model dict to view what models we have."""
        
        if os.path.exists(self.model_dict_file):
            with open(self.model_dict_file, 'rb') as md:
                model_dict = pickle.load(md)
        else:
            print("Model dict file does not exist for viewing, yet")
            model_dict = None

        return model_dict

    def load_state(self):
        raise NotImplementedError(
            "To be implemented by the inheriting agent.")