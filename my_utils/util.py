import numpy as np

def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

    @classmethod
    def recursive_init(cls, myDict):
        """
        recursively look into a dictionary and convert each sub_dictionary entry to AttrDict
        This is a little bit messy
        """
#        myDict = copy.deepcopy(myDict)

        def _rec_into_subdict(curr_dict):
            for key, entry in curr_dict.items():
                if type(entry) is dict:
                    _rec_into_subdict(entry)
                    curr_dict[key] = cls(entry)

        _rec_into_subdict(myDict)
        return cls(myDict)


